// backends/cuda/nbody_cuda.cu — Owner: Shadhin. Contract v1.
// --variant <kernel>-<precision>: kernel = naive | tiled | copystep, precision = f64 | f32
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>
#include <cuda_runtime.h>
#include "nbody_io.hpp"

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err_ = (call);                                                       \
        if (err_ != cudaSuccess) {                                                       \
            std::fprintf(stderr, "CUDA error '%s' at %s:%d\n", cudaGetErrorString(err_), \
                         __FILE__, __LINE__);                                            \
            std::exit(1);                                                                \
        }                                                                                \
    } while (0)

template <typename T> struct Vec4;
template <> struct Vec4<float>  { using type = float4;  };
template <> struct Vec4<double> { using type = double4; };

__device__ __forceinline__ float  rsqrt_t(float v)  { return rsqrtf(v); }
__device__ __forceinline__ double rsqrt_t(double v) { return rsqrt(v); }

// pm[i] = (x, y, z, mass). acc is interleaved: ax0, ay0, az0, ax1, ...
template <typename T>
__global__ void accel_naive(const typename Vec4<T>::type* __restrict__ pm,
                            T* __restrict__ acc, int n, T g) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    const auto me = pm[p];
    T ax = 0, ay = 0, az = 0;
    for (int j = 0; j < n; ++j) {
        if (j == p) continue;                       // contract: softening 0, skip self by index
        const auto o = pm[j];
        const T dx = o.x - me.x, dy = o.y - me.y, dz = o.z - me.z;
        const T inv = rsqrt_t(dx * dx + dy * dy + dz * dz);
        const T s = o.w * inv * inv * inv;
        ax += s * dx; ay += s * dy; az += s * dz;
    }
    acc[3 * p] = g * ax; acc[3 * p + 1] = g * ay; acc[3 * p + 2] = g * az;
}

template <typename T>
__global__ void accel_tiled(const typename Vec4<T>::type* __restrict__ pm,
                            T* __restrict__ acc, int n, T g) {
    using V = typename Vec4<T>::type;
    // One untyped shared buffer: typed `extern __shared__` arrays clash between template instantiations.
    extern __shared__ __align__(16) unsigned char smem_raw[];
    V* tile = reinterpret_cast<V*>(smem_raw);

    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    const V me = (p < n) ? pm[p] : V{0, 0, 0, 0};   // out-of-range threads still help load tiles
    T ax = 0, ay = 0, az = 0;

    for (int base = 0; base < n; base += blockDim.x) {
        const int j = base + threadIdx.x;
        tile[threadIdx.x] = (j < n) ? pm[j] : V{0, 0, 0, 0};
        __syncthreads();
        const int lim = min(static_cast<int>(blockDim.x), n - base);
        #pragma unroll 4
        for (int k = 0; k < lim; ++k) {
            if (base + k == p) continue;
            const V o = tile[k];
            const T dx = o.x - me.x, dy = o.y - me.y, dz = o.z - me.z;
            const T inv = rsqrt_t(dx * dx + dy * dy + dz * dz);
            const T s = o.w * inv * inv * inv;
            ax += s * dx; ay += s * dy; az += s * dz;
        }
        __syncthreads();                            // don't overwrite the tile while others read it
    }
    if (p < n) { acc[3 * p] = g * ax; acc[3 * p + 1] = g * ay; acc[3 * p + 2] = g * az; }
}

template <typename T>
__global__ void drift(typename Vec4<T>::type* __restrict__ pm, const T* __restrict__ vel,
                      const T* __restrict__ acc, int n, T dt) {
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    const T h = T(0.5) * dt * dt;
    pm[p].x += vel[3 * p] * dt + acc[3 * p] * h;
    pm[p].y += vel[3 * p + 1] * dt + acc[3 * p + 1] * h;
    pm[p].z += vel[3 * p + 2] * dt + acc[3 * p + 2] * h;
}

template <typename T>
__global__ void kick(T* __restrict__ vel, const T* __restrict__ a0, const T* __restrict__ a1,
                     int n3, T dt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n3) vel[i] += T(0.5) * (a0[i] + a1[i]) * dt;
}

template <typename T>
int run(const nbio::Args& a, const nbio::System& s, const std::string& kernel, const std::string& prec) {
    using V = typename Vec4<T>::type;
    const int n = s.n(), B = a.block_size;
    const int grid = (n + B - 1) / B, grid3 = (3 * n + B - 1) / B;
    const size_t shmem = static_cast<size_t>(B) * sizeof(V);
    const bool tiled = (kernel != "naive"), copy_each_step = (kernel == "copystep");

    std::vector<V> h_pm(n);
    std::vector<T> h_vel(3 * n);
    for (int i = 0; i < n; ++i) {
        h_pm[i] = V{T(s.x[i]), T(s.y[i]), T(s.z[i]), T(s.m[i])};
        h_vel[3 * i] = T(s.vx[i]); h_vel[3 * i + 1] = T(s.vy[i]); h_vel[3 * i + 2] = T(s.vz[i]);
    }

    nbio::Timings t;
    double t0 = nbio::now();
    CUDA_CHECK(cudaFree(0));                        // create the CUDA context inside setup
    V* d_pm; T *d_vel, *d_a, *d_b;
    CUDA_CHECK(cudaMalloc(&d_pm, n * sizeof(V)));
    CUDA_CHECK(cudaMalloc(&d_vel, 3 * n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_a, 3 * n * sizeof(T)));
    CUDA_CHECK(cudaMalloc(&d_b, 3 * n * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d_pm, h_pm.data(), n * sizeof(V), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel, h_vel.data(), 3 * n * sizeof(T), cudaMemcpyHostToDevice));

    const T g = T(nbio::G), dt = T(a.dt);
    auto accel = [&](T* out) {
        if (tiled) accel_tiled<T><<<grid, B, shmem>>>(d_pm, out, n, g);
        else       accel_naive<T><<<grid, B>>>(d_pm, out, n, g);
    };
    accel(d_a);                                     // a0 (also warms up the kernel)
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    t.setup = nbio::now() - t0;

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventRecord(ev0));
    for (int step = 0; step < a.steps; ++step) {
        drift<T><<<grid, B>>>(d_pm, d_vel, d_a, n, dt);
        if (copy_each_step) {                       // deliberately bad: host round-trip every step
            CUDA_CHECK(cudaMemcpy(h_pm.data(), d_pm, n * sizeof(V), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(d_pm, h_pm.data(), n * sizeof(V), cudaMemcpyHostToDevice));
        }
        accel(d_b);
        kick<T><<<grid3, B>>>(d_vel, d_a, d_b, 3 * n, dt);
        std::swap(d_a, d_b);
    }
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));         // kernels are async: wait before reading the clock
    CUDA_CHECK(cudaGetLastError());
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    t.loop = ms / 1000.0;

    t0 = nbio::now();
    CUDA_CHECK(cudaMemcpy(h_pm.data(), d_pm, n * sizeof(V), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vel.data(), d_vel, 3 * n * sizeof(T), cudaMemcpyDeviceToHost));
    t.transfer = nbio::now() - t0;

    if (!a.final_state.empty()) {
        nbio::System out = s;
        for (int i = 0; i < n; ++i) {
            out.x[i] = h_pm[i].x; out.y[i] = h_pm[i].y; out.z[i] = h_pm[i].z;
            out.vx[i] = h_vel[3 * i]; out.vy[i] = h_vel[3 * i + 1]; out.vz[i] = h_vel[3 * i + 2];
        }
        nbio::write_state_csv(a.final_state, out);
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    char extra[512];
    std::snprintf(extra, sizeof extra,
                  "{\"kernel\": \"%s\", \"precision\": \"%s\", \"block_size\": %d, "
                  "\"gpu\": \"%s\", \"sm\": \"%d.%d\"}",
                  kernel.c_str(), prec.c_str(), B, prop.name, prop.major, prop.minor);
    nbio::write_result_json(a, "cuda", "cuda", n, 1, t, extra);

    std::printf("cuda[%s] N=%d B=%d steps=%d loop=%.4fs (%.3f ms/step) on %s\n", a.variant.c_str(),
                n, B, a.steps, t.loop, 1e3 * t.loop / a.steps, prop.name);

    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    cudaFree(d_pm); cudaFree(d_vel); cudaFree(d_a); cudaFree(d_b);
    return 0;
}

int main(int argc, char** argv) {
    nbio::Args a = nbio::parse_args(argc, argv);
    if (a.variant == "default") a.variant = "tiled-f64";
    const auto dash = a.variant.find('-');
    const std::string kernel = a.variant.substr(0, dash);
    const std::string prec = (dash == std::string::npos) ? "f64" : a.variant.substr(dash + 1);
    if ((kernel != "naive" && kernel != "tiled" && kernel != "copystep") ||
        (prec != "f64" && prec != "f32") || a.block_size < 32 || a.block_size > 1024) {
        std::fprintf(stderr, "bad --variant %s or --block-size %d\n", a.variant.c_str(), a.block_size);
        return 2;
    }
    const nbio::System s = nbio::read_ic_csv(a.input);     // not timed (§C5)
    return prec == "f64" ? run<double>(a, s, kernel, prec) : run<float>(a, s, kernel, prec);
}
