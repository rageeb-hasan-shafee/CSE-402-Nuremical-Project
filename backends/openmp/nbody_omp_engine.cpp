// backends/openmp/nbody_omp_engine.cpp — High-Performance Native Live Engine
// Exposes the team's C++ OpenMP implementations via a portable C API.
// Statically linked (-static) with zero external runtime dependencies on any Windows machine.
//
// Evaluated Team Velocity Verlet Variants:
// Mode 0: newton3 (Velocity Verlet, Newton's 3rd Law Pairwise Halving)
// Mode 1: simd    (Velocity Verlet, AVX2-SIMD Vectorized Branchless Loop)
// Mode 2: dynamic (Velocity Verlet, Dynamic Work-Stealing)
// Mode 3: static  (Velocity Verlet, Static Scheduling)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

// Gravitational constant in AU^3 Msun^-1 day^-2 (Contract v1)
static const double G_CONST = 2.9591220828559115e-4;

using Vec = std::vector<double>;

// Internal system structure
struct LiveSystem {
    int n;
    Vec m;
    Vec x, y, z;
    Vec vx, vy, vz;
};

// 1. Newton's 3rd Law: pairwise halving using a_ij = -a_ji with dynamic thread buffers
static void accel_newton3(const LiveSystem& s, Vec& ax, Vec& ay, Vec& az) {
    const int n = s.n;
    const double *x = s.x.data(), *y = s.y.data(), *z = s.z.data(), *m = s.m.data();
    std::fill(ax.begin(), ax.end(), 0.0);
    std::fill(ay.begin(), ay.end(), 0.0);
    std::fill(az.begin(), az.end(), 0.0);
    #pragma omp parallel
    {
        Vec lx(n, 0.0), ly(n, 0.0), lz(n, 0.0);
        #pragma omp for schedule(dynamic, 16)
        for (int p = 0; p < n; ++p) {
            for (int j = p + 1; j < n; ++j) {
                const double dx = x[j] - x[p], dy = y[j] - y[p], dz = z[j] - z[p];
                const double r2 = dx * dx + dy * dy + dz * dz;
                if (r2 <= 0.0) continue;
                const double inv3 = G_CONST / (r2 * std::sqrt(r2));
                const double fp = m[j] * inv3, fj = m[p] * inv3;
                lx[p] += fp * dx; ly[p] += fp * dy; lz[p] += fp * dz;
                lx[j] -= fj * dx; ly[j] -= fj * dy; lz[j] -= fj * dz;
            }
        }
        #pragma omp critical
        for (int i = 0; i < n; ++i) { ax[i] += lx[i]; ay[i] += ly[i]; az[i] += lz[i]; }
    }
}

// 2. AVX2-SIMD Vectorization (team variant "simd")
static inline void accum_range(const double* x, const double* y, const double* z, const double* m,
                               int j0, int j1, double xp, double yp, double zp,
                               double& sx, double& sy, double& sz) {
    double ax = 0.0, ay = 0.0, az = 0.0;
    #pragma omp simd reduction(+ : ax, ay, az)
    for (int j = j0; j < j1; ++j) {
        const double dx = x[j] - xp, dy = y[j] - yp, dz = z[j] - zp;
        const double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 > 0.0) {
            const double f = m[j] / (r2 * std::sqrt(r2));
            ax += f * dx; ay += f * dy; az += f * dz;
        }
    }
    sx += ax; sy += ay; sz += az;
}

static void accel_simd(const LiveSystem& s, Vec& ax, Vec& ay, Vec& az) {
    const int n = s.n;
    const double *x = s.x.data(), *y = s.y.data(), *z = s.z.data(), *m = s.m.data();
    #pragma omp parallel for schedule(static)
    for (int p = 0; p < n; ++p) {
        double sx = 0.0, sy = 0.0, sz = 0.0;
        accum_range(x, y, z, m, 0, p, x[p], y[p], z[p], sx, sy, sz);
        accum_range(x, y, z, m, p + 1, n, x[p], y[p], z[p], sx, sy, sz);
        ax[p] = G_CONST * sx; ay[p] = G_CONST * sy; az[p] = G_CONST * sz;
    }
}

// 3. Basic loop (team variants "static" and "dynamic")
static void accel_basic(const LiveSystem& s, Vec& ax, Vec& ay, Vec& az) {
    const int n = s.n;
    const double *x = s.x.data(), *y = s.y.data(), *z = s.z.data(), *m = s.m.data();
    #pragma omp parallel for schedule(runtime)
    for (int p = 0; p < n; ++p) {
        const double xp = x[p], yp = y[p], zp = z[p];
        double sx = 0.0, sy = 0.0, sz = 0.0;
        for (int j = 0; j < n; ++j) {
            if (j == p) continue;
            const double dx = x[j] - xp, dy = y[j] - yp, dz = z[j] - zp;
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 <= 0.0) continue;
            const double f = G_CONST * m[j] / (r2 * std::sqrt(r2));
            sx += f * dx; sy += f * dy; sz += f * dz;
        }
        ax[p] = sx; ay[p] = sy; az[p] = sz;
    }
}

// Total energy for relative drift computation
static double compute_total_energy(const LiveSystem& s) {
    double kinetic = 0.0;
    #pragma omp parallel for reduction(+ : kinetic) schedule(static)
    for (int i = 0; i < s.n; ++i) {
        kinetic += 0.5 * s.m[i] * (s.vx[i] * s.vx[i] + s.vy[i] * s.vy[i] + s.vz[i] * s.vz[i]);
    }

    double potential = 0.0;
    #pragma omp parallel for reduction(+ : potential) schedule(dynamic, 16)
    for (int i = 0; i < s.n; ++i) {
        double p_i = 0.0;
        for (int j = i + 1; j < s.n; ++j) {
            const double dx = s.x[j] - s.x[i], dy = s.y[j] - s.y[i], dz = s.z[j] - s.z[i];
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
            if (dist > 0.0) p_i -= (G_CONST * s.m[i] * s.m[j]) / dist;
        }
        potential += p_i;
    }
    return kinetic + potential;
}

extern "C" {

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

EXPORT int get_max_threads() {
#ifdef _OPENMP
    return omp_get_num_procs();
#else
    return 1;
#endif
}

// Simulates a batch of steps using our team's C++ OpenMP Velocity Verlet implementation
// variant_id:
//   0 = newton3 (Velocity Verlet, Newton's 3rd Law Pairwise Halving)
//   1 = simd    (Velocity Verlet, AVX2-SIMD Vectorized)
//   2 = dynamic (Velocity Verlet, Dynamic Work-Stealing)
//   3 = static  (Velocity Verlet, Static Scheduling)
EXPORT int run_openmp_batch(
    int n,
    const double* masses,
    double* pos,           // inout: [x0, y0, z0, ...] (length 3*n)
    double* vel,           // inout: [vx0, vy0, vz0, ...] (length 3*n)
    double dt,
    int steps,
    int variant_id,
    int num_threads,
    double* out_positions,  // buffer: steps * 3 * n (optional, can be null)
    double* out_velocities, // buffer: steps * 3 * n (optional, can be null)
    double* out_drifts,     // buffer: steps (optional, can be null)
    double* out_elapsed_ms  // returns execution time in ms
) {
    if (n <= 0 || steps <= 0) return -1;

#ifdef _OPENMP
    int max_t = omp_get_num_procs();
    int threads = (num_threads > 0 && num_threads <= max_t) ? num_threads : max_t;
    omp_set_num_threads(threads);
    if (variant_id == 2) omp_set_schedule(omp_sched_dynamic, 16);
    else omp_set_schedule(omp_sched_static, 0);
#endif

    LiveSystem s;
    s.n = n;
    s.m.assign(masses, masses + n);
    s.x.resize(n); s.y.resize(n); s.z.resize(n);
    s.vx.resize(n); s.vy.resize(n); s.vz.resize(n);

    for (int i = 0; i < n; ++i) {
        s.x[i] = pos[3 * i + 0];
        s.y[i] = pos[3 * i + 1];
        s.z[i] = pos[3 * i + 2];
        s.vx[i] = vel[3 * i + 0];
        s.vy[i] = vel[3 * i + 1];
        s.vz[i] = vel[3 * i + 2];
    }

    auto accel = accel_newton3;
    if (variant_id == 1) accel = accel_simd;
    else if (variant_id == 2 || variant_id == 3) accel = accel_basic;

    double e0 = compute_total_energy(s);

    Vec ax(n), ay(n), az(n), bx(n), by(n), bz(n);
    accel(s, ax, ay, az);

    const double half_dt2 = 0.5 * dt * dt;
    const double half_dt = 0.5 * dt;

    auto t_start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < steps; ++step) {
        // Symplectic Velocity Verlet
        #pragma omp parallel for schedule(static)
        for (int p = 0; p < n; ++p) {
            s.x[p] += s.vx[p] * dt + ax[p] * half_dt2;
            s.y[p] += s.vy[p] * dt + ay[p] * half_dt2;
            s.z[p] += s.vz[p] * dt + az[p] * half_dt2;
        }

        accel(s, bx, by, bz);

        #pragma omp parallel for schedule(static)
        for (int p = 0; p < n; ++p) {
            s.vx[p] += (ax[p] + bx[p]) * half_dt;
            s.vy[p] += (ay[p] + by[p]) * half_dt;
            s.vz[p] += (az[p] + bz[p]) * half_dt;
        }

        std::swap(ax, bx);
        std::swap(ay, by);
        std::swap(az, bz);

        // Record trajectory positions
        if (out_positions) {
            for (int i = 0; i < n; ++i) {
                out_positions[step * 3 * n + 3 * i + 0] = s.x[i];
                out_positions[step * 3 * n + 3 * i + 1] = s.y[i];
                out_positions[step * 3 * n + 3 * i + 2] = s.z[i];
            }
        }

        // Record trajectory velocities
        if (out_velocities) {
            for (int i = 0; i < n; ++i) {
                out_velocities[step * 3 * n + 3 * i + 0] = s.vx[i];
                out_velocities[step * 3 * n + 3 * i + 1] = s.vy[i];
                out_velocities[step * 3 * n + 3 * i + 2] = s.vz[i];
            }
        }

        // Record energy drift
        if (out_drifts) {
            double current_e = compute_total_energy(s);
            double drift = (std::abs(e0) > 1e-15) ? (std::abs(current_e - e0) / std::abs(e0)) : 0.0;
            out_drifts[step] = drift;
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();
    if (out_elapsed_ms) *out_elapsed_ms = elapsed_ms;

    // Output updated position and velocity arrays
    for (int i = 0; i < n; ++i) {
        pos[3 * i + 0] = s.x[i];
        pos[3 * i + 1] = s.y[i];
        pos[3 * i + 2] = s.z[i];
        vel[3 * i + 0] = s.vx[i];
        vel[3 * i + 1] = s.vy[i];
        vel[3 * i + 2] = s.vz[i];
    }

    return 0;
}

} // extern "C"
