// backends/openmp/nbody_omp.cpp — Owner: Siam. Contract v1 (common/CONTRACT.md).
// Built twice from this one file:
//   without OpenMP -> nbody_serial  (backend "cpp-serial", the compiled baseline)
//   with OpenMP    -> nbody_omp     (backend "cpp-openmp")
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <utility>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "nbody_io.hpp"

using Vec = std::vector<double>;

// variant "static" / "dynamic": outer loop over targets p, full inner loop over sources j
static void accel_basic(const nbio::System& s, Vec& ax, Vec& ay, Vec& az) {
    const int n = s.n();
    const double *x = s.x.data(), *y = s.y.data(), *z = s.z.data(), *m = s.m.data();
    #pragma omp parallel for schedule(runtime)
    for (int p = 0; p < n; ++p) {
        const double xp = x[p], yp = y[p], zp = z[p];
        double sx = 0.0, sy = 0.0, sz = 0.0;
        for (int j = 0; j < n; ++j) {
            if (j == p) continue;
            const double dx = x[j] - xp, dy = y[j] - yp, dz = z[j] - zp;
            const double r2 = dx * dx + dy * dy + dz * dz;
            const double f = nbio::G * m[j] / (r2 * std::sqrt(r2));
            sx += f * dx; sy += f * dy; sz += f * dz;
        }
        ax[p] = sx; ay[p] = sy; az[p] = sz;
    }
}

// variant "simd": split the j-loop around p so there is no branch, then ask for SIMD
static inline void accum_range(const double* x, const double* y, const double* z, const double* m,
                               int j0, int j1, double xp, double yp, double zp,
                               double& sx, double& sy, double& sz) {
    double ax = 0.0, ay = 0.0, az = 0.0;
    #pragma omp simd reduction(+ : ax, ay, az)
    for (int j = j0; j < j1; ++j) {
        const double dx = x[j] - xp, dy = y[j] - yp, dz = z[j] - zp;
        const double r2 = dx * dx + dy * dy + dz * dz;
        const double f = m[j] / (r2 * std::sqrt(r2));
        ax += f * dx; ay += f * dy; az += f * dz;
    }
    sx += ax; sy += ay; sz += az;
}

static void accel_simd(const nbio::System& s, Vec& ax, Vec& ay, Vec& az) {
    const int n = s.n();
    const double *x = s.x.data(), *y = s.y.data(), *z = s.z.data(), *m = s.m.data();
    #pragma omp parallel for schedule(static)
    for (int p = 0; p < n; ++p) {
        double sx = 0.0, sy = 0.0, sz = 0.0;
        accum_range(x, y, z, m, 0, p, x[p], y[p], z[p], sx, sy, sz);
        accum_range(x, y, z, m, p + 1, n, x[p], y[p], z[p], sx, sy, sz);
        ax[p] = nbio::G * sx; ay[p] = nbio::G * sy; az[p] = nbio::G * sz;
    }
}

// variant "newton3": each pair computed once (half the flops) using a_ij = -a_ji.
// Two threads can hit the same body, so each thread accumulates into private
// buffers, which are then reduced. The triangular loop is unbalanced, hence schedule(dynamic).
static void accel_newton3(const nbio::System& s, Vec& ax, Vec& ay, Vec& az) {
    const int n = s.n();
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
                const double inv3 = nbio::G / (r2 * std::sqrt(r2));
                const double fp = m[j] * inv3, fj = m[p] * inv3;
                lx[p] += fp * dx; ly[p] += fp * dy; lz[p] += fp * dz;
                lx[j] -= fj * dx; ly[j] -= fj * dy; lz[j] -= fj * dz;
            }
        }
        #pragma omp critical
        for (int i = 0; i < n; ++i) { ax[i] += lx[i]; ay[i] += ly[i]; az[i] += lz[i]; }
    }
}

int main(int argc, char** argv) {
    nbio::Args a = nbio::parse_args(argc, argv);
    if (a.variant == "default") a.variant = "static";
#ifdef _OPENMP
    const std::string backend = "cpp-openmp";
    omp_set_num_threads(a.workers);
  #if _OPENMP >= 200805  // OpenMP 3.0+ (MSVC's default /openmp is 2.0: stays static)
    if (a.variant == "dynamic") omp_set_schedule(omp_sched_dynamic, 16);
    else omp_set_schedule(omp_sched_static, 0);
  #endif
#else
    const std::string backend = "cpp-serial";
    a.workers = 1;
#endif
    auto accel = accel_basic;
    if (a.variant == "simd") accel = accel_simd;
    else if (a.variant == "newton3") accel = accel_newton3;
    else if (a.variant != "static" && a.variant != "dynamic") {
        std::fprintf(stderr, "unknown --variant %s\n", a.variant.c_str());
        return 2;
    }

    nbio::System s = nbio::read_ic_csv(a.input);  // not timed (contract §C5)
    const int n = s.n();
    nbio::Timings t;

    double t0 = nbio::now();
    Vec ax(n), ay(n), az(n), bx(n), by(n), bz(n);
    accel(s, ax, ay, az);  // initial a0
    t.setup = nbio::now() - t0;

    const double dt = a.dt, half_dt2 = 0.5 * dt * dt, half_dt = 0.5 * dt;
    double *x = s.x.data(), *y = s.y.data(), *z = s.z.data();
    double *vx = s.vx.data(), *vy = s.vy.data(), *vz = s.vz.data();

    t0 = nbio::now();
    for (int step = 0; step < a.steps; ++step) {
        #pragma omp parallel for schedule(static)
        for (int p = 0; p < n; ++p) {  // drift
            x[p] += vx[p] * dt + ax[p] * half_dt2;
            y[p] += vy[p] * dt + ay[p] * half_dt2;
            z[p] += vz[p] * dt + az[p] * half_dt2;
        }
        const double tf = nbio::now();
        accel(s, bx, by, bz);
        t.force += nbio::now() - tf;
        #pragma omp parallel for schedule(static)
        for (int p = 0; p < n; ++p) {  // kick
            vx[p] += (ax[p] + bx[p]) * half_dt;
            vy[p] += (ay[p] + by[p]) * half_dt;
            vz[p] += (az[p] + bz[p]) * half_dt;
        }
        std::swap(ax, bx); std::swap(ay, by); std::swap(az, bz);
    }
    t.loop = nbio::now() - t0;

    if (!a.final_state.empty()) nbio::write_state_csv(a.final_state, s);
    nbio::write_result_json(a, backend, "cpp", n, a.workers, t);
    std::printf("%s[%s] N=%d P=%d steps=%d loop=%.4fs (%.3f ms/step)\n", backend.c_str(),
                a.variant.c_str(), n, a.workers, a.steps, t.loop, 1e3 * t.loop / a.steps);
    return 0;
}
