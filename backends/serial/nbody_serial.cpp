// backends/serial/nbody_serial.cpp — Single-core CPU scalar baseline for N-body simulation.
// Implements Velocity Verlet second-order symplectic integration in C++17.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "nbody_io.hpp"

// Computes gravitational acceleration on all bodies: O(N^2) direct summation
void compute_accelerations(const nbio::System& s, std::vector<double>& ax,
                           std::vector<double>& ay, std::vector<double>& az) {
    const int n = s.n();
    const double g = nbio::G;

    for (int i = 0; i < n; ++i) {
        double aix = 0.0, aiy = 0.0, aiz = 0.0;
        const double xi = s.x[i], yi = s.y[i], zi = s.z[i];

        for (int j = 0; j < n; ++j) {
            if (i == j) continue; // Skip self-interaction

            const double dx = s.x[j] - xi;
            const double dy = s.y[j] - yi;
            const double dz = s.z[j] - zi;
            const double r2 = dx * dx + dy * dy + dz * dz;
            const double r = std::sqrt(r2);
            const double inv_r3 = 1.0 / (r * r2);
            const double factor = s.m[j] * inv_r3;

            aix += factor * dx;
            aiy += factor * dy;
            aiz += factor * dz;
        }

        ax[i] = g * aix;
        ay[i] = g * aiy;
        az[i] = g * aiz;
    }
}

int main(int argc, char** argv) {
    nbio::Args a = nbio::parse_args(argc, argv);
    if (a.input.empty()) {
        std::fprintf(stderr, "Usage: %s --input <ic.csv> [--steps N] [--dt 0.1] [--final-state out.csv] [--output-json res.json]\n", argv[0]);
        return 1;
    }

    nbio::System s = nbio::read_ic_csv(a.input);
    const int n = s.n();
    const double dt = a.dt;
    const double half_dt = 0.5 * dt;
    const double half_dt2 = 0.5 * dt * dt;

    std::vector<double> ax(n, 0.0), ay(n, 0.0), az(n, 0.0);
    std::vector<double> new_ax(n, 0.0), new_ay(n, 0.0), new_az(n, 0.0);

    nbio::Timings t;
    double t0 = nbio::now();

    // Initial acceleration calculation (a0)
    compute_accelerations(s, ax, ay, az);
    t.setup = nbio::now() - t0;

    // Integration loop (timed)
    double loop_start = nbio::now();
    for (int step = 0; step < a.steps; ++step) {
        // Step 1: Position drift
        for (int i = 0; i < n; ++i) {
            s.x[i] += s.vx[i] * dt + ax[i] * half_dt2;
            s.y[i] += s.vy[i] * dt + ay[i] * half_dt2;
            s.z[i] += s.vz[i] * dt + az[i] * half_dt2;
        }

        // Step 2: Acceleration at new positions
        compute_accelerations(s, new_ax, new_ay, new_az);

        // Step 3: Velocity kick with averaged acceleration
        for (int i = 0; i < n; ++i) {
            s.vx[i] += (ax[i] + new_ax[i]) * half_dt;
            s.vy[i] += (ay[i] + new_ay[i]) * half_dt;
            s.vz[i] += (az[i] + new_az[i]) * half_dt;

            ax[i] = new_ax[i];
            ay[i] = new_ay[i];
            az[i] = new_az[i];
        }
    }
    t.loop = nbio::now() - loop_start;
    t.transfer = 0.0; // CPU is host-local

    // Write final state if requested
    if (!a.final_state.empty()) {
        nbio::write_state_csv(a.final_state, s);
    }

    // Write JSON timing metrics
    nbio::write_result_json(a, "cpp-serial", "scalar", n, 1, t, "{\"threads\": 1}");

    std::printf("cpp-serial[scalar] N=%d steps=%d loop=%.4fs (%.3f ms/step)\n",
                n, a.steps, t.loop, (1e3 * t.loop / a.steps));

    return 0;
}
