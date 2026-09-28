// Shared I/O for C++ and CUDA backends, contract v1 (see common/CONTRACT.md). Owner: Fahad.
// Header-only, standard library only, compiles with g++, clang++, MSVC and nvcc.
#pragma once
#include <chrono>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace nbio {

constexpr double G = 2.9591220828559115e-4;  // AU^3 Msun^-1 day^-2
constexpr int SCHEMA_VERSION = 1;

struct System {  // structure-of-arrays
    std::vector<std::string> name;
    std::vector<double> m, x, y, z, vx, vy, vz;
    int n() const { return static_cast<int>(m.size()); }
};

struct Args {
    std::string input, output_json, final_state, variant = "default", machine = "unknown";
    int steps = 100, workers = 1, block_size = 256;
    double dt = 0.1;
};

struct Timings { double setup = 0, loop = 0, force = 0, comm = 0, transfer = 0; };

inline double now() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}

inline Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        auto val = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + k);
            return argv[++i];
        };
        if (k == "--input") a.input = val();
        else if (k == "--output-json") a.output_json = val();
        else if (k == "--final-state") a.final_state = val();
        else if (k == "--variant") a.variant = val();
        else if (k == "--machine") a.machine = val();
        else if (k == "--steps") a.steps = std::stoi(val());
        else if (k == "--workers") a.workers = std::stoi(val());
        else if (k == "--block-size") a.block_size = std::stoi(val());
        else if (k == "--dt") a.dt = std::stod(val());
        else throw std::runtime_error("unknown argument " + k);
    }
    if (a.input.empty()) throw std::runtime_error("--input is required");
    return a;
}

inline System read_ic_csv(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    System s;
    std::string line, tok;
    std::getline(f, line);  // header: name,mass,x,y,z,vx,vy,vz
    std::vector<double>* cols[7] = {&s.m, &s.x, &s.y, &s.z, &s.vx, &s.vy, &s.vz};
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();  // CRLF-safe
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::getline(ss, tok, ',');
        s.name.push_back(tok);
        for (auto* c : cols) {
            std::getline(ss, tok, ',');
            c->push_back(std::stod(tok));
        }
    }
    return s;
}

inline void write_state_csv(const std::string& path, const System& s) {
    std::FILE* f = std::fopen(path.c_str(), "w");
    if (!f) throw std::runtime_error("cannot write " + path);
    std::fprintf(f, "name,mass,x,y,z,vx,vy,vz\n");
    for (int i = 0; i < s.n(); ++i)
        std::fprintf(f, "%s,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n", s.name[i].c_str(),
                     s.m[i], s.x[i], s.y[i], s.z[i], s.vx[i], s.vy[i], s.vz[i]);
    std::fclose(f);
}

// extra_json must be a JSON object literal, e.g. "{\"block_size\": 256}"
inline void write_result_json(const Args& a, const std::string& backend, const std::string& lang,
                              int n_bodies, int workers, const Timings& t,
                              const std::string& extra_json = "{}") {
    if (a.output_json.empty()) return;
    std::FILE* f = std::fopen(a.output_json.c_str(), "w");
    if (!f) throw std::runtime_error("cannot write " + a.output_json);
    std::fprintf(f,
        "{\n  \"schema_version\": %d,\n  \"backend\": \"%s\",\n  \"variant\": \"%s\",\n"
        "  \"lang\": \"%s\",\n  \"n_bodies\": %d,\n  \"workers\": %d,\n  \"steps\": %d,\n"
        "  \"dt\": %.17g,\n  \"t_setup_s\": %.9f,\n  \"t_loop_s\": %.9f,\n  \"t_force_s\": %.9f,\n"
        "  \"t_comm_s\": %.9f,\n  \"t_transfer_s\": %.9f,\n  \"machine\": \"%s\",\n  \"extra\": %s\n}\n",
        SCHEMA_VERSION, backend.c_str(), a.variant.c_str(), lang.c_str(), n_bodies, workers,
        a.steps, a.dt, t.setup, t.loop, t.force, t.comm, t.transfer, a.machine.c_str(),
        extra_json.c_str());
    std::fclose(f);
}

}  // namespace nbio
