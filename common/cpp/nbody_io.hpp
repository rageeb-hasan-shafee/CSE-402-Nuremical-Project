#pragma once
// common/cpp/nbody_io.hpp — Common I/O and timing utilities for N-body simulation backends.
// Follows C++17 standard, compatible with host compilers and nvcc.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <filesystem>

namespace nbio {

// Gaussian Gravitational Constant squared: (0.01720209895)^2 in AU^3 / (Msun * day^2)
inline constexpr double G = 0.0002959122082855911;

struct Args {
    std::string input;
    int steps = 10;
    double dt = 0.1;
    std::string variant = "default";
    int block_size = 256;
    std::string final_state;
    std::string output_json;
    std::string machine = "unknown";
};

struct System {
    std::vector<std::string> names;
    std::vector<double> m;
    std::vector<double> x, y, z;
    std::vector<double> vx, vy, vz;

    int n() const { return static_cast<int>(m.size()); }
};

struct Timings {
    double setup = 0.0;
    double loop = 0.0;
    double transfer = 0.0;
};

inline double now() {
    auto tp = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(tp.time_since_epoch()).count();
}

inline Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--input" && i + 1 < argc) {
            a.input = argv[++i];
        } else if (arg == "--steps" && i + 1 < argc) {
            a.steps = std::atoi(argv[++i]);
        } else if (arg == "--dt" && i + 1 < argc) {
            a.dt = std::atof(argv[++i]);
        } else if (arg == "--variant" && i + 1 < argc) {
            a.variant = argv[++i];
        } else if (arg == "--block-size" && i + 1 < argc) {
            a.block_size = std::atoi(argv[++i]);
        } else if (arg == "--final-state" && i + 1 < argc) {
            a.final_state = argv[++i];
        } else if (arg == "--output-json" && i + 1 < argc) {
            a.output_json = argv[++i];
        } else if (arg == "--machine" && i + 1 < argc) {
            a.machine = argv[++i];
        }
    }
    return a;
}

inline System read_ic_csv(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::fprintf(stderr, "Error: cannot open input CSV file '%s'\n", path.c_str());
        std::exit(1);
    }

    System s;
    std::string line;
    bool first = true;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;

        // Check for header line
        if (first) {
            first = false;
            if (line.find("name") != std::string::npos || line.find("mass") != std::string::npos) {
                continue;
            }
        }

        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> parts;
        while (std::getline(ss, token, ',')) {
            // Trim whitespace
            size_t start = token.find_first_not_of(" \t\r\n");
            size_t end = token.find_last_not_of(" \t\r\n");
            if (start != std::string::npos && end != std::string::npos) {
                parts.push_back(token.substr(start, end - start + 1));
            } else {
                parts.push_back("");
            }
        }

        if (parts.size() >= 8) {
            s.names.push_back(parts[0]);
            s.m.push_back(std::stod(parts[1]));
            s.x.push_back(std::stod(parts[2]));
            s.y.push_back(std::stod(parts[3]));
            s.z.push_back(std::stod(parts[4]));
            s.vx.push_back(std::stod(parts[5]));
            s.vy.push_back(std::stod(parts[6]));
            s.vz.push_back(std::stod(parts[7]));
        }
    }
    return s;
}

inline void write_state_csv(const std::string& path, const System& s) {
    namespace fs = std::filesystem;
    fs::path p(path);
    if (p.has_parent_path()) {
        fs::create_directories(p.parent_path());
    }

    std::ofstream file(path);
    if (!file.is_open()) {
        std::fprintf(stderr, "Error: cannot open output CSV '%s'\n", path.c_str());
        return;
    }

    file << "name,mass,x,y,z,vx,vy,vz\n";
    file << std::setprecision(17);
    for (int i = 0; i < s.n(); ++i) {
        std::string name = (i < static_cast<int>(s.names.size())) ? s.names[i] : ("body_" + std::to_string(i));
        file << name << ","
             << s.m[i] << ","
             << s.x[i] << "," << s.y[i] << "," << s.z[i] << ","
             << s.vx[i] << "," << s.vy[i] << "," << s.vz[i] << "\n";
    }
}

inline void write_result_json(const Args& a, const std::string& backend, const std::string& method,
                              int n, int p, const Timings& t, const std::string& extra_json) {
    if (a.output_json.empty()) return;

    namespace fs = std::filesystem;
    fs::path pth(a.output_json);
    if (pth.has_parent_path()) {
        fs::create_directories(pth.parent_path());
    }

    std::ofstream f(a.output_json);
    if (!f.is_open()) {
        std::fprintf(stderr, "Error: cannot open output JSON '%s'\n", a.output_json.c_str());
        return;
    }

    f << "{\n";
    f << "  \"machine\": \"" << a.machine << "\",\n";
    f << "  \"backend\": \"" << backend << "\",\n";
    f << "  \"method\": \"" << method << "\",\n";
    f << "  \"variant\": \"" << a.variant << "\",\n";
    f << "  \"n\": " << n << ",\n";
    f << "  \"p\": " << p << ",\n";
    f << "  \"steps\": " << a.steps << ",\n";
    f << "  \"dt\": " << a.dt << ",\n";
    f << "  \"timings\": {\n";
    f << "    \"setup\": " << t.setup << ",\n";
    f << "    \"loop\": " << t.loop << ",\n";
    f << "    \"transfer\": " << t.transfer << ",\n";
    f << "    \"total\": " << (t.setup + t.loop + t.transfer) << ",\n";
    f << "    \"ms_per_step\": " << (a.steps > 0 ? (1e3 * t.loop / a.steps) : 0.0) << "\n";
    f << "  }";
    if (!extra_json.empty()) {
        f << ",\n  \"extra\": " << extra_json;
    }
    f << "\n}\n";
}

} // namespace nbio
