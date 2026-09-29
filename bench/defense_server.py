#!/usr/bin/env python3
"""
bench/defense_server.py — Portable Local Server for CSE 402 Oral Defense Hub
- Serves the defense dashboard at http://localhost:8000
- Exposes /api/sim/info and /api/sim/batch for LIVE C++ OpenMP Native Simulation at MAX threads
- Exposes /api/run for on-demand execution of C++ OpenMP, MS-MPI, and Serial benchmark runs
- Uses Python standard library only (http.server + socketserver + ctypes)
- Zero external runtime dependencies: statically compiled C++ DLL (runs on any Windows machine)
"""

import ctypes
import http.server
import json
import os
import pathlib
import platform
import shutil
import socketserver
import subprocess
import sys
import time
import urllib.parse
import webbrowser

# Dynamic repo root resolution
ROOT = pathlib.Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT / "backends" / "openmp" / "build"
SERIAL_EXE = BUILD_DIR / ("nbody_serial.exe" if os.name == "nt" else "nbody_serial")
OMP_EXE = BUILD_DIR / ("nbody_omp.exe" if os.name == "nt" else "nbody_omp")
MPI_SCRIPT = ROOT / "backends" / "mpi" / "nbody_mpi.py"
PYTHON_EXE = ROOT / ".venv" / "Scripts" / "python.exe"
if not PYTHON_EXE.exists():
    PYTHON_EXE = pathlib.Path(sys.executable)

ENGINE_DLL = BUILD_DIR / "nbody_omp_engine.dll"
_omp_lib = None

PORT = 8000


def get_machine_tag():
    """Detect a consistent machine tag for benchmark reporting."""
    node = platform.node().lower()
    if "desktop-mfjj68d" in node or "ryzen" in platform.processor().lower():
        return "fahad-ryzen-5600g"
    return f"win-{node}" if node else "windows-pc"


def find_mpiexec():
    """Dynamically discover mpiexec executable across common Windows installations."""
    which_path = shutil.which("mpiexec")
    if which_path:
        return which_path

    candidates = [
        r"C:\Program Files\Microsoft MPI\Bin\mpiexec.exe",
        r"C:\Program Files (x86)\Microsoft SDKs\MPI\mpiexec.exe",
        os.path.expandvars(r"%USERPROFILE%\scoop\shims\mpiexec.exe"),
        r"E:\system\scoop\shims\mpiexec.exe",
        r"E:\system\scoop\apps\msmpi\10.1.1\mpiexec.exe",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return "mpiexec"


def init_omp_engine():
    """Load or compile the native C++ OpenMP simulation engine."""
    global _omp_lib
    if not ENGINE_DLL.exists():
        engine_src = ROOT / "backends" / "openmp" / "nbody_omp_engine.cpp"
        if engine_src.exists() and shutil.which("g++"):
            try:
                BUILD_DIR.mkdir(parents=True, exist_ok=True)
                print(
                    "[ENGINE] Compiling native OpenMP engine DLL with -static -fopenmp..."
                )
                subprocess.run(
                    [
                        "g++",
                        "-O3",
                        "-fopenmp",
                        "-shared",
                        "-static",
                        "-std=c++17",
                        "-o",
                        str(ENGINE_DLL),
                        str(engine_src),
                    ],
                    check=True,
                    cwd=str(ROOT),
                )
            except Exception as e:
                print(f"[ENGINE WARN] Auto-compile failed: {e}")

    if ENGINE_DLL.exists():
        try:
            lib = ctypes.CDLL(str(ENGINE_DLL))
            lib.get_max_threads.restype = ctypes.c_int
            lib.run_openmp_batch.argtypes = [
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.c_double,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_double),
            ]
            lib.run_openmp_batch.restype = ctypes.c_int
            _omp_lib = lib
            print(
                f"  [ENGINE] Native C++ OpenMP Engine Active! (Max Hardware Threads: {lib.get_max_threads()})"
            )
        except Exception as e:
            print(f"[ENGINE WARN] Failed to load {ENGINE_DLL}: {e}")


class DefenseHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        if url.path in ("/", "/defense", "/dashboard"):
            self.send_response(302)
            self.send_header("Location", "/bench/defense_dashboard.html")
            self.end_headers()
            return
        elif url.path == "/api/sim/info":
            max_t = _omp_lib.get_max_threads() if _omp_lib else os.cpu_count()
            self.send_json_response(
                {
                    "available": _omp_lib is not None,
                    "engine": "cpp-openmp-native" if _omp_lib else "none",
                    "max_threads": max_t,
                    "variants": ["static", "dynamic", "simd", "newton3"],
                    "machine": get_machine_tag(),
                }
            )
            return
        elif url.path == "/api/run":
            self.handle_api_run(urllib.parse.parse_qs(url.query))
            return
        return super().do_GET()

    def do_POST(self):
        url = urllib.parse.urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", 0))
        body = (
            self.rfile.read(content_length).decode("utf-8")
            if content_length > 0
            else "{}"
        )

        if url.path == "/api/sim/batch":
            try:
                data = json.loads(body)
                self.handle_sim_batch(data)
            except Exception as e:
                self.send_json_response({"success": False, "error": str(e)}, 400)
            return

        elif url.path == "/api/run":
            try:
                params_dict = json.loads(body) if body else {}
                params = {k: [str(v)] for k, v in params_dict.items()}
            except Exception:
                params = urllib.parse.parse_qs(body)
            self.handle_api_run(params)
            return

        self.send_error(404, "Endpoint not found")

    def handle_sim_batch(self, req):
        """Executes a batch of simulation steps using native C++ OpenMP at MAX threads."""
        global _omp_lib
        if _omp_lib is None:
            init_omp_engine()

        if _omp_lib is None:
            self.send_json_response(
                {
                    "success": False,
                    "error": "Native C++ OpenMP engine not compiled or unavailable. Falling back to browser engine.",
                },
                503,
            )
            return

        n = int(req.get("n", 0))
        masses_list = req.get("masses", [])
        pos_list = req.get("positions", [])
        vel_list = req.get("velocities", [])
        dt = float(req.get("dt", 0.05))
        steps = int(req.get("steps", 60))
        variant = str(req.get("variant", "simd")).lower()
        threads = int(req.get("threads", 0))  # 0 = max available

        if (
            n <= 0
            or len(pos_list) < 3 * n
            or len(vel_list) < 3 * n
            or len(masses_list) < n
        ):
            self.send_json_response(
                {"success": False, "error": "Invalid body arrays length"}, 400
            )
            return

        variant_map = {
            "newton3": 0,
            "cpp_newton3": 0,
            "verlet": 0,
            "simd": 1,
            "cpp_simd": 1,
            "dynamic": 2,
            "cpp_dynamic": 2,
            "static": 3,
            "cpp_static": 3,
        }
        variant_id = variant_map.get(variant, 0)  # default to Newton3 Verlet

        c_masses = (ctypes.c_double * n)(*masses_list)
        c_pos = (ctypes.c_double * (3 * n))(*pos_list)
        c_vel = (ctypes.c_double * (3 * n))(*vel_list)

        c_out_pos = (ctypes.c_double * (steps * 3 * n))()
        c_out_vel = (ctypes.c_double * (steps * 3 * n))()
        c_out_drifts = (ctypes.c_double * steps)()
        c_elapsed_ms = ctypes.c_double(0.0)

        ret = _omp_lib.run_openmp_batch(
            n,
            c_masses,
            c_pos,
            c_vel,
            ctypes.c_double(dt),
            steps,
            variant_id,
            threads,
            c_out_pos,
            c_out_vel,
            c_out_drifts,
            ctypes.byref(c_elapsed_ms),
        )

        if ret != 0:
            self.send_json_response(
                {"success": False, "error": f"OpenMP engine returned code {ret}"}, 500
            )
            return

        max_t = _omp_lib.get_max_threads()
        active_t = threads if (threads > 0 and threads <= max_t) else max_t

        self.send_json_response(
            {
                "success": True,
                "engine": "cpp-openmp-native",
                "variant": variant,
                "threads": active_t,
                "steps": steps,
                "dt": dt,
                "cpp_elapsed_ms": round(c_elapsed_ms.value, 3),
                "cpp_ms_per_step": round(c_elapsed_ms.value / steps, 5),
                "positions": list(c_out_pos),
                "velocities": list(c_out_vel),
                "drifts": list(c_out_drifts),
                "final_pos": list(c_pos),
                "final_vel": list(c_vel),
            }
        )

    def handle_api_run(self, params):
        backend = params.get("backend", ["openmp"])[0].lower()
        n_bodies = int(params.get("n", ["1000"])[0])
        steps = int(params.get("steps", ["100"])[0])
        workers = int(params.get("workers", ["6"])[0])
        variant = params.get("variant", ["static"])[0].lower()

        machine_tag = get_machine_tag()
        ic_file = ROOT / "data" / "ic" / f"ic_N{n_bodies}_s42.csv"

        if not ic_file.exists():
            export_script = ROOT / "common" / "python" / "export_ic.py"
            if export_script.exists():
                try:
                    subprocess.run(
                        [str(PYTHON_EXE), str(export_script), "--n", str(n_bodies)],
                        check=True,
                        cwd=str(ROOT),
                    )
                except Exception as e:
                    self.send_json_response(
                        {
                            "success": False,
                            "status": "error",
                            "error": f"Failed to generate IC file for N={n_bodies}: {e}",
                        },
                        500,
                    )
                    return

        temp_json = ROOT / "bench" / "results" / "live_defense_run.json"
        temp_json.parent.mkdir(parents=True, exist_ok=True)

        env = dict(os.environ)
        env["OPENBLAS_NUM_THREADS"] = "1"
        env["OMP_NUM_THREADS"] = str(workers) if backend == "openmp" else "1"

        if backend == "openmp":
            if not OMP_EXE.exists():
                self.send_json_response(
                    {
                        "success": False,
                        "status": "error",
                        "error": f"Executable not found: {OMP_EXE}.",
                    },
                    400,
                )
                return
            cmd = [
                str(OMP_EXE),
                "--input",
                str(ic_file),
                "--steps",
                str(steps),
                "--dt",
                "0.1",
                "--workers",
                str(workers),
                "--variant",
                variant,
                "--machine",
                machine_tag,
                "--output-json",
                str(temp_json),
            ]
        elif backend == "mpi":
            mpiexec_bin = find_mpiexec()
            cmd = [
                str(mpiexec_bin),
                "-n",
                str(workers),
                str(PYTHON_EXE),
                str(MPI_SCRIPT),
                "--input",
                str(ic_file),
                "--steps",
                str(steps),
                "--dt",
                "0.1",
                "--variant",
                variant,
                "--machine",
                machine_tag,
                "--output-json",
                str(temp_json),
            ]
        else:  # serial
            if not SERIAL_EXE.exists():
                self.send_json_response(
                    {
                        "success": False,
                        "status": "error",
                        "error": f"Executable not found: {SERIAL_EXE}.",
                    },
                    400,
                )
                return
            cmd = [
                str(SERIAL_EXE),
                "--input",
                str(ic_file),
                "--steps",
                str(steps),
                "--dt",
                "0.1",
                "--workers",
                "1",
                "--variant",
                "static",
                "--machine",
                machine_tag,
                "--output-json",
                str(temp_json),
            ]

        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(ROOT),
                env=env,
            )
            elapsed = time.time() - t0
        except Exception as e:
            self.send_json_response(
                {
                    "success": False,
                    "status": "error",
                    "cmd": " ".join(cmd),
                    "error": str(e),
                },
                500,
            )
            return

        telemetry = {}
        if temp_json.exists():
            try:
                with open(temp_json, "r", encoding="utf-8") as f:
                    telemetry = json.load(f)
            except Exception:
                pass

        res = {
            "success": proc.returncode == 0,
            "status": "ok" if proc.returncode == 0 else "error",
            "cmd": " ".join(cmd),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "elapsed_sec": round(elapsed, 3),
            "loop_ms": round(telemetry.get("median_ms", 0.0), 4) if telemetry else None,
            "telemetry": telemetry,
        }
        self.send_json_response(res)

    def send_json_response(self, data, status_code=200):
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))


def main():
    init_omp_engine()
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), DefenseHandler) as httpd:
        url = f"http://localhost:{PORT}/bench/defense_dashboard.html"
        print("==================================================================")
        print("  [SERVER] CSE 402 Defense & Live Demonstration Server Running")
        print(f"  Root Directory: {ROOT}")
        print(f"  Machine Tag:    {get_machine_tag()}")
        if _omp_lib:
            print(
                f"  OpenMP Engine:  ACTIVE ({_omp_lib.get_max_threads()} Hardware Threads MAX)"
            )
        else:
            print(f"  OpenMP Engine:  INACTIVE (Client-side Fallback)")
        print(f"  Open in browser: {url}")
        print("==================================================================")
        try:
            webbrowser.open(url)
        except Exception:
            pass
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down defense server.")


if __name__ == "__main__":
    main()
