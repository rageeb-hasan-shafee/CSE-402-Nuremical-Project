#!/usr/bin/env python3
"""
bench/defense_server.py — Lightweight Local Backend for CSE 402 Oral Defense Hub
Uses Python's standard library http.server to serve the dashboard and expose
the /api/run endpoint for live C++ OpenMP and MS-MPI subprocess execution.
Zero heavy dependencies, 100% reliable.
"""

import http.server
import json
import os
import pathlib
import socketserver
import subprocess
import sys
import time
import urllib.parse
import webbrowser

ROOT = pathlib.Path(__file__).resolve().parent.parent
BUILD_DIR = ROOT / "backends" / "openmp" / "build"
SERIAL_EXE = BUILD_DIR / "nbody_serial.exe"
OMP_EXE = BUILD_DIR / "nbody_omp.exe"
MPI_SCRIPT = ROOT / "backends" / "mpi" / "nbody_mpi.py"
PYTHON_EXE = ROOT / ".venv" / "Scripts" / "python.exe"
if not PYTHON_EXE.exists():
    PYTHON_EXE = pathlib.Path(sys.executable)

PORT = 8000

class DefenseHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        if url.path == "/" or url.path == "/defense":
            self.send_response(302)
            self.send_header("Location", "/bench/defense_dashboard.html")
            self.end_headers()
            return
        elif url.path == "/api/run":
            self.handle_api_run(urllib.parse.parse_qs(url.query))
            return
        return super().do_GET()

    def handle_api_run(self, params):
        backend = params.get("backend", ["openmp"])[0]
        n_bodies = int(params.get("n", ["1000"])[0])
        steps = int(params.get("steps", ["100"])[0])
        workers = int(params.get("workers", ["6"])[0])
        variant = params.get("variant", ["static"])[0]

        ic_file = ROOT / "data" / "ic" / f"ic_N{n_bodies}_s42.csv"
        if not ic_file.exists():
            subprocess.run([sys.executable, str(ROOT / "common" / "python" / "export_ic.py"), "--n", str(n_bodies)], check=True)

        temp_json = ROOT / "bench" / "results" / "live_defense_run.json"
        temp_json.parent.mkdir(parents=True, exist_ok=True)

        if backend == "openmp":
            cmd = [
                str(OMP_EXE),
                "--input", str(ic_file),
                "--steps", str(steps),
                "--dt", "0.1",
                "--workers", str(workers),
                "--variant", variant,
                "--machine", "fahad-ryzen-5600g",
                "--output-json", str(temp_json)
            ]
        elif backend == "mpi":
            mpiexec = ROOT / "system" / "scoop" / "apps" / "msmpi" / "10.1.1" / "mpiexec.exe"
            if not mpiexec.exists():
                mpiexec = "mpiexec"
            cmd = [
                str(mpiexec), "-n", str(workers),
                str(PYTHON_EXE), str(MPI_SCRIPT),
                "--input", str(ic_file),
                "--steps", str(steps),
                "--dt", "0.1",
                "--variant", variant,
                "--machine", "fahad-ryzen-5600g",
                "--output-json", str(temp_json)
            ]
        else: # serial
            cmd = [
                str(SERIAL_EXE),
                "--input", str(ic_file),
                "--steps", str(steps),
                "--dt", "0.1",
                "--workers", "1",
                "--variant", "static",
                "--machine", "fahad-ryzen-5600g",
                "--output-json", str(temp_json)
            ]

        t0 = time.time()
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        elapsed = time.time() - t0

        telemetry = {}
        if temp_json.exists():
            try:
                with open(temp_json, "r", encoding="utf-8") as f:
                    telemetry = json.load(f)
            except Exception:
                pass

        res = {
            "success": proc.returncode == 0,
            "cmd": " ".join(cmd),
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "elapsed_s": elapsed,
            "telemetry": telemetry
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(res).encode("utf-8"))

def main():
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), DefenseHandler) as httpd:
        url = f"http://localhost:{PORT}/bench/defense_dashboard.html"
        print(f"==================================================================")
        print(f"  [SERVER] CSE 402 Defense & Live Demonstration Server Running")
        print(f"  Open in browser: {url}")
        print(f"==================================================================")
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
