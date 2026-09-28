# bench/check_result.py — contract §C4 schema v1 validator
import json
import sys

ALLOWED_BACKENDS = ["py-numpy", "py-loop", "cpp-serial", "cpp-openmp", "py-mpi", "cuda"]
REQUIRED_KEYS = [
    "schema_version", "backend", "variant", "lang", "n_bodies", "workers",
    "steps", "dt", "t_setup_s", "t_loop_s", "t_force_s", "t_comm_s",
    "t_transfer_s", "machine", "extra"
]


def check(path):
    with open(path) as f:
        doc = json.load(f)
    if doc.get("schema_version") != 1:
        sys.exit(f"FAIL: schema_version {doc.get('schema_version')} != 1")
    for k in REQUIRED_KEYS:
        if k not in doc:
            sys.exit(f"FAIL: missing key '{k}' in {path}")
    if doc["backend"] not in ALLOWED_BACKENDS:
        sys.exit(f"FAIL: backend '{doc['backend']}' not in {ALLOWED_BACKENDS}")
    if not isinstance(doc["n_bodies"], int) or doc["n_bodies"] <= 0:
        sys.exit(f"FAIL: invalid n_bodies: {doc['n_bodies']}")
    if doc["t_loop_s"] <= 0:
        sys.exit(f"FAIL: t_loop_s must be > 0, got {doc['t_loop_s']}")
    print(f"PASS: {path} matches Contract schema v1 ({doc['backend']}[{doc['variant']}] N={doc['n_bodies']})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Usage: python check_result.py <file.json>")
    check(sys.argv[1])
