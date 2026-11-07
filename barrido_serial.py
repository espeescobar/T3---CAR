# Ejecuta lu_serial.py (sin modificarlo) para varios b y guarda los tiempos en un CSV.
# Uso: python barrido_serial.py --N 4096 --bs 4096 2048 ... --reps 3 --csv salida.csv --maquina nombre
# Fija 1 hilo salvo que la variable BLAS_THREADS indique otra cosa.
import argparse
import csv
import os
import re
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--N", type=int, default=4096)
parser.add_argument("--bs", type=int, nargs="+", default=[2 ** e for e in range(12, 5, -1)])
parser.add_argument("--reps", type=int, default=3)
parser.add_argument("--csv", required=True)
parser.add_argument("--maquina", default="")
args = parser.parse_args()

threads = os.environ.get("BLAS_THREADS", "1")
env = dict(os.environ)
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    env[v] = threads
here = os.path.dirname(os.path.abspath(__file__))

new = not os.path.exists(args.csv)
with open(args.csv, "a", newline="") as f:
    w = csv.writer(f)
    if new:
        w.writerow(["maquina", "N", "b", "T", "rep", "tiempo", "gflops", "checksum"])
    for rep in range(1, args.reps + 1):
        for b in args.bs:
            out = subprocess.run([sys.executable, os.path.join(here, "lu_serial.py"), "--N", str(args.N),
                                  "--b", str(b)], env=env, capture_output=True, text=True, check=True).stdout
            print(f"rep={rep} {out.strip()}", flush=True)
            m = re.search(r"tiempo=([\d.]+) s GFLOP/s=([\d.]+) checksum=(\S+)", out)
            w.writerow([args.maquina, args.N, b, threads, rep, *m.groups()])
            f.flush()
