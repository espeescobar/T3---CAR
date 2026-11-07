"""Estima latencia (alpha) y tiempo por double (beta) entre los ranks 0 y 1.

Uso: mpirun -n 2 python pingpong.py [--csv archivo] [--etiqueta texto]
Ajusta t = alpha + n*beta por mínimos cuadrados sobre la mediana de cada tamaño."""
import argparse
import csv
import os
import time

import numpy as np
from mpi4py import MPI


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv")
    ap.add_argument("--etiqueta", default="")
    args = ap.parse_args()
    comm = MPI.COMM_WORLD
    rank = comm.rank
    tamanos = [2 ** e for e in range(0, 24, 2)]  # 1 a ~4 M doubles
    filas = []
    for n in tamanos:
        buf = np.zeros(n)
        reps = max(5, min(300, 40_000_000 // (n * 8)))
        tiempos = []
        for _ in range(reps):
            comm.Barrier()
            t0 = time.perf_counter()
            if rank == 0:
                comm.Send(buf, 1)
                comm.Recv(buf, 1)
            else:
                comm.Recv(buf, 0)
                comm.Send(buf, 0)
            tiempos.append((time.perf_counter() - t0) / 2)
        filas.append((n, float(np.median(tiempos))))
    if rank == 0:
        n, t = np.array(filas).T
        # El ajuste pondera por 1/t para no ignorar los mensajes pequeños.
        beta, alpha = np.polyfit(n, t, 1, w=1 / t)
        nodos = comm.gather(MPI.Get_processor_name(), root=0)
        print(f"nodos={nodos} alpha={alpha:.3e}s beta={beta:.3e}s/double "
              f"BW={8 / beta / 1e9:.2f}GB/s t(1)={t[0]:.3e}s")
        if args.csv:
            nuevo = not os.path.exists(args.csv)
            with open(args.csv, "a", newline="") as f:
                w = csv.writer(f)
                if nuevo:
                    w.writerow(["etiqueta", "nodos", "n", "t", "alpha", "beta"])
                for a, b in filas:
                    w.writerow([args.etiqueta, "+".join(nodos), a, b, alpha, beta])
    else:
        comm.gather(MPI.Get_processor_name(), root=0)


if __name__ == "__main__":
    main()
