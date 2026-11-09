# LU por bloques distribuida con MPI, en distribucion bloque-ciclica 2D.
# Uso: mpirun -n P python lu_distributed.py --N N --b b [--pr pr --pc pc] [--check]
#                                           [--csv archivo] [--etiqueta texto]
# El bloque A_IJ pertenece al proceso (I mod pr, J mod pc).
#
# Colectivas del paso k, con (r, c) = (k mod pr, k mod pc):
#   A_kk factorizado: Bcast en row_comm (raiz c) y en col_comm (raiz r).
#   Panel de L      : Bcast en row_comm (raiz c), en cada fila de procesos.
#   Panel de U      : Bcast en col_comm (raiz r), en cada columna de procesos.
#
# Comunicacion = tiempo dentro de Bcast (incluye la espera a que llegue la raiz).
# Computo = lu_factor, solve_triangular, producto @ y copia de paneles a buffers.

import argparse
import csv
import os
import time

import numpy as np
from mpi4py import MPI
from scipy.linalg import solve_triangular

from lu_serial import lu_blocked, lu_diag_block, matrix_entries, residual

ETAPAS = ["lu", "trsm", "gemm", "copia", "bcast_diag", "bcast_panel"]


def global_indices(N, b, mine, p):
    """Indices globales de las filas (o columnas) locales, ordenados."""
    blocks = range(mine, (N + b - 1) // b, p)
    if not blocks:
        return np.empty(0, dtype=np.int64)
    return np.concatenate([np.arange(I * b, min((I + 1) * b, N)) for I in blocks])


def lu_distributed(A, N, b, pr, pc, myrow, mycol, row_comm, col_comm):
    """Factoriza el arreglo local A en su lugar y devuelve los tiempos del proceso."""
    t = dict.fromkeys(ETAPAS, 0.0)
    clock = time.perf_counter
    for k in range((N + b - 1) // b):
        r, c = k % pr, k % pc
        bk = min(b, N - k * b)
        in_row, in_col = myrow == r, mycol == c

        # Los bloques locales estan ordenados por indice global: los bloques > k
        # forman el sub-arreglo contiguo A[i1:, j1:].
        i0 = (k // pr) * b                  # fila local del bloque k (si es de mi fila)
        j0 = (k // pc) * b                  # columna local del bloque k (si es de mi columna)
        i1 = ((k - myrow) // pr + 1) * b    # primera fila local con bloque I > k
        j1 = ((k - mycol) // pc + 1) * b    # primera columna local con bloque J > k
        m = max(A.shape[0] - i1, 0)
        n = max(A.shape[1] - j1, 0)

        # 1. Bloque diagonal: lo factoriza (r, c) y lo difunde en su fila y su columna.
        D = None
        if in_row and in_col:
            t0 = clock()
            D = A[i0:i0 + bk, j0:j0 + bk]
            lu_diag_block(D)
            D = D.copy()
            t["lu"] += clock() - t0
        elif in_row or in_col:
            D = np.empty((bk, bk))
        t0 = clock()
        if in_row:
            row_comm.Bcast(D, root=c)
        if in_col:
            col_comm.Bcast(D, root=r)
        t["bcast_diag"] += clock() - t0

        # 2 y 3. Paneles de U (fila r) y de L (columna c): una resolucion por proceso.
        t0 = clock()
        if in_row and n > 0:
            A[i0:i0 + bk, j1:] = solve_triangular(D, A[i0:i0 + bk, j1:], lower=True,
                                                  unit_diagonal=True, check_finite=False)
        if in_col and m > 0:
            A[i1:, j0:j0 + bk] = solve_triangular(D, A[i1:, j0:j0 + bk].T, trans="T",
                                                  lower=False, check_finite=False).T
        t["trsm"] += clock() - t0

        # 4. Envio de paneles, en buffers contiguos.
        t0 = clock()
        L = A[i1:, j0:j0 + bk].copy() if in_col else np.empty((m, bk))
        U = A[i0:i0 + bk, j1:].copy() if in_row else np.empty((bk, n))
        t["copia"] += clock() - t0
        t0 = clock()
        row_comm.Bcast(L, root=c)
        col_comm.Bcast(U, root=r)
        t["bcast_panel"] += clock() - t0

        # 5. Actualizacion: una sola multiplicacion de matrices.
        t0 = clock()
        if m > 0 and n > 0:
            A[i1:, j1:] -= L @ U
        t["gemm"] += clock() - t0
    return t


def check(comm, A, N, b, rows, cols):
    """Reune la factorizacion en la raiz y la compara con A y con lu_serial."""
    parts = comm.gather((rows, cols, A), root=0)
    if comm.rank != 0:
        return
    idx = np.arange(N)
    A0 = matrix_entries(idx, idx, N)
    LU = np.empty((N, N))
    for f, co, a in parts:
        LU[np.ix_(f, co)] = a
    ref = A0.copy()
    lu_blocked(ref, b)
    print(f"verificacion: residuo={residual(A0, LU):.3e} "
          f"max|LU_dist-LU_serial|={np.abs(LU - ref).max():.3e} "
          f"checksum_dist={LU.sum():.6e} checksum_serial={ref.sum():.6e}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=4096)
    parser.add_argument("--b", type=int, default=256)
    parser.add_argument("--pr", type=int, help="filas del grid (por defecto, Compute_dims)")
    parser.add_argument("--pc", type=int, help="columnas del grid (por defecto, Compute_dims)")
    parser.add_argument("--check", action="store_true", help="compara con la version serial")
    parser.add_argument("--csv", help="agrega una fila por proceso a este CSV")
    parser.add_argument("--etiqueta", default="")
    args = parser.parse_args()
    N, b = args.N, args.b

    comm = MPI.COMM_WORLD
    rank, size = comm.rank, comm.size
    if (args.pr is None) != (args.pc is None):
        raise SystemExit("pasa --pr y --pc juntos, o ninguno")
    pr, pc = (args.pr, args.pc) if args.pr else MPI.Compute_dims(size, 2)
    if pr * pc != size:
        raise SystemExit(f"pr*pc={pr * pc} no coincide con los {size} procesos")

    myrow, mycol = rank // pc, rank % pc
    row_comm = comm.Split(myrow, mycol)  # rango en row_comm = mycol
    col_comm = comm.Split(mycol, myrow)  # rango en col_comm = myrow

    rows = global_indices(N, b, myrow, pr)
    cols = global_indices(N, b, mycol, pc)
    A = matrix_entries(rows, cols, N)

    comm.Barrier()
    t0 = time.perf_counter()
    t = lu_distributed(A, N, b, pr, pc, myrow, mycol, row_comm, col_comm)
    total = time.perf_counter() - t0

    threads = os.environ.get("OMP_NUM_THREADS", "?")
    node = MPI.Get_processor_name()
    data = comm.gather(dict(rank=rank, myrow=myrow, mycol=mycol, nodo=node, total=total, **t), root=0)

    if rank == 0:
        wall = max(x["total"] for x in data)
        comp = [x["lu"] + x["trsm"] + x["gemm"] + x["copia"] for x in data]
        comu = [x["bcast_diag"] + x["bcast_panel"] for x in data]
        nodes = sorted({x["nodo"] for x in data})
        print(f"N={N} b={b} grid={pr}x{pc} T={threads} tiempo={wall:.3f} s "
              f"GFLOP/s={2 * N**3 / 3 / wall / 1e9:.1f} "
              f"computo(prom/max)={np.mean(comp):.3f}/{max(comp):.3f} s "
              f"comunicacion(prom/max)={np.mean(comu):.3f}/{max(comu):.3f} s "
              f"nodos={','.join(nodes)}", flush=True)
        if args.csv:
            new = not os.path.exists(args.csv)
            with open(args.csv, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["etiqueta", "N", "b", "pr", "pc", "T", "rank", "myrow", "mycol",
                                "nodo", "total"] + ETAPAS)
                for x in data:
                    w.writerow([args.etiqueta, N, b, pr, pc, threads, x["rank"], x["myrow"],
                                x["mycol"], x["nodo"], x["total"]] + [x[e] for e in ETAPAS])
    if args.check:
        check(comm, A, N, b, rows, cols)


if __name__ == "__main__":
    main()
