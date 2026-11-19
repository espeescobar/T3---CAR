# Item (g): modelo de costo de la LU distribuida y su validación.
#
# 1. Ajusta gamma_g, gamma_p y delta con el barrido de (c) (grid 2x2, T=2) y toma alpha y beta
#    del ping-pong (lo hace item_g de graficos.py, que también genera g_modelo y g_pingpong).
# 2. Con esos parámetros fijos, predice el tiempo de las configuraciones de (d) y (e), que no se
#    usaron en el ajuste, y lo compara con lo medido.
# 3. Desglosa el modelo por término para las configuraciones 3, 4 y 5 (cuello de botella).
#
# Uso: python modelo_costo.py   (desde la raíz del repo; lee salidas/cluster/*.csv)
# Salidas: salidas/figuras/g_validacion.png, salidas/figuras/g_terminos_nodos.png,
#          salidas/tablas/g_validacion.{md,tex}, salidas/tablas/g_bstar.{md,tex}
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import graficos as G

N = 8192


def terminos(b, pr, pc, nnodos, h, p):
    """Tiempo de la ruta crítica, separado por término del modelo (en segundos).

    h es cuántos procesos comparten un núcleo físico (escala el cómputo). Con dos nodos, cada
    fila del grid queda en un nodo: row_comm es local y col_comm cruza la red. Por el enlace
    pasan a la vez los paneles de U de todas las columnas de procesos, por eso se usa su suma.
    """
    a_i, b_i, a_e, b_e = p["a_i"], p["b_i"], p["a_e"], p["b_e"]
    w = G.obras(N, b, pr, pc)
    t = dict(producto=h * p["g_g"] * w["gemm"], por_paso=h * p["delta"] * w["memo"],
             paneles=h * p["g_p"] * w["panel"])
    if nnodos == 1:
        t["ancho_banda"] = b_i * (w["bdiag"] + w["bpan"])
        t["latencia"] = a_i * w["nlat"]
        return t
    bw = lat = 0.0
    for bk, r, c, m, n in G.tamanos(N, b, pr, pc):
        bw += b_i * (bk ** 2 + m.max() * bk)            # row_comm: A_kk y panel de L, en el nodo
        bw += b_e * (bk ** 2 + n.sum() * bk)            # col_comm: A_kk y paneles de U, por la red
        hops_col = max(math.ceil(math.log2(pr)) - 1, 0)
        lat += 2 * (a_i * math.ceil(math.log2(pc)) + a_e + a_i * hops_col)
    t["ancho_banda"], t["latencia"] = bw, lat
    return t


def comparticion(pr, pc, T, nnodos):
    """Procesos por núcleo físico. En kraken una CPU de SLURM es un hilo de hardware (2 por núcleo).

    En el ajuste (2x2, T=2) cada proceso tiene un núcleo entero: h=1. Con T=1 y varios procesos,
    dos procesos comparten un núcleo: h=2. Un solo proceso con T=1 tiene el núcleo para él: h=1.
    """
    por_nodo = pr * pc // nnodos
    return 1 if (T >= 2 or por_nodo == 1) else 2


def main():
    p = G.item_g()

    d = G.mejor(G.por_corrida(G.leer_mpi(G.CL / "d.csv")))
    e = G.mejor(G.por_corrida(G.leer_mpi(G.CL / "e.csv")))
    cfg = {(1, 1): 0, (1, 2): 1, (2, 1): 1, (2, 2): 2, (2, 4): 3, (4, 2): 3}
    m = pd.concat([d, e])
    m = m[(m["T"] <= 2) & ~((m.pr == 2) & (m.pc == 2) & (m["T"] == 2))]   # T>=4 y el ajuste, fuera
    filas = []
    for _, x in m.sort_values(["nnodos", "T", "pr", "pc", "b"]).iterrows():
        pr, pc, T, nn, b = int(x.pr), int(x.pc), int(x["T"]), int(x.nnodos), int(x.b)
        h = comparticion(pr, pc, T, nn)
        t = terminos(b, pr, pc, nn, h, p)
        mod = sum(t.values())
        c = cfg[(pr, pc)] if nn == 1 else (4 if pc == 4 and pr == 2 else 5)
        filas.append(dict(Cfg=c, Grid=f"{pr}×{pc}", T=T, Nodos=nn, b=b, h=h,
                          **{"Medido (s)": x.tiempo, "Modelo (s)": mod,
                             "Error": (mod - x.tiempo) / x.tiempo * 100}))
    v = pd.DataFrame(filas)
    G.tex(v, "g_validacion", {"Medido (s)": "{:.2f}", "Modelo (s)": "{:.2f}", "Error": "{:+.0f}%"})
    print(v.to_string(index=False))
    print(f"error absoluto medio: {v.Error.abs().mean():.1f}%  máximo: {v.Error.abs().max():.1f}%")

    # --- Figura 1: modelo frente a medición en configuraciones no usadas en el ajuste.
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    lim = [0, max(v["Medido (s)"].max(), v["Modelo (s)"].max()) * 1.1]
    x = np.linspace(*lim, 50)
    ax.fill_between(x, 0.8 * x, 1.2 * x, color="#e4e3df", alpha=0.7, label="±20 %", linewidth=0)
    ax.plot(x, x, color=G.GRIS, linewidth=1.2, linestyle="--", label="Modelo = medido")
    grupos = [("1 nodo, T = 1", (v.Nodos == 1) & (v["T"] == 1), G.AZUL, "o"),
              ("1 nodo, T = 2", (v.Nodos == 1) & (v["T"] == 2), G.AQUA, "s"),
              ("2 nodos, T = 1", v.Nodos == 2, G.NARANJA, "D")]
    for lab, sel, col, mk in grupos:
        ax.plot(v[sel]["Medido (s)"], v[sel]["Modelo (s)"], mk, color=col, label=lab,
                markeredgecolor="white", markeredgewidth=1.5, markersize=9)
    for _, r in v[(v.Nodos == 2) | ((v.Grid == "2×4") & (v.Nodos == 1))].iterrows():
        dos = r.Nodos == 2
        ax.annotate(f"{r.Grid}, b={r.b}, 2 nodos" if dos else f"2×4 y 4×2, b={r.b}",
                    (r["Medido (s)"], r["Modelo (s)"]), textcoords="offset points",
                    xytext=(-10, 2) if dos else (10, -6), ha="right" if dos else "left",
                    fontsize=9.5, color=G.TINTA)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("Tiempo medido (s)")
    ax.set_ylabel("Tiempo del modelo (s)")
    ax.set_title("(g) Validación: configuraciones de (d) y (e)")
    ax.legend(loc="upper left")
    G.guardar(fig, "g_validacion")

    # --- Figura 2: aporte de cada término en las configuraciones 3, 4 y 5 (b=512).
    casos = [("Cfg 3\n2×4, 1 nodo", 2, 4, 1), ("Cfg 4\n2×4, 2 nodos", 2, 4, 2), ("Cfg 5\n4×4, 2 nodos", 4, 4, 2)]
    nombres = [("producto", r"Producto ($\gamma_g$)", G.AZUL), ("por_paso", r"Por elemento y paso ($\delta$)", G.NARANJA),
               ("paneles", r"Paneles y $A_{kk}$ ($\gamma_p$)", G.AQUA), ("ancho_banda", r"Ancho de banda ($\beta$)", G.VIOLETA),
               ("latencia", r"Latencia ($\alpha$)", G.MAGENTA)]
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    xs = np.arange(len(casos))
    medidos = []
    for i, (lab, pr, pc, nn) in enumerate(casos):
        t = terminos(512, pr, pc, nn, comparticion(pr, pc, 1, nn), p)
        base = 0.0
        for k, nom, col in nombres:
            ax.bar(i, t[k], 0.55, bottom=base, color=col, label=nom if i == 0 else None,
                   edgecolor="white", linewidth=1.5)
            base += t[k]
        med = v[(v.Grid == f"{pr}×{pc}") & (v.Nodos == nn) & (v.b == 512)]["Medido (s)"].iloc[0]
        medidos.append(med)
        ax.text(i, base + 0.12, f"modelo {base:.2f} s", ha="center", fontsize=10, color=G.TINTA)
    ax.plot(xs, medidos, "D", color=G.TINTA, markersize=9, label="Medido", markeredgecolor="white")
    ax.set_xticks(xs, [c[0] for c in casos])
    ax.set_ylabel("Tiempo (s)")
    ax.set_ylim(0, max(medidos) * 1.3)
    ax.set_title("(g) Aporte de cada término del modelo, b = 512")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    G.guardar(fig, "g_terminos_nodos")
    for lab, pr, pc, nn in casos:
        t = terminos(512, pr, pc, nn, comparticion(pr, pc, 1, nn), p)
        print(lab.replace("\n", " "), {k: round(val, 3) for k, val in t.items()})
    for a_e in (p["a_e"], 9.465e-05):
        q = dict(p, a_e=a_e)
        print(f"alpha_inter={a_e:.2e}: latencia cfg4 b=512 = {terminos(512, 2, 4, 2, 2, q)['latencia']:.3f} s")

    # --- b* del modelo continuo según el grid, frente al mejor b medido.
    bs = []
    for (pr, pc), T in (((1, 1), 1), ((1, 2), 1), ((2, 2), 2), ((2, 4), 1), ((4, 2), 1)):
        bstar = math.sqrt(2 * p["delta"] * N / (3 * p["g_p"] * (pr + pc)))
        q = d[(d.pr == pr) & (d.pc == pc) & (d["T"] == T)]
        bm = int(q.loc[q.tiempo.idxmin(), "b"]) if len(q) else None
        disc = min((512, 1024), key=lambda b: sum(terminos(b, pr, pc, 1, comparticion(pr, pc, T, 1), p).values()))
        bs.append({"Grid": f"{pr}×{pc}", "T": T, "b* continuo": bstar, "Mejor b del modelo (512 o 1024)": disc,
                   "Mejor b medido (512 o 1024)": bm})
    bs = pd.DataFrame(bs)
    G.tex(bs, "g_bstar", {"b* continuo": "{:.0f}"})
    print(bs.to_string(index=False))


if __name__ == "__main__":
    main()
