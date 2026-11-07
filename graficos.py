# Lee los CSV de salidas/ y genera las figuras (salidas/figuras/) y tablas (salidas/tablas/).
# Uso: python graficos.py
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

SAL = Path("salidas")
CL = SAL / "cluster"
FIG = SAL / "figuras"
TAB = SAL / "tablas"
FIG.mkdir(exist_ok=True)
TAB.mkdir(exist_ok=True)

AZUL, NARANJA, AQUA, AMARILLO, MAGENTA, VERDE, VIOLETA = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6")
TINTA, GRIS = "#0b0b0b", "#52514e"
plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 14, "axes.titleweight": "bold", "axes.labelsize": 13,
    "legend.fontsize": 11, "xtick.labelsize": 11.5, "ytick.labelsize": 11.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": GRIS,
    "axes.grid": True, "axes.axisbelow": True, "grid.color": "#e4e3df", "grid.linewidth": 0.8,
    "legend.frameon": False, "figure.dpi": 110, "savefig.dpi": 220, "savefig.bbox": "tight",
    "lines.linewidth": 2, "lines.markersize": 8, "font.family": "DejaVu Sans",
})


def guardar(fig, nombre):
    fig.savefig(FIG / f"{nombre}.png")
    plt.close(fig)


def log2x(ax, valores):
    ax.set_xscale("log", base=2)
    ax.set_xticks(valores)
    ax.set_xticklabels([f"$2^{{{int(math.log2(v))}}}$" for v in valores])
    ax.minorticks_off()
    ax.grid(axis="x", visible=False)


def leer_mpi(path):
    """Una fila por proceso y corrida -> columnas derivadas por proceso."""
    d = pd.read_csv(path)
    d[["nnodos", "rep"]] = d["etiqueta"].str.split(",", expand=True).astype(int)
    d["comp"] = d[["lu", "trsm", "gemm", "copia"]].sum(axis=1)
    d["comu"] = d[["bcast_diag", "bcast_panel"]].sum(axis=1)
    return d


CFG = ["N", "b", "pr", "pc", "T", "nnodos"]


def por_corrida(d):
    """Una fila por corrida: tiempo real (máximo entre procesos), promedios y máximos."""
    g = d.groupby(CFG + ["rep"])
    return g.agg(tiempo=("total", "max"), comp=("comp", "mean"), comu=("comu", "mean"),
                 comp_max=("comp", "max"), comu_max=("comu", "max"), comp_min=("comp", "min"),
                 comu_min=("comu", "min")).reset_index()


def mejor(r):
    """Por configuración, la repetición más rápida y la dispersión entre repeticiones."""
    i = r.groupby(CFG)["tiempo"].idxmin()
    m = r.loc[i].set_index(CFG)
    s = r.groupby(CFG)["tiempo"]
    m["reps"] = s.count()
    m["disp"] = (s.max() - s.min()) / s.min() * 100
    return m.reset_index()


def tex(df, nombre, fmt=None):
    """Tabla booktabs y copia en Markdown."""
    fmt = fmt or {}
    t = df.copy()
    for c in t.columns:
        if c in fmt:
            t[c] = t[c].map(lambda x: fmt[c].format(x) if pd.notna(x) else "--")
    t = t.astype(str)
    cols = "".join("l" if t[c].str.contains(r"[a-zA-Z×]").any() else "r" for c in t.columns)
    lin = [rf"\begin{{tabular}}{{{cols}}}", r"\toprule", " & ".join(t.columns) + r" \\", r"\midrule"]
    lin += [" & ".join(row) + r" \\" for row in t.values]
    lin += [r"\bottomrule", r"\end{tabular}"]
    (TAB / f"{nombre}.tex").write_text("\n".join(lin).replace("%", r"\%").replace("×", r"$\times$").replace("β", r"$\beta$").replace("α", r"$\alpha$") + "\n")
    md = ["| " + " | ".join(t.columns) + " |", "|" + "---|" * len(t.columns)]
    md += ["| " + " | ".join(row) + " |" for row in t.values]
    (TAB / f"{nombre}.md").write_text("\n".join(md).replace("$", "") + "\n")


def apilado(ax, x, comp, comu, ancho=0.7, etq=True, horiz=False):
    """Barras apiladas de cómputo y comunicación, con el total sobre cada barra."""
    ax.bar(x, comp, ancho, color=AZUL, label="Cómputo", edgecolor="white", linewidth=1.5)
    ax.bar(x, comu, ancho, bottom=comp, color=NARANJA, label="Comunicación", edgecolor="white", linewidth=1.5)
    ax.grid(axis="x", visible=False)


# ---------------------------------------------------------------- (a)
def item_a():
    m2 = pd.read_csv(SAL / "a_serial_MacBookM2.csv")
    cl = pd.read_csv(CL / "a_serial_cluster.csv")
    bs = sorted(m2.b.unique())
    res = []
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    for d, lab, col, mk, N in ((m2, "MacBook Air M2 (Accelerate)", AZUL, "o", 4096),
                               (cl[cl.N == 4096], "Clúster, Xeon E5-2640 v4 (OpenBLAS)", NARANJA, "s", 4096)):
        g = d.groupby("b").tiempo.min()
        ax.plot(g.index, g.values, marker=mk, color=col, label=lab)
        ax.annotate(f"mín: b = {g.idxmin()}\n{g.min():.2f} s", (g.idxmin(), g.min()), xytext=(0, 14 if col == NARANJA else -34),
                    textcoords="offset points", ha="center", fontsize=11, color=TINTA)
    log2x(ax, bs)
    ax.set_yscale("log")
    ax.set_yticks([0.3, 0.5, 1, 2, 3])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax.set_ylim(0.14, 6)
    ax.set_xlabel("Tamaño de bloque b")
    ax.set_ylabel("Tiempo (s, escala log)")
    ax.set_title("(a) LU serial con N = $2^{12}$ y 1 hilo")
    ax.legend(loc="upper center", ncol=1)
    guardar(fig, "a_serial")
    t = pd.DataFrame({"b": bs})
    gm = m2.groupby("b").tiempo.agg(["min", "median"])
    gc = cl[cl.N == 4096].groupby("b").tiempo.min()
    g8 = cl[cl.N == 8192].groupby("b").tiempo.min()
    t["M2 mín (s)"] = gm["min"].values
    t["M2 mediana (s)"] = gm["median"].values
    t["M2 GFLOP/s"] = 2 * 4096 ** 3 / 3 / gm["min"].values / 1e9
    t["Clúster N=4096 (s)"] = gc.reindex(bs).values
    t["Clúster N=8192 (s)"] = g8.reindex(bs).values
    tex(t, "a_serial", {c: "{:.3f}" for c in ("M2 mín (s)", "M2 mediana (s)", "Clúster N=4096 (s)")} |
        {"M2 GFLOP/s": "{:.1f}", "Clúster N=8192 (s)": "{:.2f}"})
    return int(gm["min"].idxmin())


# ---------------------------------------------------------------- (b)
def item_b():
    d = pd.read_csv(SAL / "b_local_MacBookM2.csv")
    d["rep"] = d.etiqueta
    d["nnodos"] = 1
    d["comp"] = d[["lu", "trsm", "gemm", "copia"]].sum(axis=1)
    d["comu"] = d[["bcast_diag", "bcast_panel"]].sum(axis=1)
    m = mejor(por_corrida(d))
    m = m[m.b == 512].copy()
    orden = [(1, 1), (1, 2), (2, 1), (2, 2), (2, 4), (4, 2)]
    m = pd.concat([m[(m.pr == a) & (m.pc == c)] for a, c in orden])
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    x = np.arange(len(m))
    apilado(ax, x, m.comp, m.comu)
    for i, t in enumerate(m.tiempo):
        ax.text(i, m.comp.iloc[i] + m.comu.iloc[i] + 0.012, f"{t:.2f} s", ha="center", fontsize=11)
    ax.set_xticks(x, [f"{a}×{c}" for a, c in orden])
    ax.set_xlabel("Grid $p_r × p_c$")
    ax.set_ylabel("Tiempo promedio por proceso (s)")
    ax.set_title("(b) LU distribuida en el M2: N = $2^{12}$, b = 512, T = 1")
    ax.set_ylim(0, (m.comp + m.comu).max() * 1.28)
    ax.legend(loc="upper left", ncol=2)
    guardar(fig, "b_local")
    tex(pd.DataFrame({"Grid": [f"{a}×{c}" for a, c in orden], "Tiempo (s)": m.tiempo.values,
                      "Cómputo prom. (s)": m.comp.values, "Comunicación prom. (s)": m.comu.values}),
        "b_local", {"Tiempo (s)": "{:.3f}", "Cómputo prom. (s)": "{:.3f}", "Comunicación prom. (s)": "{:.3f}"})


# ---------------------------------------------------------------- (c)
def item_c():
    r = por_corrida(leer_mpi(CL / "c.csv"))
    m = mejor(r).sort_values("b")
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    x = np.arange(len(m))
    apilado(ax, x, m.comp, m.comu)
    ax.plot(x, m.tiempo, "D", color=TINTA, ms=7, label="Tiempo total", linestyle="none")
    for i, t in enumerate(m.tiempo):
        ax.text(i, max(t, m.comp.iloc[i] + m.comu.iloc[i]) + 0.35, f"{t:.2f}", ha="center", fontsize=11)
    ax.set_xticks(x, [f"$2^{{{int(math.log2(b))}}}$" for b in m.b])
    ax.set_xlabel("Tamaño de bloque b")
    ax.set_ylabel("Tiempo (s)")
    ax.set_title("(c) Grid 2×2, T = 2, N = $2^{13}$: tiempo según b")
    ax.set_ylim(0, m.tiempo.max() * 1.22)
    ax.legend(loc="upper center", ncol=3)
    guardar(fig, "c_barrido_b")
    t = pd.DataFrame({"b": m.b.values, "Tiempo (s)": m.tiempo.values, "GFLOP/s": 2 * 8192 ** 3 / 3 / m.tiempo.values / 1e9,
                      "Cómputo prom. (s)": m.comp.values, "Comunic. prom. (s)": m.comu.values,
                      "Dispersión": m.disp.values})
    tex(t, "c_barrido_b", {"Tiempo (s)": "{:.2f}", "GFLOP/s": "{:.1f}", "Cómputo prom. (s)": "{:.2f}",
                           "Comunic. prom. (s)": "{:.2f}", "Dispersión": "{:.1f}%"})
    return int(m.loc[m.tiempo.idxmin(), "b"])


# ---------------------------------------------------------------- (d)
def item_d():
    m = mejor(por_corrida(leer_mpi(CL / "d.csv")))
    m["p"] = m.pr * m.pc
    m["cpus"] = m.p * m["T"]
    T1 = m[(m.p == 1) & (m["T"] == 1)].tiempo.min()
    m["S"] = T1 / m.tiempo
    m["E"] = m.S / m.cpus
    cfgs = {(1, 1): 0, (1, 2): 1, (2, 1): 1, (2, 2): 2, (2, 4): 3, (4, 2): 3}
    m["cfg"] = [cfgs[(a, c)] for a, c in zip(m.pr, m.pc)]
    m = m.sort_values(["cfg", "pr", "T", "b"])
    filas = []
    for (c, pr, pc, T), g in m.groupby(["cfg", "pr", "pc", "T"], sort=True):
        f = {"Cfg": c, "Grid": f"{pr}×{pc}", "T": T, "CPUs": pr * pc * T}
        for b in sorted(g.b):
            h = g[g.b == b].iloc[0]
            f |= {f"T{b}": h.tiempo, f"S{b}": h.S, f"E{b}": h.E}
        filas.append(f)
    t = pd.DataFrame(filas)
    bs = sorted(m.b.unique())
    ren = {}
    fmt = {}
    for b in bs:
        ren |= {f"T{b}": f"t(s) b={b}", f"S{b}": f"S b={b}", f"E{b}": f"E b={b}"}
        fmt |= {f"t(s) b={b}": "{:.2f}", f"S b={b}": "{:.2f}", f"E b={b}": "{:.2f}"}
    tex(t.rename(columns=ren), "d_configs", fmt)

    b0 = 512
    q = m[m.b == b0]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.3))
    series = [(1, "p = 1 (grid 1×1)", AZUL, "o"), (2, "p = 2 (mejor grid)", NARANJA, "s"),
              (4, "p = 4 (grid 2×2)", AQUA, "^"), (8, "p = 8 (mejor grid)", VIOLETA, "D")]
    for p, lab, col, mk in series:
        g = q[q.p == p].groupby("cpus").agg(S=("S", "max"), E=("E", "max")).reset_index()
        a1.plot(g.cpus, g.S, marker=mk, color=col, label=lab, linestyle="-" if len(g) > 1 else "none")
        a2.plot(g.cpus, g.E, marker=mk, color=col, label=lab, linestyle="-" if len(g) > 1 else "none")
    a1.plot([1, 8], [1, 8], "--", color=GRIS, lw=1.3, label="Ideal")
    a1.axhline(4, color=GRIS, lw=1, ls=":")
    a1.text(1.05, 4.08, "4 núcleos físicos", color=GRIS, fontsize=10.5)
    a2.axhline(1, ls="--", color=GRIS, lw=1.3)
    for a in (a1, a2):
        a.set_xscale("log", base=2)
        a.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
        a.minorticks_off()
        a.set_xlabel("CPUs totales ($p × T$)")
    a1.set_ylabel("Speedup S")
    a2.set_ylabel("Eficiencia E")
    a1.set_ylim(0, 8.4)
    a2.set_ylim(0, 1.1)
    a1.set_title(f"(d) Speedup, b = {b0}, N = $2^{{13}}$")
    a2.set_title(f"(d) Eficiencia, b = {b0}, N = $2^{{13}}$")
    a1.legend(loc="upper left", bbox_to_anchor=(0, 1.0), fontsize=10)
    guardar(fig, "d_speedup")

    q = q.sort_values(["cpus", "p"])
    fig, ax = plt.subplots(figsize=(10.4, 4.2))
    x = np.arange(len(q))
    apilado(ax, x, q.comp, q.comu)
    for i, t_ in enumerate(q.tiempo):
        ax.text(i, q.comp.iloc[i] + q.comu.iloc[i] + 0.25, f"{t_:.1f}", ha="center", fontsize=10.5)
    ax.set_xticks(x, [f"{a}×{c}\nT={t_}" for a, c, t_ in zip(q.pr, q.pc, q["T"])], fontsize=10.5)
    ax.set_xlabel("Configuración (grid y hilos), ordenada por CPUs totales")
    ax.set_ylabel("Tiempo promedio por proceso (s)")
    ax.set_title(f"(d) Cómputo y comunicación por configuración, b = {b0}")
    ax.set_ylim(0, (q.comp + q.comu).max() * 1.15)
    ax.legend(loc="upper right", ncol=2)
    guardar(fig, "d_comp_comu")
    return T1, m


# ---------------------------------------------------------------- (e)
def item_e(T1, d):
    e = mejor(por_corrida(leer_mpi(CL / "e.csv")))
    c3 = d[(d.pr == 2) & (d.pc == 4)]
    filas = pd.concat([c3.assign(cfg=3), e.assign(cfg=[4 if p == 2 else 5 for p in e.pr])])
    filas["S"] = T1 / filas.tiempo
    filas["cpus"] = filas.pr * filas.pc
    filas["E"] = filas.S / filas.cpus
    filas = filas.sort_values(["cfg", "b"])
    tex(pd.DataFrame({
        "Cfg": filas.cfg.values, "Grid": [f"{a}×{c}" for a, c in zip(filas.pr, filas.pc)],
        "Nodos": filas.nnodos.values, "b": filas.b.values, "Tiempo (s)": filas.tiempo.values,
        "Cómputo (s)": filas.comp.values, "Comunic. (s)": filas.comu.values,
        "S": filas.S.values, "E": filas.E.values}),
        "e_nodos", {"Tiempo (s)": "{:.2f}", "Cómputo (s)": "{:.2f}", "Comunic. (s)": "{:.2f}", "S": "{:.2f}", "E": "{:.2f}"})
    bs = sorted(filas.b.unique())
    fig, ejes = plt.subplots(1, len(bs), figsize=(10.4, 4.3), sharey=True)
    for ax, b in zip(np.atleast_1d(ejes), bs):
        q = filas[filas.b == b]
        x = np.arange(len(q))
        apilado(ax, x, q.comp, q.comu)
        for i, t_ in enumerate(q.tiempo):
            ax.text(i, q.comp.iloc[i] + q.comu.iloc[i] + 0.5, f"{t_:.1f} s", ha="center", fontsize=11)
        ax.set_xticks(x, [f"Cfg {c}: {a}×{p}\n{n} nodo{'s' if n > 1 else ''}\n{a * p} CPUs"
                          for c, a, p, n in zip(q.cfg, q.pr, q.pc, q.nnodos)], fontsize=10.5)
        ax.set_title(f"(e) b = {b}, N = $2^{{13}}$")
    np.atleast_1d(ejes)[0].set_ylabel("Tiempo promedio por proceso (s)")
    top = (filas.comp + filas.comu).max()
    np.atleast_1d(ejes)[0].set_ylim(0, top * 1.22)
    fig.legend(*np.atleast_1d(ejes)[0].get_legend_handles_labels(), loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.07))
    guardar(fig, "e_nodos")
    return filas


# ---------------------------------------------------------------- (f)
def item_f():
    d = leer_mpi(CL / "c.csv")
    r = por_corrida(d)
    N = 8192
    sel = []
    for b in (N // 2, N // 16):
        q = r[r.b == b]
        sel.append(d[(d.b == b) & (d.rep == q.loc[q.tiempo.idxmin(), "rep"])].sort_values("rank"))
    etapas = [("lu", "lu_factor", VIOLETA), ("trsm", "solve_triangular", AQUA), ("gemm_c", "Producto @ y copias", AZUL),
              ("bcast_diag", "Bcast de $A_{kk}$", AMARILLO), ("bcast_panel", "Bcast de paneles", NARANJA)]
    fig, ejes = plt.subplots(1, 2, figsize=(10.4, 4.4), sharey=True)
    filas = []
    for ax, g in zip(ejes, sel):
        g = g.assign(gemm_c=g.gemm + g.copia)
        x = np.arange(4)
        base = np.zeros(4)
        for c, lab, col in etapas:
            ax.bar(x, g[c], 0.68, bottom=base, color=col, label=lab, edgecolor="white", linewidth=1.5)
            base += g[c].values
        b = g.b.iloc[0]
        for i, v in enumerate(base):
            ax.text(i, v + 0.15, f"{v:.2f} s", ha="center", fontsize=11)
        ax.set_xticks(x, [f"({a},{c})" for a, c in zip(g.myrow, g.mycol)])
        ax.set_xlabel("Proceso (fila, columna)")
        den = "2" if b == N // 2 else "16"
        ax.set_title(f"(f) b = N/{den} = {b}: {g.total.max():.2f} s")
        ax.grid(axis="x", visible=False)
        for _, h in g.iterrows():
            filas.append({"b": b, "Proceso": f"({h.myrow},{h.mycol})", "lu (s)": h.lu, "trsm (s)": h.trsm,
                          "gemm+copia (s)": h.gemm_c, "Cómputo (s)": h.comp, "Bcast diag (s)": h.bcast_diag,
                          "Bcast panel (s)": h.bcast_panel, "Comunic. (s)": h.comu})
    ejes[0].set_ylabel("Tiempo del proceso (s)")
    ejes[0].set_ylim(0, 13.5)
    fig.legend(*ejes[0].get_legend_handles_labels(), loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.1), fontsize=10.5)
    guardar(fig, "f_procesos")
    tex(pd.DataFrame(filas), "f_procesos", {c: "{:.2f}" for c in pd.DataFrame(filas).columns if "(s)" in c})


# ---------------------------------------------------------------- (g)
def tamanos(N, b, pr, pc):
    """Por paso k: (bk, r, c, filas por fila de procesos, columnas por columna de procesos)."""
    nb = (N + b - 1) // b
    for k in range(nb):
        bk = min(b, N - k * b)
        m = np.zeros(pr)
        n = np.zeros(pc)
        for I in range(k + 1, nb):
            s = min(b, N - I * b)
            m[I % pr] += s
            n[I % pc] += s
        yield bk, k % pr, k % pc, m, n


def obras(N, b, pr, pc):
    """Trabajo de la ruta crítica en cada término (paso a paso, procesos sincronizados por Bcast)."""
    w = dict(panel=0.0, gemm=0.0, memo=0.0, bdiag=0.0, bpan=0.0, nlat=0.0)
    for bk, r, c, m, n in tamanos(N, b, pr, pc):
        w["panel"] += (2 / 3) * bk ** 3
        # El proceso más cargado en paneles: (r, c) resuelve ambos si tiene sus bloques.
        tr = np.outer(np.zeros(pr), np.zeros(pc))
        tr[r, :] += n * bk ** 2
        tr[:, c] += m * bk ** 2
        w["panel"] += tr.max()
        area = np.outer(m, n)
        w["gemm"] += 2 * bk * area.max()
        w["memo"] += area.max()
        w["bdiag"] += 2 * bk ** 2 if pr * pc > 1 else 0
        w["bpan"] += (m.max() + n.max()) * bk
        w["nlat"] += 2 * (math.ceil(math.log2(pr)) + math.ceil(math.log2(pc)))
    return w


def item_g():
    # Mediana entre corridas: la primera corrida entre nodos fue anómala (alpha de 5 ms).
    pp = pd.read_csv(CL / "pingpong.csv").drop_duplicates(["etiqueta", "alpha"])
    pp = pp.groupby("etiqueta")[["alpha", "beta"]].median()
    a_i, b_i = pp.loc["intra", ["alpha", "beta"]]
    a_e, b_e = pp.loc["inter", ["alpha", "beta"]]
    d = leer_mpi(CL / "c.csv")
    r = por_corrida(d)
    filas = []
    for b, q in r.groupby("b"):
        rep = q.loc[q.tiempo.idxmin(), "rep"]
        g = d[(d.b == b) & (d.rep == rep)]
        filas.append(dict(b=b, tiempo=g.total.max(), lu=g.lu.sum(), trsm=g.trsm.sum(), gemm=g.gemm.sum(),
                          comu=g.comu.mean()))
    f = pd.DataFrame(filas).sort_values("b")
    N, pr, pc = 8192, 2, 2
    # gamma_g y delta: tiempo del producto = flops * gamma_g + elementos actualizados * delta.
    tot_f = []
    tot_m = []
    tot_p = []
    for b in f.b:
        fg = fm = fp = 0.0
        for bk, r_, c_, m, n in tamanos(N, b, pr, pc):
            nk = m.sum()
            fg += 2 * bk * nk ** 2
            fm += nk ** 2
            fp += (2 / 3) * bk ** 3 + 2 * nk * bk ** 2
        tot_f.append(fg); tot_m.append(fm); tot_p.append(fp)
    tot_f, tot_m, tot_p = map(np.array, (tot_f, tot_m, tot_p))
    A = np.c_[tot_f, tot_m] / f.gemm.values[:, None]
    (g_g, delta), *_ = np.linalg.lstsq(A, np.ones(len(f)), rcond=None)
    A2 = (tot_p / (f.lu + f.trsm).values)[:, None]
    g_p = np.linalg.lstsq(A2, np.ones(len(f)), rcond=None)[0][0]

    pred = []
    for b in f.b:
        w = obras(N, b, pr, pc)
        pred.append(dict(b=b, panel=g_p * w["panel"], gemm=g_g * w["gemm"], memo=delta * w["memo"],
                         lat=a_i * w["nlat"], beta=b_i * (w["bdiag"] + w["bpan"])))
    P = pd.DataFrame(pred)
    P["total"] = P[["panel", "gemm", "memo", "lat", "beta"]].sum(axis=1)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.4))
    bs = list(f.b)
    a1.plot(bs, f.tiempo, "o", color=TINTA, label="Medido")
    a1.plot(bs, P.total, "-", color=AZUL, label="Modelo")
    log2x(a1, bs)
    a1.set_xlabel("Tamaño de bloque b")
    a1.set_ylabel("Tiempo (s)")
    a1.set_ylim(0, max(f.tiempo.max(), P.total.max()) * 1.2)
    a1.set_title("(g) Modelo y medición, grid 2×2, T = 2")
    a1.legend(loc="upper center", ncol=2)
    fig.subplots_adjust(wspace=0.27)
    terms = [("gemm", r"Producto: $\gamma_g\,\frac{2N^3}{3p}$", AZUL, "o"),
             ("memo", r"Costo por elemento y paso: $\delta\,\frac{N^3}{3bp}$", NARANJA, "s"),
             ("panel", r"Paneles: $\gamma_p\,(\cdots)\,b$", AQUA, "^"),
             ("beta", r"Ancho de banda: $\beta\,(\cdots)$", VIOLETA, "D"),
             ("lat", r"Latencia: $\alpha\,\frac{N}{b}\,2\log_2 p$", MAGENTA, "v")]
    for c, lab, col, mk in terms:
        a2.plot(bs, P[c], marker=mk, color=col, label=lab, markersize=6)
    log2x(a2, bs)
    a2.set_yscale("log")
    a2.set_ylim(1e-5, 30)
    a2.set_xlabel("Tamaño de bloque b")
    a2.set_ylabel("Aporte al tiempo (s, escala log)")
    a2.set_title("(g) Aporte de cada término")
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=10)
    guardar(fig, "g_modelo")
    tex(pd.DataFrame({"b": P.b, "Medido (s)": f.tiempo.values, "Modelo (s)": P.total, "Producto (s)": P.gemm,
                      "Por paso (s)": P.memo, "Paneles (s)": P.panel, "β (s)": P.beta, "α (s)": P["lat"]}),
        "g_modelo", {"Medido (s)": "{:.2f}", "Modelo (s)": "{:.2f}", "Producto (s)": "{:.2f}", "Por paso (s)": "{:.2f}",
                     "Paneles (s)": "{:.2f}", "β (s)": "{:.3f}", "α (s)": "{:.5f}"})
    params = (f"gamma_g={g_g:.4e} s/flop ({1e-9 / g_g:.1f} GFLOP/s)\n"
              f"delta={delta:.4e} s/elemento\n"
              f"gamma_p={g_p:.4e} s/flop ({1e-9 / g_p:.1f} GFLOP/s)\n"
              f"alpha_intra={a_i:.3e} s, beta_intra={b_i:.3e} s/double ({8e-9 / b_i:.2f} GB/s)\n"
              f"alpha_inter={a_e:.3e} s, beta_inter={b_e:.3e} s/double ({8e-9 / b_e:.3f} GB/s)\n"
              f"b* continuo={math.sqrt((2 / 3) * delta * N / (g_p * 4)):.0f}\n")
    e = mejor(por_corrida(leer_mpi(CL / "e.csv")))
    c3 = mejor(por_corrida(leer_mpi(CL / "d.csv")))
    for b in (512, 1024):
        extra = (e[(e.pr == 2) & (e.b == b)].comu.iloc[0]
                 - c3[(c3.pr == 2) & (c3.pc == 4) & (c3.b == b)].comu.iloc[0])
        pred = b_e * (N ** 2 / 2 + (N / b) * b ** 2)
        params += f"cfg4-cfg3 b={b}: comunicacion extra medida={extra:.2f} s, beta_inter*(N^2/2+N*b)={pred:.2f} s\n"
    (TAB / "g_parametros.txt").write_text(params)
    print(params)
    return dict(g_g=g_g, delta=delta, g_p=g_p, a_i=a_i, b_i=b_i, a_e=a_e, b_e=b_e)


def item_pingpong():
    p = pd.read_csv(CL / "pingpong.csv")
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for e, lab, col, mk in (("intra", "Mismo nodo", AZUL, "o"), ("inter", "Entre nodos", NARANJA, "s")):
        g = p[p.etiqueta == e]
        g = g.groupby("n").t.median()
        ax.loglog(g.index, g.values * 1e6, marker=mk, color=col, label=lab, base=10)
    ax.set_xlabel("Tamaño del mensaje (doubles)")
    ax.set_ylabel("Tiempo de un mensaje (µs)")
    ax.set_title("Ping-pong en el clúster: latencia y ancho de banda")
    ax.legend(loc="upper left")
    guardar(fig, "g_pingpong")


if __name__ == "__main__":
    print("b(a) =", item_a())
    item_b()
    print("b(c) =", item_c())
    T1, d = item_d()
    print("T(1) =", T1)
    item_e(T1, d)
    item_f()
    item_g()
    item_pingpong()
