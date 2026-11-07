# Tarea 3 — CAR (IIC3533)

Factorización LU por bloques distribuida con MPI4PY, en distribución bloque-cíclica 2D.

| Ítem | Archivo | Qué hace |
|---|---|---|
| (a) | `lu_serial.py`, `barrido_serial.py` | Versión serial entregada por el curso (sin modificar) y barrido de b |
| (b) | `lu_distributed.py` | LU distribuida; mide cómputo y comunicación por proceso; `--check` compara con la serial |
| (c)–(f) | `slurm/*.sbatch` | Jobs de las configuraciones 0 a 5 en el clúster |
| (e), (g) | `pingpong.py` | Estima latencia α y tiempo por dato β, en un nodo y entre nodos |
| todos | `graficos.py` | Genera figuras y tablas (incluye el modelo de costo de (g)) a partir de los CSV |

Las salidas ya medidas (CSV, logs, tablas y figuras) están en `salidas/`; ver `salidas/README.md`. El informe en PDF no está en este repo.

## Requisitos
```bash
pip install -r requirements.txt
```
Más un MPI. Si no hay uno instalado, `pip install openmpi` lo deja en el mismo entorno.

## En el computador
```bash
python lu_serial.py --N 4096 --b 512 --check
python barrido_serial.py --N 4096 --reps 5 --csv salidas/a_serial_MiPC.csv --maquina MiPC
mpirun -n 8 python lu_distributed.py --N 4096 --b 512 --pr 2 --pc 4 --check
```
Fija los hilos de BLAS con `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` y, en macOS, `VECLIB_MAXIMUM_THREADS`. No uses más procesos × hilos que núcleos.

## En el clúster
Conéctate con `ssh kraken` (ver enunciado). Copia el repo y crea el entorno en el home, porque **`yodaxico` no tiene Open MPI instalado**:
```bash
python3 -m venv ~/t3/venv
~/t3/venv/bin/pip install numpy scipy mpi4py openmpi
```
Los jobs se envían desde la raíz del repo. Solo puede haber 1 job en ejecución, así que `enviar.sh` los manda uno tras otro; `BS` son los valores de b y `REPS` las repeticiones:
```bash
BS="512 1024" nohup bash slurm/enviar.sh slurm/c_cfg2_T2.sbatch slurm/d_cfg*.sbatch slurm/e_cfg*.sbatch > tanda.log 2>&1 &
```
El clúster exige `#SBATCH --comment`, y los scripts ya lo incluyen. Los resultados quedan en `salidas/cluster/`.

## Figuras y tablas
```bash
python graficos.py
```
