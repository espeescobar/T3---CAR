# Salidas ya medidas

Mediciones del 7/11/2026 con N = 2^12 en el computador y N = 2^13 en el clúster (salvo donde se indica). Sirven para generar figuras y tablas sin correr nada.

| Máquina | Detalle |
|---|---|
| `MacBookM2` | Apple M2, 8 núcleos (4 P + 4 E), 8 GB, Accelerate. Ver `maquina_MacBookM2.txt` |
| `cluster` | `ventress` y `yodaxico`, Xeon E5-2640 v4, OpenBLAS. Una CPU de SLURM es un hilo de hardware. Ver `cluster/maquina_kraken.txt` |

| Archivo | Ítem | Contenido |
|---|---|---|
| `a_serial_MacBookM2.csv`, `.txt` | (a) | Barrido de b con `lu_serial.py`, N = 2^12, 5 repeticiones |
| `b_verificacion_MacBookM2.txt` | (b) | 72 comparaciones contra la versión serial y la prueba con un error introducido |
| `b_local_MacBookM2.csv`, `.txt` | (b) | Tiempos locales por grid |
| `cluster/a_serial_cluster.csv` | (a), (c), (d) | Barrido serial en el clúster, N = 2^12 y 2^13, 3 repeticiones |
| `cluster/prueba.csv` | (b) | Corridas con `--check` en el clúster; el resultado está en `cluster/logs/prueba_*.out` |
| `cluster/c.csv` | (c), (f) | Grid 2×2, T = 2, barrido de b |
| `cluster/d.csv` | (d) | Configuraciones 0 a 3 con b = 512 y 1024 |
| `cluster/e.csv` | (e) | Configuraciones 4 y 5 con b = 512 y 1024 |
| `cluster/pingpong.csv` | (e), (g) | Tiempos por tamaño de mensaje, una corrida por par (`alpha`, `beta`). La primera corrida `inter` fue anómala (α ≈ 5 ms); se usa la mediana |
| `cluster/logs/` | todos | Salida de cada job |
| `figuras/`, `tablas/` | todos | Lo que genera `graficos.py` (tablas en `.tex` y `.md`; parámetros del modelo en `g_parametros.txt`) |

Formato de los CSV de MPI: una fila por proceso y corrida. `etiqueta` = `nodos,repetición`; `lu`, `trsm`, `gemm`, `copia` son cómputo y `bcast_diag`, `bcast_panel` son comunicación, en segundos.
