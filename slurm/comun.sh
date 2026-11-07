# Funciones comunes de los jobs. Se carga con: source slurm/comun.sh
# Hilos de BLAS = $SLURM_CPUS_PER_TASK, como pide el enunciado.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
# Open MPI y Python de un entorno en el home, compartido por los nodos (yodaxico no trae MPI).
VENV=${VENV:-$HOME/t3/venv}
export PATH=$VENV/bin:$PATH
PY=$VENV/bin/python
OUT=salidas/cluster
mkdir -p $OUT
echo "job=$SLURM_JOB_ID nodos=$SLURM_JOB_NODELIST ntasks=$SLURM_NTASKS cpus_por_tarea=$SLURM_CPUS_PER_TASK"

# corre <item> <N> <b> <pr> <pc> <rep> [opciones extra]
corre () {
  local item=$1 N=$2 b=$3 pr=$4 pc=$5 rep=$6; shift 6
  mpirun -n $SLURM_NTASKS --map-by slot:PE=$SLURM_CPUS_PER_TASK:hwtcpus --bind-to hwthread \
    $PY lu_distributed.py --N $N --b $b --pr $pr --pc $pc \
    --csv $OUT/$item.csv --etiqueta "$SLURM_NNODES,$rep" "$@"
}
REPS=${REPS:-1 2 3 4 5}   # repeticiones; se pueden completar por tandas
