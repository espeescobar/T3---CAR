#!/bin/bash
#SBATCH --qos=hpc-iic3533
#SBATCH --partition=hpc-iic3533
#SBATCH --account=default-account
#SBATCH --job-name=t3-prueba
#SBATCH --comment="IIC3533 Tarea 3: factorizacion LU distribuida con MPI4PY"
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=3900M
#SBATCH --time=00:15:00
#SBATCH --output=salidas/cluster/logs/prueba_%j.out

source slurm/comun.sh
lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core|Socket|L2|L3"
free -g | head -2
mpirun -n $SLURM_NTASKS --map-by slot:PE=1:hwtcpus --bind-to hwthread sh -c "grep Cpus_allowed_list /proc/self/status"
$PY -c "import numpy,scipy,mpi4py,sys;print(sys.version.split()[0],numpy.__version__,scipy.__version__,mpi4py.__version__);numpy.show_config()" | grep -E "^[0-9]|name" | head -6
corre prueba 1024 128 2 4 1 --check
corre prueba 4096 256 2 4 1 --check
corre prueba 4096 256 4 2 1 --check
