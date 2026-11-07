#!/bin/bash
#SBATCH --qos=hpc-iic3533
#SBATCH --partition=hpc-iic3533
#SBATCH --account=default-account
#SBATCH --job-name=t3-g_pingpong_inter
#SBATCH --comment="IIC3533 Tarea 3: factorizacion LU distribuida con MPI4PY"
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --mem=3900M
#SBATCH --time=00:15:00
#SBATCH --output=salidas/cluster/logs/g_pingpong_inter_%j.out

source slurm/comun.sh
mpirun -n 2 --map-by slot:PE=1:hwtcpus --bind-to hwthread $PY pingpong.py --csv $OUT/pingpong.csv --etiqueta inter
