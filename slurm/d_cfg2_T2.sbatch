#!/bin/bash
#SBATCH --qos=hpc-iic3533
#SBATCH --partition=hpc-iic3533
#SBATCH --account=default-account
#SBATCH --job-name=t3-d_cfg2_T2
#SBATCH --comment="IIC3533 Tarea 3: factorizacion LU distribuida con MPI4PY"
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=3900M
#SBATCH --time=00:15:00
#SBATCH --output=salidas/cluster/logs/d_cfg2_T2_%j.out

source slurm/comun.sh
for rep in $REPS; do for b in $BS; do
  corre d 8192 $b 2 2 $rep
done; done
