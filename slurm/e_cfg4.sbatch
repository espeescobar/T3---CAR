#!/bin/bash
#SBATCH --qos=hpc-iic3533
#SBATCH --partition=hpc-iic3533
#SBATCH --account=default-account
#SBATCH --job-name=t3-e_cfg4
#SBATCH --comment="IIC3533 Tarea 3: factorizacion LU distribuida con MPI4PY"
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --mem=3900M
#SBATCH --time=00:15:00
#SBATCH --output=salidas/cluster/logs/e_cfg4_%j.out

source slurm/comun.sh
# Con 4 procesos por nodo, rank//pc separa los nodos: en 2x4 cada fila del grid
# queda en un nodo (row_comm intra-nodo, col_comm entre nodos).
for rep in $REPS; do for b in $BS; do
  corre e 8192 $b 2 4 $rep
done; done
