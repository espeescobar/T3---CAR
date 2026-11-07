#!/bin/bash
#SBATCH --qos=hpc-iic3533
#SBATCH --partition=hpc-iic3533
#SBATCH --account=default-account
#SBATCH --job-name=t3-serial
#SBATCH --comment="IIC3533 Tarea 3: factorizacion LU distribuida con MPI4PY"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=3900M
#SBATCH --time=00:15:00
#SBATCH --output=salidas/cluster/logs/serial_%j.out

source slurm/comun.sh
$PY barrido_serial.py --N 4096 --reps 3 --csv $OUT/a_serial_cluster.csv --maquina cluster
$PY barrido_serial.py --N 8192 --reps 3 --csv $OUT/a_serial_cluster.csv --maquina cluster
