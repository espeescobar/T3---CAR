#!/bin/bash
# Envia los jobs uno tras otro (limite del curso: 1 job en ejecucion) y espera a cada uno.
# Uso, desde la raiz del repo: BS="512 256" REPS="1 2 3" nohup bash slurm/enviar.sh slurm/c_cfg2_T2.sbatch ... &
mkdir -p salidas/cluster/logs
for job in "$@"; do
  echo "$(date +%T) enviando $job"
  sbatch --wait "$job"
done
echo "$(date +%T) FIN"
