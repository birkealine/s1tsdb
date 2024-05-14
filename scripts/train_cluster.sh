#!/bin/bash
#SBATCH --time=24:00:00

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=12288

#SBATCH --gpus=1
#SBATCH --gres=gpumem:4g

#SBATCH --output=/dev/null # we define the stdin and stderr below with the run_name
#SBATCH --error=/dev/null
#SBATCH --open-mode=append

#SBATCH -J $1 # the job name

# See 'filename pattern' in https://slurm.schedmd.com/sbatch.html to get abbreviations (eg %x = job name)

################ TINY README ################
#
# /!\ Conda environment and modules need to be activated/loaded before launching the run !! /!\
#
# This script allows to run jobs in the cluster. The equivalent of this command locally:
# `python src/full_run.py run_name=my-run-name arg1=foo arg2=bar`
# is this (cf scrupts/launcher_cluster.sh for examples):
# `sbatch scripts/train_cluster.sh my-run-name arg1=foo arg2=bar`
#############################################


# Get run_name from arguments, and shift all other inputs (could also take ${@: 2} instead of shfting)
run_name=$1
shift;

# Logs
logpath="logs/$run_name"
mkdir -p $logpath
logfile_out="$logpath/out.txt"
logfile_err="$logpath/err.txt"

# Job (stdin and stderr redirected accordingly)
# The "$@" gathers all extra command line arguments after the run_name
python src/full_run.py run_name=$run_name "$@" 1> ${logfile_out} 2> ${logfile_err}