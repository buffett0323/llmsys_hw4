#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 2:00:00
#SBATCH --gpus=v100-32:1
#SBATCH --output=fk_false_output_%j.log
#SBATCH --error=fk_false_error_%j.log

# load cuda
module load cuda/12.6.1

# activate environment
nvidia-smi

cd /jet/home/bliu10/llmsys_hw4
source .venv/bin/activate

uv pip install -r requirements.txt
uv pip install -Ue .

bash compile_cuda.sh

python -m project.run_machine_translation --use-fused-kernel False
# python -m project.run_machine_translation --use-fused-kernel True
