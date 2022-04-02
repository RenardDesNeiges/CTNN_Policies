#!/bin/bash

#################################### NODE RESOURCE LIMITS: 
#################################### dev  node: 24 cpus, 22GB ram
#################################### prod node: 40 cpus, 30GB ram
#################################### gpu  node: 16 cpus, 64GB ram (so less)
#################################### name: il, drl, cpgrl
#################################### DO NOT put in anything GPU related UNLESS using GPU node (not supported on other nodes)

#SBATCH --job-name=rltest
#SBATCH --output=rltest.log

################################### PROD

#SBATCH --partition=prod
#SBATCH --gpus-per-task=0
#SBATCH --cpus-per-task=40
#SBATCH --mem=30GB
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
###SBATCH --time=00:10:00

################################### DEV

### Give all resources to a single Ray task, ray can manage the resources internally
#####SBATCH --ntasks-per-node=1
#####SBATCH --gpus-per-task=0
#####SBATCH --gpus-per-task=1

set -x
# __doc_head_address_start__

## Load modules or your own conda environment here
## module load pytorch/v1.4.0-gpu
## conda activate {{CONDA_ENV}}
. ~/clenv/bin/activate
python_out="rltest.txt"

################# DO NOT CHANGE THINGS HERE UNLESS YOU KNOW WHAT YOU ARE DOING ###############
# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi
# __doc_head_address_end__

# __doc_head_ray_start__
port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
##srun --nodes=1 --ntasks=1 -w "$head_node" \
##    ray start --head --node-ip-address="$head_node_ip" --port=$port \
##    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_TASK}" --block &
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --block &
# __doc_head_ray_end__

# __doc_worker_ray_start__
# optional, though may be useful in certain versions of Ray < 1.0.
sleep 10

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_TASK}" --block &
    sleep 5
done
# __doc_worker_ray_end__

# __doc_script_start__
##############################################################################################

#### call your code below
python -u /home/username/.../run_rllib.py > $python_out 