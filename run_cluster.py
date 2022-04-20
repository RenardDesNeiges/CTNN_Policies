import os
import stat
from datetime import datetime
PYTHON_VERSION = "/shared/renard/conda/envs/torchLTC/bin/python"
PROJECT_FOLDER = "/home/renard/Documents/CTNN_Policies"

################ Train parameters

NODE = "dev"
JOBNAME = "LTC_PPO"
MAX_TIME = 10
SCRIPT = "tune_wrapper.py"
SCRIPT_NAME = "run_cluster.sh"

if NODE == "dev":
    cpu = 24
    mem = 22
elif NODE == "prod":
    cpu = 24
    mem = 22
else:   
    raise Exception("Invalid node type : {}".format(NODE))

################ Generating the output folder

datestr = datetime.now().strftime("%Y-%m-%d-%H-%M")
foldername = "/home/renard/Documents/run_archives/R_{}_{}".format(datestr,JOBNAME)
print("Created {} folder".format(foldername))
os.mkdir(foldername)


################ Generating the sbatch script

scriptout = "{}/tune_out.txt".format(foldername)
logpath = "{}/{}.out".format(foldername,JOBNAME)

header =  \
"#!/bin/bash \n\
\n\
#SBATCH --job-name={} \n\
#SBATCH --output={} \n\
\n\
#SBATCH --partition={} \n\
#SBATCH --cpus-per-task={} \n\
#SBATCH --mem={}GB \n\
#SBATCH --nodes=1 \n\
#SBATCH --tasks-per-node=1 \n\
#SBATCH --time=00:{}:00 \n".format(JOBNAME,logpath,NODE,cpu,mem,MAX_TIME)

head_address =  \
'set -x \n\
# __doc_head_address_start__ \n\
\n\
################# DO NOT CHANGE THINGS HERE UNLESS YOU KNOW WHAT YOU ARE DOING ###############\n\
# Getting the node names \n\
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST") \n\
nodes_array=($nodes) \n\
\n\
head_node=${nodes_array[0]}\n\
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)\n\
\n\
# if we detect a space character in the head node IP, well\n\
# convert it to an ipv4 address. This step is optional.\n\
if [[ "$head_node_ip" == *" "* ]]; then\n\
IFS=' ' read -ra ADDR <<<"$head_node_ip"\n\
if [[ ${#ADDR[0]} -gt 16 ]]; then\n\
  head_node_ip=${ADDR[1]}\n\
else\n\
  head_node_ip=${ADDR[0]}\n\
fi\n\
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"\n\
fi\n\
# __doc_head_address_end__ \n'


head_ray =  \
'# __doc_head_ray_start__\n\
port=6379\n\
ip_head=$head_node_ip:$port\n\
export ip_head\n\
echo "IP Head: $ip_head"\n\
\n\
echo "Starting HEAD at $head_node"\n\
##srun --nodes=1 --ntasks=1 -w "$head_node" \n\
##    ray start --head --node-ip-address="$head_node_ip" --port=$port \n\
##    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_TASK}" --block &\n\
srun --nodes=1 --ntasks=1 -w "$head_node" \n\
    /shared/renard/conda/envs/torchLTC/bin/ray start --head --node-ip-address="$head_node_ip" --port=$port \n\
    --num-cpus "${SLURM_CPUS_PER_TASK}" --block &\n\
# __doc_head_ray_end__\n\
\n\
# __doc_worker_ray_start__\n\
# optional, though may be useful in certain versions of Ray < 1.0.\n\
sleep 10\n\
\n\
# number of nodes other than the head node\n\
worker_num=$((SLURM_JOB_NUM_NODES - 1))\n\
\n\
for ((i = 1; i <= worker_num; i++)); do\n\
    node_i=${nodes_array[$i]}\n\
    echo "Starting WORKER $i at $node_i"\n\
    srun --nodes=1 --ntasks=1 -w "$node_i" \n\
        ray start --address "$ip_head" \n\
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_TASK}" --block &\n\
    sleep 5\n\
done\n\
# __doc_worker_ray_end__ \n'


start_run = "{} -u {}/{} > {} \n".format(PYTHON_VERSION,PROJECT_FOLDER,SCRIPT,scriptout)

################ Writing the sbatch script

print("writing sbatch script")
# create the shell script
sbatch_script = header + head_address + head_ray + start_run
script_path = "{}/{}".format(foldername,SCRIPT_NAME)
sbatch_script_file = open(script_path,"x")
sbatch_script_file.write(sbatch_script)
sbatch_script_file.close()

################ Archiving

print("Copying the script file for tracability")
os.system('cp {} {}/archive_{}'.format(SCRIPT,foldername,SCRIPT)) 

################ Running the script

print("running sbatch")
os.system("sbatch {}".format(script_path))