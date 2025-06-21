#!/bin/bash
module load brics/singularity-multi-node

# Launch master node (rank 0)
srun --jobid=master-jobid --nodelist=node1 --mpi=pmix \
  singularity exec --nv your-image.sif \
  torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
           --master_addr=node1 --master_port=1234 \
           train.py &

# Launch worker node (rank 1) 
srun --jobid=worker-jobid --nodelist=node2 --mpi=pmix \
  singularity exec --nv your-image.sif \
  torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 \
           --master_addr=node1 --master_port=1234 \
           train.py &
