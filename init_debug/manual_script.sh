module load brics/singularity-multi-node

export NNODES=2
export MASTER_ADDR=your-master-noode-id

srun --overlap --jobid your-master-node-jobid --nodelist=your-master-noode-id --mpi=pmix singularity exec --nv -H your-env-path.sif /bin/bash -c "export TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib.real; \
cd your-project-path; \
echo `pwd`; \
export MASTER_ADDR=your-master-noode-id; \
export NNODES=2; \
export NODE_RANK=0; \
/bin/bash init_debug.sh"

srun --overlap --jobid your-second-node-jobid --nodelist=your-second-noode-id --mpi=pmix singularity exec --nv -H your-env-path.sif /bin/bash -c "export TRITON_LIBCUDA_PATH=/usr/local/cuda/compat/lib.real; \
cd your-project-path; \
echo `pwd`; \
export MASTER_ADDR=your-master-noode-id; \
export NNODES=2; \
export NODE_RANK=1; \
/bin/bash init_debug.sh"
