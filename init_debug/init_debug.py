import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import time

def init_weights(tensor, std=0.02):
    return torch.nn.init.normal_(tensor, mean=0.0, std=std)

def main():
    # Initialize distributed training
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    global_rank = int(os.environ['RANK'])
    torch.cuda.set_device(local_rank)
    
    print(f"Rank {global_rank}/{int(os.environ['WORLD_SIZE'])-1} initialized on {os.environ['MASTER_ADDR']}")
    
    # Create model
    model = torch.nn.Linear(10, 10).cuda()
    
    # Initialize 100 times
    init_times = []
    weight_sums = []
    
    for i in range(100):
        start_time = time.time()
        
        # Reinitialize weights
        with torch.no_grad():
            model.weight.data = init_weights(model.weight.data)
    
    print(f"Rank {global_rank} completed successfully")

if __name__ == "__main__":
    import os
    main()
