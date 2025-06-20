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
    
    # Initialize 100 times and measure statistics
    init_times = []
    weight_sums = []
    
    for i in range(100):
        start_time = time.time()
        
        # Reinitialize weights
        with torch.no_grad():
            model.weight.data = init_weights(model.weight.data)
        
        # Record initialization time
        init_time = time.time() - start_time
        init_times.append(init_time)
        
        # Calculate sum of weights for verification
        weight_sum = model.weight.data.sum().item()
        weight_sums.append(weight_sum)
        
        if global_rank == 0 and (i+1) % 10 == 0:
            print(f"Initialization {i+1}/100 completed")
    
    # Wrap with DDP (after initialization)
    model = DDP(model, device_ids=[local_rank])
    
    # Example forward pass
    x = torch.randn(5, 10).cuda()
    output = model(x)
    
    # Print statistics on rank 0
    if global_rank == 0:
        print("\nInitialization Statistics:")
        print(f"Average initialization time: {sum(init_times)/len(init_times):.6f}s")
        print(f"Total initialization time: {sum(init_times):.6f}s")
        print(f"Average weight sum: {sum(weight_sums)/len(weight_sums):.6f}")
        print(f"Weight sum stddev: {torch.std(torch.tensor(weight_sums)).item():.6f}")
    
    print(f"Rank {global_rank} completed successfully")

if __name__ == "__main__":
    import os
    main()
