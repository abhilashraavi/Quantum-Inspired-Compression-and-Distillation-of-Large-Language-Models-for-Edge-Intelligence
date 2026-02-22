def compute_sparsity(model):
    total_params = 0
    zero_params = 0
    
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            weight = module.weight
            total_params += weight.numel()
            zero_params += torch.sum(weight == 0).item()
    
    sparsity = zero_params / total_params
    return sparsity * 100
sparsity = compute_sparsity(final_model)
print(f"Sparsity: {sparsity:.2f}%")
