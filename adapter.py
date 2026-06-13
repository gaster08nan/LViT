import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model


def add_peft_adapter(
    model: nn.Module,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules = None,
    verbose: bool = True
) -> nn.Module:
    """
    Wraps the LViT model with Low-Rank Adaptation (LoRA) using the PEFT library.
    
    By default, it targets the projection layers in all Multi-Head Attention blocks 
    and Feed-Forward MLP blocks inside the Vision Transformer modules.

    Args:
        model (nn.Module): The base LViT model to be wrapped.
        r (int): The rank of the low-rank adaptation matrices (bottleneck dimension).
        lora_alpha (int): The scaling factor for the LoRA adapter.
        lora_dropout (float): Dropout probability for LoRA layers.
        target_modules (list or None): Specific submodule names to inject LoRA into. 
                                      If None, defaults to MHA and MLP layers.
        verbose (bool): If True, prints parameter reduction statistics.

    Returns:
        nn.Module: The model wrapped with PEFT.
    """
    if target_modules is None:
        # Target the Query, Key, Value and Output projections in Multi-Head Attention
        # and the fc1, fc2 layers in MLP blocks.
        target_modules = ["query", "key", "value", "out", "fc1", "fc2"]
        
    peft_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        modules_to_save=[]  # Add target modules like outc if you want to keep them fully trainable
    )
    
    peft_model = get_peft_model(model, peft_config)
    
    if verbose:
        print_trainable_parameters(peft_model)
        
    return peft_model


def print_trainable_parameters(model: nn.Module):
    """
    Helper to print the number of trainable parameters vs total parameters.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    
    reduction_percentage = 100 * trainable_params / all_param
    print(
        f"trainable params: {trainable_params:,} || "
        f"all params: {all_param:,} || "
        f"trainable%: {reduction_percentage:.4f}%"
    )
