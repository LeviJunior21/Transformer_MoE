import torch
from finetuning.lora import LoRA

def apply_lora(model, rank, alpha_is_equals=False):
    numero_antigo_parametros_treinaveis = sum(p.numel() for p in model.parameters() if p.requires_grad)
    layers = ["wq", "wk", "wv", "wo_proj", "fc1", "fc2", "fc3", "out_final"]

    for module in model.modules():
        for name, child in module.named_children():
            if name in layers and isinstance(child, torch.nn.Linear):
                bias = child.bias is not None
                new_child = LoRA(child, rank, alpha_is_equals, bias)
                setattr(module, name, new_child)

    numero_novo_parametros_treinaveis = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parâmetros treináveis antes da aplicação do LoRA: {numero_antigo_parametros_treinaveis}")
    print(f"Parâmetros treináveis depois da aplicação do LoRA: {numero_novo_parametros_treinaveis}")
    
    return model