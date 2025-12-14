import torch
from ..utils.tokenizer import tiktoken, tokenizer
from ..model.model import Transformer
from ..utils.generate import generate_text

config = {
    "vocab_size": tokenizer.n_vocab,
    "embedding_dim": 512,
    "context_length": 256,
    "num_layers": 8,
    "num_heads": 8,
    "bias": False,
    "num_kv_groups": 8,
    "dtype": torch.float32,
    "num_experts": 8,
    "num_experts_per_token": 2,
    "emb_dim_moe": 64,
    "apply_rope": False,
    "rope_base": 10_000,
    "max_iterations": 50000,
    "learning_rate": 0.0003,
    "weight_decay": 0.1,
    "batch_size": 4,
    "max_epochs": 1,
    "eval_freq": 200,
    "eval_iter": 50,
    "start_context": "Se o jardim",
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

torch.manual_seed(123)
tokenizer = tiktoken.get_encoding("o200k_base")
text_encoded = tokenizer.encode(config["start_context"])
text_encoded_tensor = torch.tensor(text_encoded, device=config["device"]).unsqueeze(0)

model_test = Transformer(config, config["device"]).to(config["device"])
result_encoded = generate_text(idx=text_encoded_tensor, model=model_test, max_new_tokens=10, context_size=config["context_length"])

result_encoded_list = result_encoded.squeeze(0).tolist()
result = tokenizer.decode(result_encoded_list)
print(result)

assert result == "Se o jardim spotting classifier Gemeადი hiểm يجوز Faça studenten glossary.sa"
