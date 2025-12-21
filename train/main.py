from .setup import setup_environment
setup_environment(requirements_path="requirements.txt")

import os, torch, time
from src.model.model import Transformer
from src.utils import tokenizer, get_loaders
from .train_model import train_model
from .github import clone_and_setup_repo
from .prepare_data import download_books

SEED = 123
MAX_MB = 10000
torch.manual_seed(SEED)
MIN_PARAGRAPH_LENGTH = 0
download_books_drive = True
download_books_scraping = False

project_path = clone_and_setup_repo(
    repo_dir="Trabalho",
    username="LeviJunior21",
    token="",
    seed=SEED
)

download_books(
    download_books_drive=True,
    download_books_scraping=False,
    repo_dir="Trabalho",
    max_mb=MAX_MB,
    seed=SEED,
    min_paragraph_length=MIN_PARAGRAPH_LENGTH,
)

try:
    import torch_xla.core.xla_model as xm
    device = xm.xla_device()
except ImportError:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Usando dispositivo: {device.type}")


CONFIG = {
    "tokenizer": tokenizer,
    "vocab_size": tokenizer.n_vocab,
    "embedding_dim": 512,
    "context_length": 256,
    "num_layers": 8,
    "num_heads": 8,
    "bias": False,
    "num_kv_groups": 8,
    "num_experts": 8,
    "num_experts_per_token": 2,
    "emb_dim_moe": 64,
    "apply_rope": True,
    "rope_base": 10_000,
    "is_training": True,
    "learning_rate": 3e-4,
    "weight_decay": 0.1,
    "batch_size": 10,
    "max_epochs": 5,
    "num_workers": 2,
    "stride": 1,
    "initial_tolerance": None,
    "dtype": torch.float32,
    "device": device,
    "eval_freq": 5,
    "eval_iter": 5,
    "start_context": "Se o jardim",
    "save_wdb": True,
    "save_freq_wdb": 10000,
    "user": "levi-pereira-junior-ufcg",
    "project": "transformer_tcc_project",
    "name": "transformer_tcc_base_project",
    "run_id": "transformer_tcc_project_run1",
    "version": "v0",
    "file_name": "mini_mlp.pth"
}

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.environ["WANDB_API_KEY"] = "5ca14c4352d864370bad3199b26c5dab929ba976"

train_loader, test_loader, val_loader = get_loaders(
    data_path="data/processed",
    tokenizer=tokenizer,
    max_length=CONFIG["context_length"],
    batch_size=CONFIG["batch_size"],
    num_workers=CONFIG["num_workers"],
    stride=CONFIG["stride"]
)
print(f"Tamanho do conjunto de treinamento: {len(train_loader)}\nTamanho do conjunto de teste: {len(test_loader)}\nTamanho do conjunto de validação: {len(val_loader)}")

model = Transformer(CONFIG, device).to(device=device)
optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], weight_decay=CONFIG["weight_decay"])

params = sum(p.numel() for p in model.parameters())
params_gpt2 = params - sum(p.numel() for p in model.out_head.parameters())
print(f"Número de parâmetros (sem head): {params_gpt2:,}")
if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)

start_time = time.time()
tokens_processed, total_train_time = train_model(model=model, optimizer=optimizer, config=CONFIG, device=device, train_loader=train_loader, val_loader=val_loader)
end_time = time.time()

elapsed = end_time - start_time
tokens_per_sec = tokens_processed / elapsed
max_memory = "N/A (XLA - TPU)" if "xla" in device.type else torch.cuda.max_memory_allocated(device) / (1024**2) if torch.cuda.is_available() else "N/A (CPU)"

if CONFIG["is_training"]:
    print("DESEMPENHO:")
    print(f"Tempo total: {elapsed:.2f} s")
    print(f"Tokens/s: {tokens_per_sec:.2f}")
    print(f"Memória máxima: {max_memory:.2f} MB" if isinstance(max_memory, float) else f"Memória máxima: {max_memory} MB")
    print(f"\nTempo total de treino: {total_train_time:.2f} s ({total_train_time/60:.2f} min)")