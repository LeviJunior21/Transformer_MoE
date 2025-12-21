import torch, os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from src.utils import text_to_token_ids, token_ids_to_text, tokenizer, generate_text, get_loaders


# Calcula a loss no batch dos dados de treino treino
def calc_loss_batch(model, input_batch, target_batch, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

# Calcula a loss no batch nos dados de validação
def calc_loss_loader(model, data_loader, device, num_batches):
    total_loss = 0.0
    if len(data_loader) == 0: return float('nan')
    num_batches = min(num_batches, len(data_loader))
    data_iter = iter(data_loader)
    for _ in range(num_batches):
        try:
            input_batch, target_batch = next(data_iter)
            loss = calc_loss_batch(model, input_batch, target_batch, device)
            total_loss += loss.item()
        except StopIteration: break
    return total_loss / num_batches if num_batches > 0 else float('nan')

# Calcula a loss nos testes
def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(model, train_loader, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(model, val_loader, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

# Gera uma amostra a partir de uma entrada de prompt para o modelo
def generate_and_print_sample(model, tokenizer, device, start_context, max_new_tokens=50):
    model.eval()
    context_size = model.pos_embeddings.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer, device).to(device)
    with torch.no_grad():
        token_ids = generate_text(model=model, idx=encoded, max_new_tokens=max_new_tokens, context_size=context_size)
    decoded_text = token_ids_to_text(token_ids, tokenizer, device)
    print(f"Amostra Gerada: '{decoded_text.replace(os.linesep, ' ')}'")
    model.train()
    
    
# Calcula a loss via entropia cruzada
def calc_loss_batch_by_cross_entropy(model, input_batch, target_batch, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


# Calcula a loss via entropia cruzada
def calc_loss_batch_by_cross_entropy(model, input_batch, target_batch, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1),
        target_batch.flatten()
    )

    return loss


# Cálculo da perplexidade do modelo, o quão o modelo fica surpreso ao ver um token
def compute_perplexity(model, data_loader, device, print_at=1000):
    model.eval()
    total_loss = 0
    n_batches = 0

    print(f"Métricas impressas a cada {print_at} batches.\n")
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(data_loader):
            x, y = x.to(device), y.to(device)
            loss = calc_loss_batch_by_cross_entropy(model, x, y, device)
            if batch_idx % print_at == 0:
                print(f"Batch {batch_idx+1}/{len(data_loader)}, Loss: {loss.item():.4f}")

            total_loss += loss.item()
            n_batches += 1

            if (batch_idx) % print_at == 0:
                avg_loss_partial = total_loss / n_batches
                print(f"\tMédia parcial até aqui: {avg_loss_partial:.4f}")

    avg_loss = total_loss / n_batches
    perplexity = torch.exp(torch.tensor(avg_loss))
    print(f"Loss média final no dataset de teste: {avg_loss:.4f}")
    print(f"Perplexidade calculada: {perplexity.item():.2f}")

    return perplexity.item()


# Plota o gráfico da loss do treinamento do modelo
def plot_graph(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    ax1.plot(epochs_seen, train_losses, label="Perda no Treino")
    ax1.plot(epochs_seen, val_losses, linestyle="-.",label="Perda na Validação")
    ax1.set_xlabel("Épocas")
    ax1.set_ylabel("Perda")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens Vistos")

    fig.tight_layout()
    plt.savefig("loss-plot.pdf")
    plt.show()