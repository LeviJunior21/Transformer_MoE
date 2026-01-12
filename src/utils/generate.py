import torch
import torch.nn.functional as F


def generate_text(model, idx, max_new_tokens, context_size=50, reset=False):
    model.eval()
    reset = False

    try:
        for i in range((max_new_tokens + 1) if reset else max_new_tokens):
            idx_cond = idx[:, -context_size:]
            with torch.no_grad():
                logits = model(idx_cond, reset) if reset else model(idx_cond)
                if logits == None: continue

            reset = False

            logits = logits[:, -1, :]
            probas = torch.nn.functional.softmax(logits, dim=-1)
            idx_next_token = torch.argmax(probas, dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next_token), dim=1)

        return idx

    except Exception as e:
        print(e)
        return idx


def generate_text_top(model, prompt, max_new_tokens, tokenizer, device, context_size=50, temperature=0.8, top_k=50, top_p=0.95, repetition_penalty=1.2):
    model.eval()
    ids = tokenizer.encode(prompt)
    idx = torch.tensor(ids).unsqueeze(0).to(device)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        for token_id in idx[0].tolist():
            logits[:, token_id] /= repetition_penalty

        logits = logits / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")

        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
            sorted_indices_to_remove = cumulative_probs > top_p        
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
        
            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
            indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, idx_next), dim=1)

    result = idx
    return tokenizer.decode(result[0].tolist())
