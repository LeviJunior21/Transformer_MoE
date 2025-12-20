import os
import torch
from torch.utils.data import Dataset, DataLoader


class DatasetGPT(Dataset):
    def __init__(self, text, tokenizer, max_length, stride, set_name):
        allowed_special = {"<|endoftext|>"}
        self.tokens = tokenizer.encode(text, allowed_special=allowed_special)

        self.max_length = max_length
        self.stride = stride
        self.set_name = set_name

        print(
            f"Dataset {set_name} pronto | "
            f"Tokens: {len(self.tokens)} | "
            f"Amostras: {self.__len__()}"
        )

    def __len__(self):
        return max(0, (len(self.tokens) - self.max_length - 1) // self.stride)

    def __getitem__(self, idx):
        start = idx * self.stride
        end = start + self.max_length

        x = torch.tensor(self.tokens[start:end], dtype=torch.long)
        y = torch.tensor(self.tokens[start + 1:end + 1], dtype=torch.long)

        return x, y


def load_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def print_loader_info(title, loader):
    x, _ = next(iter(loader))
    print(f"- {title}")
    print(f"\tTotal de amostras: {len(loader.dataset)}")
    print(f"\tTokens por amostra: {x.shape[1]}")
    print(f"\tBatch size: {loader.batch_size}")
    print(f"\tNúmero de batches: {len(loader)}")


def create_dataloader(
    text,
    tokenizer,
    max_length,
    stride,
    batch_size,
    shuffle,
    drop_last,
    num_workers,
    set_name,
):
    dataset = DatasetGPT(
        text=text,
        tokenizer=tokenizer,
        max_length=max_length,
        stride=stride,
        set_name=set_name,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Dataloader {set_name} pronto!")
    return dataloader


def get_loaders(
    data_path,
    tokenizer,
    max_length=256,
    batch_size=4,
    num_workers=2,
    stride=None,
):
    # STRIDE SEGURO PARA LLM
    if stride is None:
        stride = max_length

    print("Carregando arquivos...")
    train_text = load_file(os.path.join(data_path, "train.txt"))
    val_text = load_file(os.path.join(data_path, "val.txt"))
    test_text = load_file(os.path.join(data_path, "test.txt"))

    print("Arquivos carregados.\n")

    train_loader = create_dataloader(
        text=train_text,
        tokenizer=tokenizer,
        max_length=max_length,
        stride=stride,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        set_name="TREINAMENTO",
    )

    val_loader = create_dataloader(
        text=val_text,
        tokenizer=tokenizer,
        max_length=max_length,
        stride=stride,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        set_name="VALIDAÇÃO",
    )

    test_loader = create_dataloader(
        text=test_text,
        tokenizer=tokenizer,
        max_length=max_length,
        stride=stride,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        set_name="TESTE",
    )

    print()
    print_loader_info("Treino", train_loader)
    print_loader_info("Validação", val_loader)
    print_loader_info("Teste", test_loader)
    print()

    return train_loader, val_loader, test_loader
