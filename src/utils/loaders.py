import os
import torch
from torch.utils.data import Dataset, DataLoader


class Dataset_GPT(Dataset):
    def __init__(self, text, tokenizer, stride, max_length, set):
        self.input_ids = []
        self.target_ids = []

        allowed_special = { '<|endoftext|>' }
        tokens = tokenizer.encode(text, allowed_special=allowed_special)

        for i in range(0, len(tokens) - max_length, stride):
            self.input_ids.append(torch.tensor(tokens[i: i + max_length]))
            self.target_ids.append(torch.tensor(tokens[i+1: i+max_length + 1]))
        
        print(f"Dataset de {set} está pronto!")


    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

    def __len__(self):
        return len(self.input_ids)
    

def create_dataset(text, stride, max_length, shuffle, drop_last, tokenizer, num_workers, batch_size, set):
    dataset = Dataset_GPT(
        text=text,
        tokenizer=tokenizer,
        stride=stride,
        max_length=max_length,
        set=set
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Dataloader de {set} está pronto!")
    return dataloader


def load_file(file_path):
    with open(file_path, 'r',encoding='utf-8') as file:
        content = file.read()
    return content


def print_loader_info(title, loader, init=False):
    print(f"- {title}")
    print(f"\tTotal de amostras: {loader.dataset.__len__()}")
    print(f"\tTokens em cada amostra: {len(loader.dataset.__getitem__(0)[0])}")
    print(f"\tNúmero de batches: {len(loader)}")
    print(f"\tNúmero de amostras por batch: {loader.batch_size}")


def get_loaders(data_path, tokenizer, max_length = 256, batch_sz = 10, num_workers = 4, stride=1):
    train_data = load_file(os.path.join(data_path, "train.txt"))
    test_data = load_file(os.path.join(data_path, "test.txt"))
    val_data = load_file(os.path.join(data_path, "val.txt"))

    print("Arquivos carregados!\n")
    
    train_loader = create_dataset(
        text=train_data,
        max_length=max_length,
        stride=stride,
        batch_size=batch_sz,
        num_workers=num_workers,
        tokenizer=tokenizer,
        drop_last=True,
        shuffle=True,
        set="TREINAMENTO"
    )

    print("Train Loader está pronto!")

    test_loader = create_dataset(
        text=test_data,
        max_length=max_length,
        stride=stride,
        batch_size=batch_sz,
        num_workers=num_workers,
        tokenizer=tokenizer,
        drop_last=True,
        shuffle=True,
        set="TESTE"
    )
    
    print("Test Loader está pronto!")

    val_loader = create_dataset(
        text=val_data,
        max_length=max_length,
        stride=stride,
        batch_size=batch_sz,
        num_workers=num_workers,
        tokenizer=tokenizer,
        drop_last=True,
        shuffle=True,
        set="VALIDAÇÃO"
    )
    
    print("Val Loader está pronto!")

    
    print("\n")
    print_loader_info("Treino", train_loader)
    print_loader_info("Teste", test_loader)
    print_loader_info("Validação", val_loader)
    print("\n")
    
    return train_loader, test_loader, val_loader
