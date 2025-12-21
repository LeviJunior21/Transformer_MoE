# Transformers: Geração de Texto com Arquitetura Grouped Query Attention e Mixture of Experts
Este projeto tem como objetivo treinar um modelo de linguagem baseado no **Decoder-only** para geração de texto em português, utilizando como corpus em português. A arquitetura foi adaptada para tarefas de geração com avaliação qualitativa e checkpoints salvos ao longo do treinamento.

## 📁 Estrutura do projeto

```
deeplearning-final/
├── notebooks/              # Notebooks de experimentação e visualização
├── scripts/                # Scripts de ingestão e pré-processamento
├── src/
│   ├── model/              # Arquitetura customizada do modelo
│   └── utils/              # Funções auxiliares
├── train/                  # Loop de treinamento e avaliação
├── data/                   # Corpus limpo e dividido
├── checkpoints/            # Modelos salvos por época
├── logs/                   # Logs de treinamento
├── requirements.txt        # Dependências do projeto
└── README.md               # Este arquivo
```

## 🚀 Execução

O treinamento está sendo realizado em um notebook do Kaggle, aproveitando os recursos gratuitos de GPU. Para reproduzir:

1. Acesse o notebook no Kaggle
   - Em breve (privado). Disponível apenas no relatório.

2. Execute as células na ordem para:
   - Baixar e limpar os dados
   - Inicializar o modelo
   - Treinar e salvar checkpoints

## 📦 Dependências
Instale os pacotes necessários com:

```bash
pip install -r requirements.txt
```

ou

```
az vm show -d -g MeuGrupo -n minhavm --query publicIps -o tsv
chmod 600 /home/levi/Downloads/minhavm_key.pem
ssh -i /home/levi/Downloads/minhavm_key.pem azureuser@20.197.240.84
yes

git clone https://github.com/LeviJunior21/Transformer_MoE.git
cd Transformer_MoE

sudo apt update
sudo apt install -y docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker $USER

sudo docker build -t transformer_moe .
sudo docker run --rm -it transformer_moe
```**Principais bibliotecas:**
- **selenium** - Raspagem de livros em português
- **torch** – Treinamento do modelo
- **tiktoken** – Tokenização eficiente
- **requests, tqdm** – Download e progresso
- **transformers** – Baseado em Decoder-only
- **wandb** - Salvamento dos pesos do modelo

## 📚 Dados
Os textos foram extraídos do [Projeto Gutenberg](https://www.gutenberg.org/) e processados para remover metadados, normalizar pontuação e dividir em parágrafos com tamanho mínimo.

## Referências
1. [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
2. [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
3. [Build a Large Language Model (From Scratch)](https://github.com/rasbt/LLMs-from-scratch)
4. [OUTRAGEOUSLY LARGE NEURAL NETWORKS: THE SPARSELY-GATED MIXTURE-OF-EXPERTS LAYER (Arxiv)](https://arxiv.org/pdf/1701.06538)
