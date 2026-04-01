# Transformers - Decoder Only: LLM usando a Arquitetura Grouped Query Attention, Mixture of Experts, KV-Cache e LoRA
Este projeto tem como objetivo treinar um modelo de linguagem baseado no **Decoder-only** para geração de texto em português, utilizando como corpus em português. A arquitetura foi adaptada para tarefas de geração com avaliação qualitativa e checkpoints salvos ao longo do treinamento.

## 📁 Estrutura do projeto

```
deeplearning-final/
├── finetuning/             # Script contendo modulos de pós-treinamento como Fine-Tuning, Low Rank Adaptation etc.
├── scripts/                # Scripts de ingestão e pré-processamento dos dados, removendo ruídos e padronizando para o pré-treinamento do modelo.
├── src/
│   ├── model/              # Arquitetura customizada do modelo
│   ├── test/               # Teste dos modulos da arquiteturs do modelo
│   └── utils/              # Funções auxiliares para gerar texto, plotar métricas, carregar datasets e converter texto para token-ids e vice-versa.
├── train/                  # Loop de treinamento e avaliação
├── data/                   # Corpus limpo e dividido
├── wandb/                  # Modelos salvos por época e Logs de treinamento
├── Dockerfile              # Arquivo Docker para inicializar o treinamento
├── README.md               # Este arquivo
└── requirements.txt        # Dependências do projeto
```

## 🚀 Execução

O treinamento pode estar sendo realizado tanto em uma VM, quanto um notebook do Kaggle (aproveitando os recursos gratuitos de GPU). Para reproduzir:

### 1. Acesse o notebook no Kaggle

#### 1.1 - Dependências

- Instale os pacotes necessários com:

```bash
pip install -r requirements.txt
```

#### 1.2 - Baixar o notebook Kaggle

- Em breve (privado).

#### 1.3 - Execute as células na ordem para:

- Baixar e limpar os dados
- Inicializar o modelo
- Treinar e salvar checkpoints

#### 2. Usando uma VM

- Use o Dockerfile para reproduzir usando o código abaixo:

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
```

**Principais bibliotecas:**
- **selenium** - Raspagem de livros em português
- **torch** – Treinamento do modelo
- **tiktoken** – Tokenização eficiente
- **requests, tqdm** – Download e progresso
- **wandb** - Salvamento dos pesos do modelo

## 📚 Dados
Os textos de 768 livros foram extraídos do [Projeto Gutenberg](https://www.gutenberg.org/) e processados para remover metadados, normalizar pontuação e dividir em parágrafos com tamanho mínimo.

Você também pode configurar o dataset de pré-treinamento em scripts/download_data.py, alterando o código para carregar seus dados.

## Referências
1. [Attention Is All You Need (Arxiv)](https://arxiv.org/abs/1706.03762)
2. [RoFormer: Enhanced Transformer with Rotary Position Embedding (Arxiv)](https://arxiv.org/pdf/2104.09864)
3. [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)
4. [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
5. [Build a Large Language Model (From Scratch)](https://github.com/rasbt/LLMs-from-scratch)
6. [OUTRAGEOUSLY LARGE NEURAL NETWORKS: THE SPARSELY-GATED MIXTURE-OF-EXPERTS LAYER (Arxiv)](https://arxiv.org/pdf/1706.03762)
7. [LoRA: Low-Rank Adaptation of Large Language Models (Arxiv)](https://arxiv.org/pdf/2106.09685)
