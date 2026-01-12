import os
import re
import glob
import sys

RAW_DATA_DIR = 'data/raw/'
PROCESSED_DATA_DIR = 'data/processed/'
CONSOLIDATED_TEXT_PATH = 'data/processed/corpora.txt'
MIN_PARAGRAPH_LENGTH = int(sys.argv[1]) if len(sys.argv) > 1 else 0


def fix_line_breaks(text, apply_regex):
    if apply_regex:
        # 1. Une linhas quebradas dentro de parágrafos
        # Substitui quebras de linha únicas (\n) que não são parte de um parágrafo (\n\n) por espaço
        # (?<!\n) → garante que não há \n antes (não é segunda quebra de linha)
        # (?!\n) → garante que não há \n depois (não é primeira de duas ou mais)
        text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)

        # 2. Marca quebras de linha únicas após ponto para preservá-las temporariamente
        # Isto evita quebras de linha que vêm depois de sentenças acabem sendo unidas por acidente
        text = re.sub(r'\.(\n)(?!\n)', '.<SINGLE_NL>', text)
        
        # 3. Remove espaços antes das quebras de linha
        # [ \t]+ → um ou mais espaços ou tabs
        # (\n) → preserva a quebra de linha
        text = re.sub(r'[ \t]+(\n)', r'\1', text)

        # 4. Remove espaços depois das quebras de linha
        # \n[ \t]+ → pega \n seguido de espaços/tabs
        # Substitui apenas pelo \n
        text = re.sub(r'\n[ \t]+', '\n', text)
        
        # 5. Restaura as quebras de linha únicas após ponto que haviam sido marcadas
        # Substitui <SINGLE_NL> pelo \n original
        text = text.replace('<SINGLE_NL>', '\n')


        # 6. Reduz três ou mais quebras de linha consecutivas para apenas duas
        # Mantém a separação de parágrafos, mas evita excesso de linhas em branco
        text = re.sub(r'\n{3,}', '\n\n', text)

        # 7. Captura do primeiro caractere maiúsculo até o último ponto do texto
        # re.DOTALL → faz o . casar também com \n
        match = re.search(r'[A-Z].*\.', text, re.DOTALL)
        if match: return match.group()
    
    else:
        return text
    

def clean_gutenberg_text(text):
    start_marker = r"\*\*\*\s*START OF (THIS|THE) PROJECT GUTENBERG EBOOK.*\*\*\*"
    end_marker = r"\*\*\*\s*END OF (THIS|THE) PROJECT GUTENBERG EBOOK.*\*\*\*"
    
    start_match = re.search(start_marker, text, re.IGNORECASE)
    if start_match:
        text = text[start_match.end():]
    
    end_match = re.search(end_marker, text, re.IGNORECASE)
    if end_match:
        text = text[:end_match.start()]
        
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    text = text + "<|endoftext|>"
    
    return text


def save_split(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(data)
    print(f"Salvo {len(data.split())} palavras")

    size_bytes = os.path.getsize(path)
    size_mb = size_bytes / (1024 * 1024)    
    print(f"Tamanho do arquivo salvo: ~({size_mb:.2f} MB) em: {path}")
    
    tamanho_bytes = len(data.encode("utf-8"))
    tamanho_mb_utf8 = tamanho_bytes / (1024 * 1024)
    print(f"Tamanho da string em UTF-8 (bytes reais de texto): (~{tamanho_mb_utf8:.4f} MB)")

    tamanho = sys.getsizeof(data)
    tamanho_mb = tamanho / (1024 * 1024)
    print(f"Tamanho em memória (objeto Python): ~{tamanho_mb:.4f} MB)")

    return tamanho_mb


def main(apply_regex=True):
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    all_cleaned_texts = []
    file_paths = glob.glob(os.path.join(RAW_DATA_DIR, '*.txt'))
    file_paths.sort()
    
    if not file_paths:
        print(f"ERRO: Nenhum arquivo .txt encontrado em '{RAW_DATA_DIR}'.")
        print("Por favor, execute 'python scripts/download_data.py' primeiro.")
        return
    print(f"Encontrados {len(file_paths)} arquivos em '{RAW_DATA_DIR}'.")
    
    qtd_palavras_sem_remover_ingles = 0
    qtd_palavras_removendo_ingles = 0

    for filepath in file_paths:
        print(f"\t-> Processando: {os.path.basename(filepath)}")
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                raw_content = f.read()
                qtd_palavras_sem_remover_ingles += len(raw_content.split())
                cleaned_content = clean_gutenberg_text(raw_content)
                qtd_palavras_removendo_ingles += len(cleaned_content.split())
                """
                print("----------------")
                print(len(raw_content.split()))
                print(len(cleaned_content.split()))
                
                visualizar = "" # input() escrever qualquer caractere para mostrar as diferenças do texto 1 que não estão no segundo texto limpo
                if visualizar != "":
                    raw_lines = set(raw_content.splitlines())
                    cleaned_lines = set(cleaned_content.splitlines())
                    only_in_raw = raw_lines - cleaned_lines
                    print("\n".join(only_in_raw))
                """
                if cleaned_content:
                    all_cleaned_texts.append(cleaned_content)
        except Exception as e:
            print(f"\t\tAVISO: Falha ao ler ou processar o arquivo {filepath}. Erro: {e}")

    print(f"\nTotal de palavras antes da limpeza do ingles: ~{qtd_palavras_sem_remover_ingles}")
    print(f"Total de palavras após a limpeza do ingles: ~{qtd_palavras_removendo_ingles}\n")
    
    consolidated_text = "\n\n".join(all_cleaned_texts)
    paragraphs_word_counts = []
    paragraphs_valid = []

    discards = 0
    words_discards = 0
    for p in consolidated_text.split('\n\n'):
        if MIN_PARAGRAPH_LENGTH is not None:
            if len(p.split()) >= MIN_PARAGRAPH_LENGTH:
                paragraphs_word_counts.append(len(p.split()))
                paragraphs_valid.append(p)
            else:
                discards += 1
                words_discards += len(p.split())
    
    print(f"Total de parágrafos descartados: {discards}, totalizando {words_discards} palavras.")
    
    min_words = min(paragraphs_word_counts)
    max_words = max(paragraphs_word_counts)
    print(f"{len(paragraphs_word_counts)} parágrafos existentes, com tamanho variando de {min_words} a {max_words} palavras.")
    if len(paragraphs_word_counts) < 10:
        print("ERRO: O corpus consolidado tem muito poucos parágrafos para ser dividido.")
        return
    
    corpora = "\n\n".join(paragraphs_valid)
    print("Quantidade de palavras totais para treino, teste e validação", len(corpora.split()))
    
    with open(CONSOLIDATED_TEXT_PATH, 'w', encoding='utf-8') as f:
        f.write(corpora)
        print(f"\nCorpus consolidado com ~{len(corpora.split())} palavras salvo em: {CONSOLIDATED_TEXT_PATH}")
    
    print("\n--- Dividindo o corpus em treino, validação e teste ---")
    n_corpora = len(corpora)
    train_end = int(0.8 * n_corpora)
    test_end = int(0.9 * n_corpora)

    train_text = fix_line_breaks(corpora[:train_end], apply_regex)
    val_text = fix_line_breaks(corpora[train_end:test_end], apply_regex)
    test_text = fix_line_breaks(corpora[test_end:], apply_regex)

    total_size = save_split(train_text, os.path.join(PROCESSED_DATA_DIR, "train.txt"))
    total_size += save_split(val_text, os.path.join(PROCESSED_DATA_DIR, "val.txt"))
    total_size += save_split(test_text, os.path.join(PROCESSED_DATA_DIR, "test.txt"))
    
    print("\n--- Preparação de dados concluída com sucesso! ---")
    print(f"O tamanho total dos arquivos processados é de {total_size:.2f} MB em '{PROCESSED_DATA_DIR}'.")


if __name__ == "__main__":
    main(apply_regex=True)