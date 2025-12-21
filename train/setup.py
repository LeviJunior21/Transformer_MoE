import sys
import subprocess


def setup_environment(requirements_path: str):
    """Instala dependências do projeto (pip + apt). Compatível com Linux / Kaggle / Colab."""

    def run(cmd):
        subprocess.run(cmd, check=True)

    python = sys.executable  # caminho exato do Python em uso

    print("Atualizando pip...")
    run([python, "-m", "pip", "install", "--upgrade", "pip"])

    print("Instalando requirements.txt...")
    run([python, "-m", "pip", "install", "-r", requirements_path])

    """
    print("Instalando chromium-chromedriver...")
    run(["sudo", "apt-get", "install", "-y", "chromium-chromedriver"])

    print("Instalando selenium...")
    run([python, "-m", "pip", "install", "selenium"])
    """
    
    print("Ambiente configurado com sucesso.")