import os
import sys
import subprocess
import random


def clone_and_setup_repo(repo_dir: str, username: str, token: str | None = None, seed: int = 123):
    """Clona um repositório GitHub e adiciona <repo_dir>/src ao sys.path"""

    if token:
        repo_url = f"https://{username}:{token}@github.com/{username}/{repo_dir}.git"
    else:
        repo_url = f"https://github.com/{username}/{repo_dir}.git"

    # Remove repositório existente
    if os.path.exists(repo_dir):
        print("Removendo repositório existente...")
        subprocess.run(["rm", "-rf", repo_dir], check=True)

    # Clona o repositório
    print("Clonando repositório...")
    subprocess.run(["git", "clone", repo_url, repo_dir], check=True)

    # Adiciona src ao sys.path
    project_path = os.path.join(os.getcwd(), repo_dir, "src")
    if project_path not in sys.path:
        sys.path.append(project_path)

    print(f"Repositório clonado e caminho adicionado ao sys.path:")
    print(project_path)

    random.seed(seed)
    print(f"Seed configurada: {seed}")

    return project_path
