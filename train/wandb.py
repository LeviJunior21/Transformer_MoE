import wandb, torch, time, os


# Carrega o modelo salvo na nuvem (wandb)
def fetch_weights_and_bias(user, project, name, version, file_name):
    try:
        api = wandb.Api()
        version_tag = version
        artifact = api.artifact(f"{user}/{project}/{name}:{version_tag}", type="model")
        artifact_dir = artifact.download()
        file_path = os.path.join(artifact_dir, file_name)
        print(f"Fetch success -> {file_path}")
        return True, file_path

    except Exception as e:
        print(f"Fetch Error: {e}")
        return False, ""


# Carrega o modelo salvo localmente
def load_weights_and_bias(file_name):
    try:
        checkpoint = torch.load(file_name, map_location="cpu")
        return True, checkpoint
    except Exception as e:
        print(f"Load error: {e}")
        return False, {}


# Envia metricas de logs do treino para a plataforma wandb.
def send_logger_wandb(wandb_run, model, epoch, batch_idx, optimizer, start_time, elapsed_time, accumulated_time, name, file_name):
    elapsed_time = time.time() - start_time + accumulated_time
    torch.save({"epoch": epoch, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "batch": batch_idx, "train_time": elapsed_time},  file_name)
    artifact = wandb.Artifact(name, type="model")
    artifact.add_file(file_name)
    wandb_run.log_artifact(artifact)