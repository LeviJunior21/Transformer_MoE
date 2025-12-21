import os, time, wandb, torch
from .wandb import send_logger_wandb, fetch_weights_and_bias, load_weights_and_bias
from .metrics import calc_loss_batch, evaluate_model, generate_and_print_sample, plot_graph


# Treino do modelo
def train_model_aux(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer, wandb_run, save_freq_wdb, file_name, save_wdb, state_dict, project, name, start_time, initial_tolerance):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    epochs_complete = state_dict.get("epoch", 0)
    batchs_complete = state_dict.get("batch", 0)
    accumulated_time = state_dict.get("train_time", 0)
    elapsed_time = 0

    for epoch in range(epochs_complete, num_epochs):
        model.train()

        tolerance = initial_tolerance
        previous_val_loss = None

        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            if batch_idx < batchs_complete: 
                global_step += 1
                print(global_step)
                continue

            optimizer.zero_grad()
            loss = calc_loss_batch(model, input_batch, target_batch, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)

                if initial_tolerance:
                    if previous_val_loss is not None and val_loss > previous_val_loss:
                        tolerance -= 1
                        if tolerance < 0:
                            print(f"Tolerância de {initial_tolerance} atingida em {batch_idx}. Últimos pesos salvos em {global_step // save_freq_wdb} batches para {epoch} época(s).")
                            elapsed_time = time.time() - start_time + accumulated_time
                            return train_losses, val_losses, track_tokens_seen, elapsed_time
                    else:
                        tolerance = initial_tolerance
                
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                wandb_run.log({"train_loss": train_loss, "val_loss": val_loss, "tokens_seen": tokens_seen, "epoch": epoch + 1, "global_step": global_step})
                print(f"Ep {epoch+1} (Step {global_step:010d}): \nTrain loss {train_loss:.3f}, Val loss {val_loss:.3f}")

            if save_wdb and global_step > 0 and global_step % save_freq_wdb == 0 and batchs_complete > 0:
                send_logger_wandb(wandb_run, model, epoch, batch_idx, optimizer, start_time, elapsed_time, accumulated_time, name, file_name)

            if batchs_complete == 0:
                batchs_complete = 1

        print(f"\nEXEMPLO DE GERAÇÃO: {generate_and_print_sample(model, tokenizer, device, start_context)}")
        elapsed_time = time.time() - start_time + accumulated_time
        if save_wdb:
            send_logger_wandb(model, epoch, batch_idx, optimizer, start_time, elapsed_time, accumulated_time, name, file_name)

    return train_losses, val_losses, track_tokens_seen, elapsed_time


def train_model(model, optimizer, config, device, train_loader, val_loader):
    run = wandb.init(project=config["project"], name=config["name"], id=config["run_id"], resume="allow")
    res_fetch, path_fetch = fetch_weights_and_bias(user=config["user"], project=config["project"], name=config["name"], version=config["version"], file_name=config["file_name"])
    
    state_dict = {"epoch": 0, "batch": 0, "train_time": 0}
    if res_fetch:
        loaded, checkpoint = load_weights_and_bias(file_name=path_fetch)

        if loaded and checkpoint:
            print(f"Checkpoint encontrado: {config['file_name']}")
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            state_dict["epoch"] = checkpoint.get("epoch", 0)
            state_dict["batch"] = checkpoint.get("batch", 0)
            state_dict["train_time"] = checkpoint.get("train_time", 0)
            if config["is_training"]:
                print(f"Pesos carregados! Retomando da época {state_dict['epoch']} e batch {state_dict['batch']}.")
            else:
                print(f"Pesos carregados, mas como a sua configuração não requer treinamento, então pularemos esta parte.")
        else:
            print("Nenhum checkpoint válido encontrado. Iniciando do zero.")
    else:
        print("Nenhum artefato remoto encontrado. Treinando do zero.")

    if config["is_training"]:
        num_epochs = config["max_epochs"]
        start_time = time.time()
        train_losses, val_loss, tokens_seen, total_train_time = train_model_aux(
            model, train_loader, val_loader, optimizer, device,
            num_epochs=num_epochs, eval_freq=5, eval_iter=5,
            start_context=config["start_context"], tokenizer=config["tokenizer"],
            wandb_run=run, save_freq_wdb=config["save_freq_wdb"], file_name=config["file_name"],
            save_wdb=config["save_wdb"], state_dict=state_dict,
            project=config["project"], name=config["name"], start_time=start_time, initial_tolerance=config["initial_tolerance"]
        )

        print("\nGRÁFICO DE PERDA DURANTE O TREINO:")
        epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
        plot_graph(epochs_tensor, tokens_seen, train_losses, val_loss)
    
        return tokens_seen[-1], total_train_time

    else: return 0, 0
