from transformer import *
import wandb
tokenized_dataset_path = 'train_tinystories.bin'
val_path = 'val_tinystories.bin'
import time

val_every = 50
everyN_savecheckpoint = 500
wandb.init(
    project='cs336-assignment1-train-transformer-experiments',
    name='experiment-1-default',
    config={
        'maxlrate':1e-3,
        'vocab_size':10000,
        'context_length': 256,
        'd_model': 512,
        'd_ff': 1344,
        'rope_theta': 10000,
        'num_layers': 4,
        'num_heads': 16,
        'batch_size':32,
        'num_steps': 5000,
        'minlrate': 1e-4,
        'warmupiters': 200,    # watch this. play it
        'coscycleiters': 5000,
        'device': 'mps',
        'max_l2_norm':1,
        'optimizer': 'AdamW'
    }
)

config = wandb.config

dataset = np.memmap(tokenized_dataset_path, dtype=np.uint16, mode='r')

val_dataset = np.memmap(val_path, dtype=np.uint16, mode='r')




model = TransformerLM(config.vocab_size, config.context_length,config.num_layers,config.d_model, config.num_heads, config.d_ff,config.rope_theta ).to(config.device)
adamw_optimizer = AdamW(model.parameters(), config.maxlrate)

starttime = time.time()
for t in range(config.num_steps):
    model.train()
    lrate = learning_rate_schedule(t, config.maxlrate, config.minlrate, config.warmupiters, config.coscycleiters)
    for g in adamw_optimizer.param_groups: g["lr"] = lrate

    inputs, targets = data_loading(dataset, config.batch_size, config.context_length, device=config.device)
    logits = model.forward(inputs)
    logits = einops.rearrange(logits, "batch seq vocab -> (batch seq) vocab")
    targets = einops.rearrange(targets, "batch seq -> (batch seq)")
    loss = cross_entropy(logits, targets)
    print(f'step: {t} - loss: {loss.item()}')
    adamw_optimizer.zero_grad()
    loss.backward()
    gradient_clipping(model.parameters(), config.max_l2_norm)
    adamw_optimizer.step()
    wandb.log({
        "train/loss": loss.item(),
        "learning_rate": lrate,
        "step": t,
        "wall_time": time.time() - starttime
    })

    if t % val_every == 0:
        model.eval()
        with torch.no_grad():
            val_losses = []
            inputs_val, targets_val = data_loading(
                    val_dataset,
                    config.batch_size,
                    config.context_length,
                    device=config.device
                )
            logits = model(inputs_val)
            logits = einops.rearrange(logits, "batch seq vocab -> (batch seq) vocab")
            targets_val = einops.rearrange(targets_val, "batch seq -> (batch seq)")

            val_losses.append(cross_entropy(logits, targets_val).item())

        wandb.log({
            "val/loss": sum(val_losses) / len(val_losses),
            "step": t,
        })

        model.train()

    if t % everyN_savecheckpoint == 0:
        save_checkpoint(model, adamw_optimizer, t, f'save_step_{t}.pt')


save_checkpoint(model, adamw_optimizer, 5000, f'finalsave_step.pt')
print('training done ✅')
