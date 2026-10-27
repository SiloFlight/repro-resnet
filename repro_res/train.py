import json

import torch
from torch.nn import Module
from torch.utils.data import Dataset,DataLoader

from repro_res.dataset import get_dataloader
from repro_res.constants import WEIGHT_DECAY,RESULTS_DIR

@torch.no_grad()
def compute_statistics(model : Module, dataloader : DataLoader, num_classes : int = 10):
    device = next(model.parameters()).device

    correct = 0
    t5_correct = 0
    total = 0
    label_correct = torch.zeros(num_classes,dtype=torch.long)
    label_total = torch.zeros(num_classes,dtype=torch.long)

    for X,y in dataloader:
        X,y = X.to(device),y.to(device)
        logits = model(X)

        # (B,5) class indices, highest score first
        top5 = logits.topk(5,dim=1).indices
        hit = top5[:,0] == y

        correct += hit.sum().item()
        t5_correct += (top5 == y.unsqueeze(1)).any(dim=1).sum().item()
        total += y.numel()

        label_total += torch.bincount(y,minlength=num_classes).cpu()
        label_correct += torch.bincount(y[hit],minlength=num_classes).cpu()

    return {
        "accuracy" : correct / total,
        "t5_acc" : t5_correct / total,
        "per_label_accuracies" : (label_correct / label_total).tolist()
    }

def lr_schedule_default(iter : int) -> float:
    if iter <= 32e3:
        return .1
    elif iter <= 48e3:
        return .01
    else:
        return .001

def lr_schedule_110(iter : int) -> float:
    if iter <= 400:
        return 0.01
    else:
        return lr_schedule_default(iter)

def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def _write_json(path, data) -> None:
    with open(path,"w") as f:
        json.dump(data,f,indent=2)

def train_model(model : Module,
                model_name : str,
                loss_fn,
                total_iter : int,
                lr_fn,
                iter_per_eval : int = 1000,
                iter_per_save : int = 10000,
                resume : bool = True):
    """Resumes from results/{model_name}_checkpoint.pt if it exists (unless resume=False).
    Checkpoints are written at the first epoch end after every iter_per_save iterations, and at the end."""

    RESULTS_DIR.mkdir(parents=True,exist_ok=True)
    train_stats_path = RESULTS_DIR / f"{model_name}_train_stats.json"
    eval_stats_path = RESULTS_DIR / f"{model_name}_eval_stats.json"
    checkpoint_path = RESULTS_DIR / f"{model_name}_checkpoint.pt"

    device = get_device()
    model.to(device)
    model.train()

    # SGD w/ momentum .9; weight_decay adds WEIGHT_DECAY * w to each gradient before the momentum update
    optimizer = torch.optim.SGD(model.parameters(),lr=lr_fn(0),momentum=0.9,weight_decay=WEIGHT_DECAY)

    # Built once: each new pass over train_data reshuffles
    train_data = get_dataloader('train')
    test_data = get_dataloader('test')

    train_stats = []
    eval_stats = []

    def evaluate(i : int, lr : float) -> None:
        model.eval()
        test_stats = compute_statistics(model,test_data)
        model.train()
        eval_stats.append({"iteration" : i, "lr" : lr, **test_stats})
        _write_json(eval_stats_path,eval_stats)

    def save_checkpoint() -> None:
        # Write to a temp file then rename, so an interrupt mid-save can't corrupt the previous checkpoint
        tmp_path = checkpoint_path.with_suffix(".pt.tmp")
        torch.save({
            "iteration" : i,
            "epoch" : epoch,
            "model" : model.state_dict(),
            "optimizer" : optimizer.state_dict(),
            "rng_state" : torch.get_rng_state(),
            "train_stats" : train_stats,
            "eval_stats" : eval_stats,
        },tmp_path)
        tmp_path.replace(checkpoint_path)

    i = 0
    epoch = 0

    if resume and checkpoint_path.exists():
        # Loaded to CPU: the RNG state must stay on CPU, and load_state_dict moves the rest onto the model's device
        checkpoint = torch.load(checkpoint_path,map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        torch.set_rng_state(checkpoint["rng_state"])
        i = checkpoint["iteration"]
        epoch = checkpoint["epoch"]
        train_stats.extend(checkpoint["train_stats"])
        eval_stats.extend(checkpoint["eval_stats"])

        # Drop anything logged after the checkpoint; it will be recomputed
        _write_json(train_stats_path,train_stats)
        _write_json(eval_stats_path,eval_stats)
        print(f"Resuming {model_name} from iteration {i}")

        if i >= total_iter:
            return train_stats,eval_stats

    last_save = i
    lr = lr_fn(i)
    while i < total_iter:
        # Create stats for epoch
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_seen = 0

        for X,y in train_data:
            X,y = X.to(device),y.to(device)

            lr = lr_fn(i)
            for group in optimizer.param_groups:
                group["lr"] = lr

            pred = model(X)
            loss = loss_fn(pred,y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate stats on the augmented training batches (train mode), as in the paper's dashed curves
            epoch_loss += loss.item() * y.numel()
            epoch_correct += (pred.argmax(dim=1) == y).sum().item()
            epoch_seen += y.numel()

            i += 1
            if i % iter_per_eval == 0:
                evaluate(i,lr)

            if i >= total_iter:
                break

        # End of epoch (the last one may be partial, see "samples")
        epoch += 1
        train_stats.append({
            "epoch" : epoch,
            "iteration" : i,
            "lr" : lr,
            "samples" : epoch_seen,
            "loss" : epoch_loss / epoch_seen,
            "accuracy" : epoch_correct / epoch_seen,
        })
        _write_json(train_stats_path,train_stats)

        # Save on epoch boundaries so a resumed run starts a clean epoch
        if i - last_save >= iter_per_save:
            save_checkpoint()
            last_save = i

    # Make sure the final model is evaluated and saved even if total_iter isn't a multiple of the intervals
    if i % iter_per_eval != 0:
        evaluate(i,lr)
    save_checkpoint()

    return train_stats,eval_stats
