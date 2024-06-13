import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

### Utility functions, training

def _forward(network: nn.Module, data: DataLoader, metric: callable):
    device = next(network.parameters()).device

    for x in data:
        x, y = x.to(device), x.to(device)
        logits = network(x)
        res = metric(logits, y)
        yield res


@torch.no_grad()
def evaluate(network: nn.Module, data: DataLoader, metric: callable) -> list:
    network.eval()

    results = _forward(network, data, metric)
    return [res.item() for res in results]


@torch.enable_grad()
def update(network: nn.Module, data: DataLoader, loss: nn.Module,
           opt: optim.Optimizer) -> list:
    network.train()

    errs = []
    for err in _forward(network, data, loss):
        errs.append(err.item())

        opt.zero_grad()
        err.backward()
        opt.step()

    return errs


def train_auto_encoder(auto_encoder: nn.Module, loader: DataLoader,
                       objective: nn.Module, optimiser: optim.Optimizer,
                       num_epochs,adjust_lr, adjust_amount):  # adjust_lr, adjust_amount

    # evaluate random performance
    errs = evaluate(auto_encoder, loader, objective)
    print(f"Epoch {0: 2d} - avg loss: {sum(errs) / len(errs):.6f}")

    # naive learning rate adjustment
    adjust_lr_flag = len(adjust_lr)
    adjust_point = adjust_lr[-adjust_lr_flag] if adjust_lr_flag else 0

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #    optimiser, mode="min", factor=0.2, patience=150, cooldown=50, verbose=True)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.1, patience=100, cooldown=2, verbose=True)

    loss_acc = []  ### for plotting

    # train for some epochs
    for epoch in range(1, num_epochs + 1):
        errs = update(auto_encoder, loader, objective, optimiser)
        avgloss = sum(errs) / len(errs)
        print(f"Epoch {epoch: 2d} - avg loss: {avgloss:.6f}")
        loss_acc.append(avgloss)
        scheduler.step(np.mean(errs))

        # Adjust lr after loss initially goes down to escape plateaus
        if avgloss < adjust_point and adjust_lr_flag > 0:
            for g in optimiser.param_groups:
                g['lr'] *= adjust_amount
            adjust_lr_flag -= 1
            adjust_point = adjust_lr[-adjust_lr_flag]
            print("learning rate x{}".format(adjust_amount))

    return loss_acc