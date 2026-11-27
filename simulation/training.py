"""Train the two nuisance models and predict on each counterfactual dose."""
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train_regression(model, train, validation, evaluation, device, config, checkpoint):
    optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=0.001)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=10, min_lr=1e-4)
    mae = nn.L1Loss()
    saved = False
    for epoch in range(1, config.epochs_dr + 1):
        model.train()
        train_loss = []
        for x, y in train:
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)
            loss = mae(model(x), y)
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            x, y = next(iter(validation))
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)
            loss = mae(model(x), y).item()
        print(f"Epoch {epoch} | train MAE={np.mean(train_loss):.6f} | validation MAE={loss:.6f}")
        if np.isfinite(loss):
            torch.save(model.state_dict(), checkpoint)
            saved = True
        scheduler.step(loss)
    if not saved:
        raise RuntimeError("The outcome regression produced no finite validation checkpoints.")
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    with torch.no_grad():
        curves = [model(next(iter(loader))[0].to(device=device, dtype=torch.float32)).cpu().numpy()
                  for loader in evaluation]
    torch.cuda.empty_cache()
    return curves


def train_density(model, train, validation, evaluation, device, config, checkpoint):
    optimizer = Adam(model.parameters(), lr=config.lr, weight_decay=0.001)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=10, min_lr=1e-4)
    best_training = np.inf
    stalled = 0
    saved = False
    for epoch in range(1, config.epochs_cnf + 1):
        model.train()
        train_loss = []
        for z in train:
            optimizer.zero_grad()
            z = z.to(device=device, dtype=torch.float32)
            nll, penalty = model(z[:, :-1], z[:, -1])
            (nll + penalty).backward()
            optimizer.step()
            train_loss.append(nll.item())
        training = np.mean(train_loss)
        if training < best_training:
            best_training = training
            stalled = 0
        else:
            stalled += 1
        if stalled > 20:
            print("Stopped after 20 epochs without improvement in training NLL.")
            break
        model.eval()
        with torch.no_grad():
            z = next(iter(validation)).to(device=device, dtype=torch.float32)
            loss, _ = model(z[:, :-1], z[:, -1])
            loss = loss.item()
        print(f"Epoch {epoch} | train NLL={training:.6f} | validation NLL={loss:.6f}")
        if np.isfinite(loss):
            torch.save(model.state_dict(), checkpoint)
            saved = True
        scheduler.step(loss)
    if not saved:
        raise RuntimeError("The density estimator produced no finite validation checkpoints.")
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    predictions = []
    with torch.no_grad():
        for loader in evaluation:
            z = next(iter(loader)).to(device=device, dtype=torch.float32)
            predictions.append(model(z[:, :-1], z[:, -1], test=True).cpu().numpy())
    torch.cuda.empty_cache()
    return predictions
