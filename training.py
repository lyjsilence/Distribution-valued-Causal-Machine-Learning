import os
import time
import math
import torch
import numpy as np
import copy
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal

def train_regression(args, model, dl_train, dl_test, dl_est_test_list, device, verbose=True):
    best_loss = float('inf')

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=False, min_lr=1e-4)

    mae = nn.L1Loss()

    for epoch in range(1, args.epochs_DR + 1):
        model.train()
        tic = time.time()

        loss_train_list = []

        for i, train_data in enumerate(dl_train):
            X_train, y_train = train_data[0], train_data[1]
            X_train = X_train.to(torch.float32).to(device)
            y_train = y_train.to(torch.float32).to(device)

            pred_train = model(X_train)
            loss_train = mae(pred_train, y_train)

            loss_train_list.append(loss_train.cpu().detach().numpy())
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            if verbose:
                print('\rEpoch [{}/{}], Batch [{}/{}], Loss = {:.4f}, time elapsed = {:.2f}, '
                      .format(epoch, args.epochs_DR, i + 1, len(dl_train), np.sum(loss_train_list) / len(loss_train_list),
                              time.time() - tic), end='')

        with torch.no_grad():
            model.eval()

            for test_data in dl_test:
                X_test, y_test = test_data[0], test_data[1]
                X_test, y_test = X_test.to(torch.float32).to(device), y_test.to(torch.float32).to(device)
                pred_y_test = model(X_test)
                loss_test = mae(pred_y_test, y_test)

            pred_y_test = pred_y_test.detach().cpu().numpy()
            y_test = y_test.detach().cpu().numpy()
            if verbose:
                print('Test: MAE={:.4f}'.format(loss_test.cpu().detach().numpy()))

            if loss_test.item() < best_loss:
                model_save_path = os.path.join(args.results_save_path, 'Simulation', 'Size='+str(args.sample_size), 'DR_sim.pkl')
                # model_save_path = os.path.join(args.results_save_path, 'DR_sim.pkl')
                torch.save(model.state_dict(), model_save_path)


        scheduler.step(loss_test)

    with torch.no_grad():
        model.eval()
        model.load_state_dict(torch.load(model_save_path))
        pred_y_test_list = []

        for dl_est_test in dl_est_test_list:
            for test_data in dl_est_test:

                X_test, y_test = test_data[0], test_data[1]
                X_test = X_test.to(torch.float32).to(device)
                y_test = y_test.to(torch.float32).to(device)

                pred_y_test = model(X_test)
                pred_y_test_list.append(pred_y_test.detach().cpu().numpy())
        torch.cuda.empty_cache()
    return pred_y_test_list

def train_IPW(args, model, dl_train, dl_test, dl_est_test_list, device, verbose=True):
    best_loss = float('inf')
    best_training_loss = float('inf')
    train_epoch = 0

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True, min_lr=1e-4)

    for epoch in range(1, args.epochs_IPW + 1):
        model.train()
        tic = time.time()
        loss_train_list = []

        '''Training'''
        for i, train_data in enumerate(dl_train):
            optimizer.zero_grad()
            X_train = (train_data[:, :-1]).to(torch.float32).to(device)
            D_train = (train_data[:, -1]).to(torch.float32).to(device)
            # predict the probability
            loss_nll, loss_reg = model(X_train, D_train)
            loss = loss_nll + loss_reg

            loss.backward()
            optimizer.step()

            loss_train_list.append(loss_nll.item())

            if verbose:
                print('\rEpoch [{}/{}], Batch [{}/{}], Loss = {:.4f}, time elapsed = {:.2f}, '
                      .format(epoch, args.epochs_IPW, i + 1, len(dl_train), np.mean(loss_train_list), time.time() - tic), end='')

        # stop training if training loss does not decrease for 20 epochs
        if np.mean(loss_train_list) < best_training_loss:
            train_epoch = 0
            best_training_loss = np.mean(loss_train_list)
        else:
            train_epoch += 1

        if train_epoch > 20:
            print('Breaking for early stopping...')
            break

        '''Validation'''
        with torch.no_grad():
            model.eval()
            loss_test_list = []
            for i, test_data in enumerate(dl_test):
                X_test = (test_data[:, 0:-1]).to(torch.float32).to(device)
                D_test = (test_data[:, -1]).to(torch.float32).to(device)
                # predict the probability
                loss_nll_test, _ = model(X_test, D_test)
                loss_test_list.append(loss_nll_test.item())

            if verbose:
                print('Test: NLL={:.4f}'.format(np.mean(loss_test_list)))

            if np.mean(loss_test_list) < best_loss:
                model_save_path = os.path.join(args.results_save_path, 'Simulation', 'Size='+str(args.sample_size), 'IPW_sim.pkl')
                # model_save_path = os.path.join(args.results_save_path, 'IPW_sim.pkl')
                torch.save(model.state_dict(), model_save_path)

        scheduler.step(np.mean(loss_test_list))


    '''Sampling from the estimated density'''
    with torch.no_grad():
        model.eval()
        model.load_state_dict(torch.load(model_save_path))

        pred_px_test_list = []

        for dl_est_test in dl_est_test_list:
            for test_data in dl_est_test:
                X_est_test = test_data[:, :-1].to(torch.float32).to(device)
                D_est_test = test_data[:, -1].to(torch.float32).to(device)
                px = model(X_est_test, D_est_test, test=True)
                pred_px_test_list.append(px.detach().cpu().numpy())
        torch.cuda.empty_cache()

    return pred_px_test_list

