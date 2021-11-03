import os

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from functions.HelperFunction import remove_with_prefix
from functions.Transforms import get_transforms
from functions.BirdDataset import Get_Train_Valid_Dataset
from functions.Network import BirdClassifier
from functions.TrainFunction import fit_an_epoch
from functions.Loss import MCCE_Loss


if __name__ == '__main__':
    # parameters setting
    num_class = 200
    num_workers = 2
    batch_size = 24
    epoch = 20
    extractor_lr = 1e-4
    fclayer_lr = 5e-4
    decay_gamma = 0.95
    weight_decay = 5e-4
    loss_lambda = 10
    loss_mu = 0.01
    FoldID = 5
    device = torch.device("cuda:0" if torch.cuda.is_available() else "gpu")

    # prepare training and validation dataloader
    TrainTform, ValidTform, _ = get_transforms()
    TrainDs, ValidDs = Get_Train_Valid_Dataset(FoldID, TrainTform, ValidTform)

    TrainLoader = DataLoader(dataset=TrainDs, batch_size=batch_size,
                             shuffle=True, num_workers=num_workers,
                             pin_memory=True)
    ValidLoader = DataLoader(dataset=ValidDs, batch_size=batch_size,
                             shuffle=True, num_workers=num_workers,
                             pin_memory=True)

    # model, optimizer, lr_scheduler and loss function
    model = BirdClassifier(num_class=num_class)

    optimizer = optim.AdamW(
        [{'params': model.FeatureExtractor.parameters(), 'lr': extractor_lr},
         {'params': model.fc.parameters(), 'lr': 5e-4}],
        weight_decay=weight_decay)

    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer,
                                                    gamma=decay_gamma)

    loss_fn = MCCE_Loss(lambda_=loss_lambda, mu_=loss_mu)

    # train (and get overfit(X))
    model = model.to(device)
    num_train, num_valid = len(TrainDs), len(ValidDs)

    remove_with_prefix('saved_model', f'fold={FoldID}')
    best_val_acc = -1

    for ep in range(1, epoch+1):
        print(f'Epoch{ep}:')

        val_acc = fit_an_epoch(device, model, optimizer, lr_scheduler, loss_fn,
                               TrainLoader, num_train,
                               ValidLoader, num_valid)

        if val_acc > best_val_acc:
            remove_with_prefix('saved_model', f'fold={FoldID}')
            best_val_acc = val_acc
            file_name = f'fold={FoldID}_ep={ep}_acc={best_val_acc:.4f}.pth'
            model.save(os.path.join('saved_model', file_name))
