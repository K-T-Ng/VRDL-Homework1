import time
import torch


def Training_step(dataloader, device, model, optimizer, loss_fn):
    total_loss, num_correct = 0, 0
    for batch in dataloader:
        image, label, _ = batch
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        predict, feature = model(image)
        loss = loss_fn(feature, predict, label)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * image.size(0)
        num_correct += torch.sum(predict.argmax(dim=1) == label).item()

        del image, batch
    return total_loss, num_correct


def Validation_step(dataloader, device, model, loss_fn):
    total_loss, num_correct = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            image, label, _ = batch
            image = image.to(device)
            label = label.to(device)

            predict, feature = model(image)
            loss = loss_fn(feature, predict, label)

            total_loss += loss.item() * image.size(0)
            num_correct += torch.sum(predict.argmax(dim=1) == label).item()

            del image, batch
    return total_loss, num_correct


def fit_an_epoch(device, model, optimizer, scheduler, loss_fn,
                 TrLoader, num_train, ValLoader, num_valid):
    start = time.time()

    model.train()
    TrLoss, TrAcc = Training_step(TrLoader, device, model, optimizer, loss_fn)

    model.eval()
    ValLoss, ValAcc = Validation_step(ValLoader, device, model, loss_fn)

    scheduler.step()

    print(f'Train loss: {TrLoss/num_train:.4f}, acc: {TrAcc/num_train:.4f}')
    print(f'Valid loss: {ValLoss/num_valid:.4f}, acc: {ValAcc/num_valid:.4f}')
    print(f'Elapsed time: {time.time()-start:.4f}')
    return ValAcc/num_valid
