import os
import glob

import torch
from torch.utils.data import DataLoader

from functions.HelperFunction import read_classes_mapping, read_models
from functions.Transforms import get_transforms
from functions.BirdDataset import Get_Test_Dataset
from functions.Network import BirdClassifier

if __name__ == '__main__':
    # some parameters
    batch_size = 100
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # prepare testing dataloader
    _, _, TestTform = get_transforms()
    TestDs = Get_Test_Dataset(TestTform)
    TestLoader = DataLoader(TestDs, batch_size=batch_size, shuffle=False)

    # prepare network
    with torch.no_grad():
        model_list = read_models(root='saved_model', model_cls=BirdClassifier)
        for model in model_list:
            model = model.to(device)
            model.eval()

    # generate testing answer.txt
    class_map = read_classes_mapping(root='dataset', filename='classes.txt')
    fp = open('answer.txt', 'w')
    with torch.no_grad():
        for imgs, names in TestLoader:
            imgs = imgs.to(device)

            # take the average score on the prediction
            preds = sum([model.predict(imgs) for model in model_list])
            preds = preds.argmax(dim=1).detach().tolist()

            # write result into answer.txt
            for pred, name in zip(preds, names):
                lbl_id = str(pred+1).zfill(3)  # recover 0-199 to 1-200
                fp.write(name+' '+lbl_id+'.'+class_map[lbl_id]+'\n')
    fp.close()
