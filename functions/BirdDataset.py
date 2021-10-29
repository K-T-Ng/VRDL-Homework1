import os

from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms, utils
from torchvision.io import read_image

from .HelperFunction import read_folds, read_test

class BirdDataset(Dataset):
    def __init__(self, root, imglist, lbllist, transform=None):
        self.root = root
        self.imglist = imglist
        self.lbllist = lbllist
        self.transform = transform

        if self.lbllist is None:
            self.mode = 'Testing'
        else:
            self.mode = 'Training'

    def __len__(self):
        return len(self.imglist)

    def __getitem__(self, index):
        # image part
        img_name = self.imglist[index]
        img = read_image(os.path.join(self.root, self.mode, img_name))
        if self.transform:
            img = self.transform(img)

        if self.mode == 'Testing':
            return img, img_name
        
        # label part
        lbl = self.lbllist[index]
        return img, lbl, img_name
    
def Get_Train_Valid_Dataset(FoldID, train_tform=None, valid_tform=None):
    train_fold = ['fold1.txt', 'fold2.txt', 'fold3.txt', 'fold4.txt', 'fold5.txt']
    valid_fold = ['fold' + str(FoldID) + '.txt']
    train_fold.remove(*valid_fold)

    train_imglist, train_lbllist = read_folds(filelist = train_fold)
    valid_imglist, valid_lbllist = read_folds(filelist = valid_fold)

    # extract label id and modify 1-200 to 0-199
    get_label = lambda label : int(label[:3])-1
    train_lbllist = tuple(map(get_label, train_lbllist))
    valid_lbllist = tuple(map(get_label, valid_lbllist))

    return BirdDataset('dataset', train_imglist, train_lbllist, train_tform),\
           BirdDataset('dataset', valid_imglist, valid_lbllist, valid_tform)

def Get_Test_Dataset(test_tform=None):
    imglist = read_test()
    lbllist = None
    return BirdDataset('dataset', imglist, lbllist, test_tform)

