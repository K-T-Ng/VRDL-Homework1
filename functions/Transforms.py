import random
import torch
from torchvision import transforms


class MinMaxScaling(object):
    def __call__(self, img):
        '''
            img is a tensor of shape (C, H, W), type = uint8
        '''
        return img.float()/255


class RandomGaussianNoise(object):
    def __init__(self, sig=0.02, p=0.5):
        self.sig = sig
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            img += self.sig * torch.randn(img.shape)
        return img


class GridMask(object):
    def __init__(self, shape=(224, 224), dmin=90, dmax=160, ratio=0.6, p=0.5):
        self.shape = shape
        self.dmin = dmin
        self.dmax = dmax
        self.ratio = ratio
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        d = random.randint(self.dmin, self.dmax)
        dx, dy = random.randint(0, d-1), random.randint(0, d-1)
        sl = int(d * (1-self.ratio))
        for i in range(dx, self.shape[0], d):
            for j in range(dy, self.shape[1], d):
                row_end = min(i+sl, self.shape[0])
                col_end = min(j+sl, self.shape[1])
                img[:, i:row_end, j:col_end] = 0
        return img


def get_transforms():
    TrainTform = transforms.Compose([
                    MinMaxScaling(),
                    transforms.Resize((500, 500)),
                    transforms.RandomCrop((400, 400)),
                    transforms.RandomHorizontalFlip(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225]),
                    GridMask(shape=(400, 400), dmin=90, dmax=200,
                             ratio=0.7, p=0.25),
                    ])

    ValidTform = transforms.Compose([
                    MinMaxScaling(),
                    transforms.Resize((500, 500)),
                    transforms.CenterCrop((400, 400)),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225]),
                ])

    TestTform = transforms.Compose([
                    MinMaxScaling(),
                    transforms.Resize((500, 500)),
                    transforms.CenterCrop((400, 400)),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225]),
                ])
    return TrainTform, ValidTform, TestTform
