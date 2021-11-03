import torch
from torch import nn
from torchvision import models


class BirdClassifier(nn.Module):
    def __init__(self, num_class=200):
        super(BirdClassifier, self).__init__()
        # we use resnet 50 as our pretrain model
        ResNet = models.resnet50(pretrained=True)
        ResNet.fc = nn.Linear(ResNet.fc.in_features, num_class)
        ResNet_child = list(ResNet.children())

        # split 2 blocks (before conv1-layer4, avgpool-fc)
        self.FeatureExtractor = nn.Sequential(*ResNet_child[:-2])
        self.avgpool = ResNet_child[-2]
        self.fc = ResNet_child[-1]

    def forward(self, image):
        # extract features for MC_loss stream
        feat = self.FeatureExtractor(image)

        # MLP output for CE_loss stream
        x = self.avgpool(feat)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x, feat

    def save(self, loc):
        torch.save(self.state_dict(), loc)
        return

    def load(self, loc):
        self.load_state_dict(torch.load(loc))
        return

    def predict(self, image):
        return self.forward(image)[0]


if __name__ == '__main__':
    model = BirdClassifier()
    x = torch.zeros(1, 3, 240, 240)
    f, z = model(x)
    print(f.shape, z.shape)
