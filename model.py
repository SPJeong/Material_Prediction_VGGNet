##### model.py

import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
from torch.nn import AdaptiveAvgPool2d


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class VGG11(nn.Module):
    def __init__(self, descriptor_size=193):
        super(VGG11, self).__init__()

        # Load the pre-trained ResNet-18 model
        self.vgg = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1)
        self.vgg.avgpool = Identity()
        self.vgg.avgpool = AdaptiveAvgPool2d(output_size=(1, 1))

        # Freeze the feature layers
        for param in self.vgg.parameters():
            param.requires_grad = False

        # Remove the final fully connected layer
        self.vgg = nn.Sequential(*list(self.vgg.children())[:-1])

        # A separate fc_layers that takes the combined features
        self.fc_layers = nn.Sequential(
            nn.Linear(512 + descriptor_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x, descriptors):
        # The output 'out' will have dimensions [batch_size, num_ftrs, 1, 1]
        vgg_features = self.vgg(x)
        vgg_features = vgg_features.view(vgg_features.size(0), -1)
        combined_features = torch.cat((vgg_features, descriptors), dim=1)
        outputs = self.fc_layers(combined_features)

        return outputs


# # Check the corrected model
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = VGG11()
# model.to(device)
# print(model)
# x = torch.randn(2, 3, 224, 224).to(device)
# d = torch.randn(2, 193).to(device)
# print("Output shape:", model(x, descriptors=d).shape)

def count_parameters(model):
    """
    Calculates the total number of trainable parameters in the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Calculate and print the total parameters
if __name__ == '__main__':
    my_deep_model = VGG11()
    total_params = count_parameters(my_deep_model)
    print(f"Total trainable parameters: {total_params:,}")