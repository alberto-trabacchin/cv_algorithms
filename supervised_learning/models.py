from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
import torch


def get_resnet50(args):
    model = resnet50()
    model.conv1 = torch.nn.Conv2d(
        in_channels=3, 
        out_channels=64, 
        kernel_size=3, 
        stride=1, 
        padding=1, 
        bias=False
    )
    num_features = model.fc.in_features
    if args.dataset == "cifar10":
        model.fc = torch.nn.Linear(num_features, 10)
    return model


model_getter = {
    "resnet50": get_resnet50
}