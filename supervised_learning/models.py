from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152


model_getter = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50
}