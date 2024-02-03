from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import transforms

def get_cifar10(args):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = CIFAR10(
        root = args.data_dir,
        train = True,
        download = True,
        transform = transform
    )
    test_dataset = CIFAR10(
        root = args.data_dir,
        train = False,
        download = True,
        transform = transform
    )
    return train_dataset, test_dataset



dataset_getter = {
    "cifar10": get_cifar10
}