import data, models

import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import argparse
import wandb
import logging
import sys


parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--data-dir", type=str, default="data")
parser.add_argument("--dataset", type=str, default="cifar10")
parser.add_argument("--model", type=str, default="resnet50")
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--train-steps", type=int, default=1000)
parser.add_argument("--test-steps", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--workers", type=int, default=4)
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--nolog", action="store_true")
args = parser.parse_args()


logging.basicConfig(
    format="%(asctime)s.%(msecs)03d %(levelname)s %(message)s", 
    datefmt='%Y-%m-%d,%H:%M:%S', 
    level=logging.INFO
)
logger = logging.getLogger()
if args.nolog:
    logging.disable(sys.maxsize)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_seeds(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def accuracy(output, target):
    class_pred = torch.argmax(output, dim = 1)
    correct = class_pred.eq(target).sum().item()
    accuracy = correct / target.size(0)
    return accuracy


def train_loop(args, train_loader, test_loader, model, criterion, optimizer, scheduler):
    wandb.init(project = "SimpleCVSupervised", name = args.name, config = args)
    logger.info(f"Training {args.model} on {args.dataset} @{args.train_steps}")
    logger.info(args)
    pbar = tqdm(total = args.test_steps)
    train_iter = iter(train_loader)
    train_accuracy = AverageMeter()
    train_loss = AverageMeter()
    top1_accuracy = 0.0
    for step in range(args.train_steps):
        wandb.log({"lr": optimizer.param_groups[0]["lr"]})
        try:
            data, target = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            data, target = next(train_iter)
        data, target = data.to(args.device), target.to(args.device)
        model.train()
        output = model(data)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss.update(loss.item())
        train_accuracy.update(accuracy(output, target))
        pbar.set_description(
            f"{(step + 1):4d}/{args.train_steps}: train/loss: {train_loss.avg:.4E}, train/acc: {train_accuracy.avg:.4f}"
        )
        pbar.update()

        if (step + 1) % args.test_steps == 0:
            pbar.close()
            test_loss, test_accuracy = evaluate(args, test_loader, model, criterion, step)
            wandb.log({
                "train/loss": train_loss.avg,
                "train/accuracy": train_accuracy.avg,
                "test/loss": test_loss,
                "test/accuracy": test_accuracy
            }, step = step + 1)
            if test_accuracy > top1_accuracy:
                top1_accuracy = test_accuracy
                wandb.save("model.pth")
                logger.info(f"Model saved with top1/acc: {top1_accuracy:.4f}")
            pbar = tqdm(total = args.test_steps)
    print("TRAINING FINISHED")


def evaluate(args, test_loader, model, criterion, step = 0):
    model.eval()
    test_loss = AverageMeter()
    test_accuracy = AverageMeter()
    pbar = tqdm(total = len(test_loader))
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(args.device), target.to(args.device)
            output = model(data)
            loss = criterion(output, target)
            test_loss.update(loss.item())
            test_accuracy.update(accuracy(output, target))
            pbar.set_description(
                f"{(step + 1):4d}/{args.train_steps}:  test/loss: {test_loss.avg:.4E},  test/acc: {test_accuracy.avg:.4f}"
            )
            pbar.update()
    pbar.close()
    return test_loss.avg, test_accuracy.avg


if __name__ == "__main__":
    set_seeds(args)
    args.device = torch.device(args.device)
    train_dataset, test_dataset = data.dataset_getter[args.dataset](args)
    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = args.batch_size,
        shuffle = True,
        num_workers = args.workers
    )
    test_loader = DataLoader(
        dataset = test_dataset,
        batch_size = args.batch_size,
        shuffle = False,
        num_workers = args.workers
    )
    model = models.model_getter[args.model]()
    model = model.to(args.device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params = model.parameters(), lr = args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max = args.train_steps, eta_min = 0.0001)
    train_loop(args, train_loader, test_loader, model, criterion, optimizer, scheduler)