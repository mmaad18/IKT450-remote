import time

import torch
from torch import nn

from FishDataset import FishDataset
from FishNetwork import FishNeuralNetwork
from main_project_utils import get_transform, get_root_path
from utils import display_info_project, load_device, dataset_to_loaders_2
from utils import plot_loss


def train_loop(dataloader, model, loss_fn, optimizer, device="cpu"):
    model.train()

    for batch, (X, T) in enumerate(dataloader):
        Y = model(X)
        loss = loss_fn(Y, T)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad() # Reset gradients to prevent accumulation


def test_loop(dataloader, model, loss_fn, device="cpu"):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Prevents PyTorch from calculating and storing gradients
    with torch.no_grad():
        for X, T in dataloader:
            Y = model(X)
            pred_class = Y.argmax(dim=0)
            test_loss += loss_fn(Y, T).item()
            correct += (pred_class == T).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size

    return test_loss


def main():
    display_info_project()

    device = load_device()
    print(f"Using {device} device")

    model = FishNeuralNetwork().to(device)
    model = torch.compile(model)
    print(model)

    learning_rate = 0.01
    momentum = 0.9
    batch_size = 100
    epochs = 100

    fish_data = FishDataset(get_root_path(), "fish", get_transform(), device)
    train_loader, test_loader = dataset_to_loaders_2(fish_data, batch_size)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.NAdam(model.parameters(), lr=learning_rate)

    start = time.perf_counter()

    test_losses = []

    for t in range(epochs):
        train_loop(train_loader, model, loss_fn, optimizer, device)
        test_loss = test_loop(test_loader, model, loss_fn, device)
        test_losses.append(test_loss)

        if t % 1 == 0:
            print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e6} MB")
            print(f"Memory cached: {torch.cuda.memory_reserved() / 1e6} MB")

            print(f"Epoch {t}\n-------------------------------")
            print(f"Test Error: {test_loss}\n")

            end = time.perf_counter()
            print(f"Elapsed time: {end - start} seconds")

    print("Done!")
    plot_loss("MSE", test_losses, learning_rate, momentum, batch_size)


if __name__ == "__main__":
    main()


