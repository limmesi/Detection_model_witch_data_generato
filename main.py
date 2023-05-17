from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision

from dataloader import MyDataloader
from model import SimpleNet
from utils import *

if __name__ == '__main__':
    # pills, imgs = load_files()
    # make_dataset(pills, imgs)

    device = torch.device("cpu")

    learning_rate = 0.001
    batch_size = 32
    num_epochs = 10

    train_dataset = MyDataloader()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Create the model and move it to the device
    model = SimpleNet().to(device)

    # Define the loss function and optimizer
    criterion = torchvision.ops.generalized_box_iou_loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        train(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch + 1}/{num_epochs} completed.")

    # test(model, train_loader, device)








