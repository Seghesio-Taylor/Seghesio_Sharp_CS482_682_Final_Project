# Author: Taylor Seghesio & Garrett Sharp
# Organization: UNR CSE
# Course: CS 682
# date_Updated: 30NOV2024

# Acknowledgements: (see final project report deliverable for documented citations)
# [1]  https://pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html ### resnet pre built model
# [2]  https://www.programiz.com/python-programming/methods/built-in/iter  ###used for iter
# [3]  https://pytorch.org/tutorials/beginner/basics/data_tutorial.html  ###data_loader/pytorch
# [4]  https://www.geeksforgeeks.org/matplotlib-pyplot-imshow-in-python/ ###imshow in python
# [5]  https://pytorch.org/vision/stable/transforms.html ### image augmentation
# [6]  https://stackoverflow.com/questions/28269157/plotting-in-a-non-blocking-way-with-matplotlib ### using plt.pause instead of block=False
# [7]  https://medium.com/@harshit4084/track-your-loop-using-tqdm-7-ways-progress-bars-in-python-make-things-easier-fcbbb9233f24 ### prog bar
# [8]  https://blog.paperspace.com/how-to-maximize-gpu-utilization-by-finding-the-right-batch-size/ ### helped with accuracy and algorithm performance
# [9]  https://www.linkedin.com/advice/3/how-can-you-improve-neural-network-performance-xkrxe#:~:text=Selecting%20the%20number%20of%20epochs,it%20based%20on%20validation%20performance. ###help with accuracy and curve performance


# imports for vision/data handling tasks
import utils
from utils import transforms, ImageFolder, DataLoader, random_split, os, get_loaders

import torchvision.models as models
from torchvision.models import ResNet18_Weights
from torchvision.utils import make_grid

# imports for neural network
import torch
from torch import nn, optim
from torchinfo import summary
from torch.optim.lr_scheduler import StepLR

# imports for visualizations
import matplotlib.pyplot as plt
from tqdm.auto import tqdm






# GLOBALS
BATCH_SIZE = 32
NUM_EPOCHS = 30

# Used for debugging CUDA execution and confirming correct initialization with correct CUDA device
print(torch.cuda.is_available())
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(torch.cuda.current_device())
#print(torch.cuda.get_device_name(0))


def visualize_dataset(train_loader, val_loader, test_loader):
    def get_data(data_loader):
        data_iter = iter(data_loader)
        images, labels = next(data_iter)
        filenames = []
        random_data_sample = data_loader.dataset.indices[:10]
        for index in random_data_sample:
            full_path = data_loader.dataset.dataset.samples[index][0]
            filename = full_path.split(os.sep)[-1]
            filenames.append(filename)

        total_glioma = 0
        total_meningioma = 0
        total_pituitary = 0
        total_normal = 0

        subset_indices = data_loader.dataset.indices
        for idx in subset_indices:
            path, label = data_loader.dataset.dataset.samples[idx]
            if label == data_loader.dataset.dataset.class_to_idx['glioma_tumor']:
                total_glioma += 1
            elif label == data_loader.dataset.dataset.class_to_idx['meningioma_tumor']:
                total_meningioma += 1
            elif label == data_loader.dataset.dataset.class_to_idx['pituitary_tumor']:
                total_pituitary += 1
            elif label == data_loader.dataset.dataset.class_to_idx['no_tumor']:
                total_normal += 1

        return filenames, total_glioma, total_meningioma, total_pituitary, total_normal

    train_files, train_glioma, train_meningioma, train_pituitary, train_normal = get_data(train_loader)
    val_files, val_glioma, val_meningioma, val_pituitary, val_normal = get_data(val_loader)
    test_files, test_glioma, test_meningioma, test_pituitary, test_normal = get_data(test_loader)

    print('Train files:', train_files)
    print('Validation files:', val_files)
    print('Test files:', test_files)

    print('\nTotal training images:', len(train_loader.dataset))
    print('Total training glioma tumor images:', train_glioma)
    print('Total training meningioma tumor images:', train_meningioma)
    print('Total training pituitary tumor images:', train_pituitary)
    print('Total training normal (non-tumor) images:', train_normal)

    print('\nTotal validation images:', len(val_loader.dataset))
    print('Total training glioma tumor images:', val_glioma)
    print('Total training meningioma tumor images:', val_meningioma)
    print('Total training pituitary tumor images:', val_pituitary)
    print('Total validation normal images:', val_normal)

    print('\nTotal test images:', len(test_loader.dataset))
    print('Total training glioma tumor images:', test_glioma)
    print('Total training meningioma tumor images:', test_meningioma)
    print('Total training pituitary tumor images:', test_pituitary)
    print('Total test normal images:', test_normal)


def show_batch(data_dir):
    for images, labels in data_dir:
        fig, ax = plt.subplots(figsize=(12, 9))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=8).permute(1, 2, 0))
        break

    plt.pause(0.001)
    input("Press [enter] to continue.")


def load_architecture(model, device):
    model.to(device)
    summary(model, (3, 256, 256))
    return model


def train_model(model, train_dir, val_dir, num_epochs, device):
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.001  # used when not using scheduler

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-6)

    scheduler = StepLR(optimizer, step_size=2, gamma=0.1) # only used for some experiments

    early_stopping_patience = 10
    best_val_loss = float('inf')
    best_epoch = 0
    early_stopping_counter = 0

    epoch_train_loss_values = []
    epoch_val_loss_values = []
    epoch_train_acc_values = []
    epoch_val_acc_values = []

    for epoch in range(num_epochs):
        print(f'\nRunning epoch {epoch + 1}/{num_epochs}')
        model.train()
        train_losses, train_accuracies = [], []

        with tqdm(total=len(train_dir), desc=f'Epoch {epoch + 1}', unit='batch', leave=False) as pbar:
            for images, labels in train_dir:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                acc = (outputs.argmax(dim=1) == labels).float().mean().item()
                train_losses.append(loss.item())
                train_accuracies.append(acc)

                pbar.update(1)

        scheduler.step()  # only used if step scheduler is enabled

        epoch_train_loss = sum(train_losses) / len(train_losses)
        epoch_train_acc = sum(train_accuracies) / len(train_accuracies)
        epoch_train_loss_values.append(epoch_train_loss)
        epoch_train_acc_values.append(epoch_train_acc)

        model.eval()
        val_losses, val_accuracies = [], []
        with torch.no_grad():
            for images, labels in val_dir:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss = criterion(outputs, labels)
                val_losses.append(val_loss.item())

                acc = (outputs.argmax(dim=1) == labels).float().mean().item()
                val_accuracies.append(acc)

        epoch_val_loss = sum(val_losses) / len(val_losses)
        epoch_val_acc = sum(val_accuracies) / len(val_accuracies)
        epoch_val_loss_values.append(epoch_val_loss)
        epoch_val_acc_values.append(epoch_val_acc)

        print(f'Epoch: {epoch + 1}\n'
              f'Train Acc: {epoch_train_acc:.3f}, Val Acc: {epoch_val_acc:.3f} '
              f'Train Loss: {epoch_train_loss:.3f}, Val Loss: {epoch_val_loss:.3f}')

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), 'best_model.pth')
            early_stopping_counter = 0
            print(f'Best Metric Updated: {best_val_loss:.3f} at epoch {best_epoch}')
        else:
            early_stopping_counter += 1
            print(f'Best Metric: {best_val_loss:.3f} at epoch: {best_epoch}\n')

        if early_stopping_counter >= early_stopping_patience:
            print(f"Early stopping after {early_stopping_patience} epochs of no improvement.")
            break

    print(f'Finished Training. Best Validation Loss: {best_val_loss:.3f} achieved at Epoch {best_epoch}')
    return epoch_train_loss_values, epoch_val_loss_values, epoch_train_acc_values, epoch_val_acc_values


def plot_results(epoch_train_loss_values, epoch_val_loss_values, epoch_train_acc_values, epoch_val_acc_values):
    # Plot results
    plt.figure(figsize=(12, 6))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.title("Loss")
    x_train = [i + 1 for i in range(len(epoch_train_loss_values))]
    y_train = epoch_train_loss_values
    x_val = [i + 1 for i in range(len(epoch_val_loss_values))]
    y_val = epoch_val_loss_values
    plt.plot(x_train, y_train, label='Train Loss')
    plt.plot(x_val, y_val, label='Val Loss', linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    x_train_acc = [i + 1 for i in range(len(epoch_train_acc_values))]
    y_train_acc = epoch_train_acc_values
    x_val_acc = [i + 1 for i in range(len(epoch_val_acc_values))]
    y_val_acc = epoch_val_acc_values
    plt.plot(x_train_acc, y_train_acc, label='Train Acc')
    plt.plot(x_val_acc, y_val_acc, label='Val Acc', linestyle='--')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.pause(0.001)
    input("Press [enter] to continue.")


def test_model(model, test_dir, device, criterion):
    print("Testing model now...")
    model.eval()
    total_accuracy, total_test_loss = 0.0, 0.0
    num_batches = len(test_dir)

    with torch.no_grad():
        with tqdm(total=len(test_dir), desc='Testing Model: ', unit='batch', leave=False) as pbar:
            for data, labels in test_dir:
                data, labels = data.to(device), labels.to(device)

                outputs = model(data)
                loss = criterion(outputs, labels)
                accuracy = (outputs.argmax(dim=1) == labels).float().mean()

                total_test_loss += loss.item()
                total_accuracy += accuracy.item()

                pbar.update(1)

    avg_loss = total_test_loss / num_batches
    avg_accuracy = total_accuracy / num_batches

    print(f'Test Accuracy: {avg_accuracy:.3f}, Test Loss: {avg_loss:.3f}')


def main():
    # Handles our Dataset
    train_loader, val_loader, test_loader = utils.get_loaders("dataset", "extracted_dataset", BATCH_SIZE)

    # Visualizes our Dataset - For debugging/confirmation
    visualize_dataset(utils.train_loader, utils.val_loader, utils.test_loader)
    show_batch(utils.train_loader)

    # Instantiates our ResNet18 Model and performs model tasks
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 4)
    model = load_architecture(model, DEVICE)
    results = train_model(model, utils.train_loader, utils.val_loader, NUM_EPOCHS, DEVICE)
    (epoch_train_loss_values, epoch_val_loss_values, epoch_train_acc_values, epoch_val_acc_values) = results
    plot_results(epoch_train_loss_values, epoch_val_loss_values, epoch_train_acc_values, epoch_val_acc_values)

    # Testing the model
    model.load_state_dict(torch.load('best_model.pth'))
    loss = nn.CrossEntropyLoss()
    test_model(model, utils.test_loader, DEVICE, loss)

    input("Press Enter to close program and close plotted data/images...")


if __name__ == '__main__':
    main()
