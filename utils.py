# Author: Taylor Seghesio & Garrett Sharp
# Organization: UNR CSE
# Course: CS 682
# date_Updated: 29NOV2024

# sources
# https://pytorch.org/vision/main/generated/torchvision.datasets.ImageFolder.html
# https://docs.python.org/3/library/zipfile.html
# https://pytorch.org/vision/0.20/transforms.html
# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
# https://pytorch.org/docs/stable/data.html

# Acknowledgments: Much of this code was written with the help of provided resources from UNR CS 682 course by Dr. Ankita
# Shukla, and the course textbook: Artificial Intelligence: A Modern Approach by Stuart Russel and Peter Norvig. My
# ability to finish the program was performed from the combined resources provided in the course and the sources listed
# above. Much of the implemented code in utils.py was provided from previous experience, resources and research
# conducted in UNR CS 687 course with Dr. Tavakkoli; specifically with zip file data extraction as previously used in
# my research graduate project in deep learning.

# imports for data extraction tasks
import os
import zipfile

# imports for vision/data handling tasks
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

train_loader = None
val_loader = None
test_loader = None


def extract_data(dataset_dir, extract_to):
    if not os.path.isdir(extract_to):
        os.makedirs(extract_to)

    for zip_file in os.listdir(dataset_dir):
        if zip_file.endswith(".zip"):
            zip_path = os.path.join(dataset_dir, zip_file)
            target_folder = os.path.join(extract_to, os.path.splitext(zip_file)[0])

            if not os.path.isdir(target_folder):
                print(f"Extracting {zip_file}...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(target_folder)
                print(f"Extraction complete for {zip_file}.")
            else:
                print(f"{zip_file} already extracted.")


def prepare_data(extracted_data, batch_size):
    test_val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = ImageFolder(extracted_data)
    train_size = int(0.7 * len(full_dataset))
    test_size = int(0.15 * len(full_dataset))
    val_size = len(full_dataset) - train_size - test_size
    train_dataset, test_dataset, val_dataset = random_split(full_dataset, [train_size, test_size, val_size])

    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = test_val_transform
    test_dataset.dataset.transform = test_val_transform

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


def get_loaders(dataset_path, extract_to, batch_size):
    extract_data(dataset_path, extract_to)
    global train_loader, val_loader, test_loader
    train_loader, val_loader, test_loader = prepare_data(extract_to, batch_size)
    return train_loader, val_loader, test_loader


def main():
    dataset_path = "dataset"
    extract_to = "extracted_dataset"
    batch_size = 32

    train_loader, val_loader, test_loader = get_loaders(dataset_path, extract_to, batch_size)

    print("Data preparation complete.")
    print(f"Train batches: {len(train_loader)}, Validation batches: {len(val_loader)}, Test batches: {len(test_loader)}")


if __name__ == '__main__':
    main()
