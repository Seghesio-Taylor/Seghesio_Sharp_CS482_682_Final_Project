import os
import zipfile
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split


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
        transforms.RandomRotation(10),
        transforms.ColorJitter(),
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


if __name__ == '__main__':

    dataset_path = "dataset"
    extract_to = "extracted_dataset"
    batch_size = 32

    extract_data(dataset_path, extract_to)
    train_loader, val_loader, test_loader = prepare_data(extract_to, batch_size)

    print("Data preparation complete.")
    print(f"Train batches: {len(train_loader)}, Validation batches: {len(val_loader)}, Test batches: {len(test_loader)}")
