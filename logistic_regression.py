# Authors: Taylor Seghesio & Garrett Sharp
# Organization: UNR CSE
# Course: CS 482/ CS 682
# date_Updated: 02DEC2024

# Acknowledgements:
# pandas.pydata.org/docs/user_guide/index.html#user-guide
# matplotlib.org/stable/tutorials/pyplot.html
# scikit-learn.org/dev/modules/generated/sklearn.metrics.accuracy_score.html
# pytorch.org/tutorials/beginner/basics/data_tutorial.html
# www.w3schools.com/python/numpy/default.asp
# builtin.com/machine-learning/pca-in-python
# scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LogisticRegression.html
# machinelearningmastery.com/building-a-logistic-regression-classifier-in-pytorch/

# Acknowledgments: Much of this code was written with the help of provided resources from UNR CS 682 course by Dr. Ankita
# Shukla, and the course textbook: Artificial Intelligence: A Modern Approach by Stuart Russel and Peter Norvig. My
# ability to finish the program was performed from the combined resources provided in the course and the sources listed
# above.

import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np
from torchvision.utils import make_grid  # Import make_grid
from sklearn.decomposition import PCA

import utils
from utils import transforms, ImageFolder, DataLoader, random_split, os, get_loaders

# GLOBALS
BATCH_SIZE = 128

def plot_confusion_matrix(cm, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def plot_classification_report(cr, title='Classification Report', cmap=plt.cm.Blues):
    lines = cr.split('\n')
    classes = []
    plot_data = []
    for line in lines[2: (len(lines) - 3)]:
        row_data = line.split()
        if len(row_data) < 2 or row_data[0] in ['accuracy', 'macro',
                                                'weighted']:  # Skip lines that do not contain the expected data
            continue
        classes.append(row_data[0])
        plot_data.append([float(x) for x in row_data[1: len(row_data) - 1]])

    plt.imshow(plot_data, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    x_tick_marks = np.arange(3)
    y_tick_marks = np.arange(len(classes))
    plt.xticks(x_tick_marks, ['Precision', 'Recall', 'F1-score'], rotation=45)
    plt.yticks(y_tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('Classes')
    plt.xlabel('Metrics')
    plt.show()


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
    print('Total validation glioma tumor images:', val_glioma)
    print('Total validation meningioma tumor images:', val_meningioma)
    print('Total validation pituitary tumor images:', val_pituitary)
    print('Total validation normal images:', val_normal)

    print('\nTotal test images:', len(test_loader.dataset))
    print('Total test glioma tumor images:', test_glioma)
    print('Total test meningioma tumor images:', test_meningioma)
    print('Total test pituitary tumor images:', test_pituitary)
    print('Total test normal images:', test_normal)


def show_batch(data_loader):
    for images, labels in data_loader:
        fig, ax = plt.subplots(figsize=(12, 9))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=8).permute(1, 2, 0))
        break

    plt.pause(0.001)
    input("Press [enter] to continue.")


def train_model(train_loader, val_loader, n_components=50):
    """
    Train a logistic regression model with PCA for dimensionality reduction.
    """
    print("Preparing training data...")
    # Prepare the data
    X_train, y_train = [], []
    for images, labels in train_loader:
        images = images.view(images.size(0), -1).numpy()  # Flatten the images
        labels = labels.numpy()
        X_train.extend(images)
        y_train.extend(labels)

    X_val, y_val = [], []
    for images, labels in val_loader:
        images = images.view(images.size(0), -1).numpy()  # Flatten the images
        labels = labels.numpy()
        X_val.extend(images)
        y_val.extend(labels)

    # Convert to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_val = np.array(X_val)
    y_val = np.array(y_val)

    print("Standardizing the data...")
    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    print("Applying PCA for dimensionality reduction...")
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)

    print(f"PCA reduced dimensions to: {X_train_pca.shape[1]}")

    print("Training the logistic regression model...")
    # Train the logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_pca, y_train)

    y_train_pred = model.predict(X_train_pca)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    print(f'Training Accuracy: {train_accuracy:.3f}')

    cr_train = classification_report(y_train, y_train_pred)
    cm_train = confusion_matrix(y_train, y_train_pred)
    print("Training Classification Report:\n", cr_train)
    print("Training Confusion Matrix:\n", cm_train)

    plot_classification_report(cr_train, title='Training Classification Report')
    plot_confusion_matrix(cm_train, classes=['glioma', 'meningioma', 'pituitary', 'normal'])

    print("Validating the model...")
    # Validate the model
    y_val_pred = model.predict(X_val_pca)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    print(f'Validation Accuracy: {val_accuracy:.3f}')

    cr_val = classification_report(y_val, y_val_pred)
    cm_val = confusion_matrix(y_val, y_val_pred)
    print("Validation Classification Report:\n", cr_val)
    print("Validation Confusion Matrix:\n", cm_val)

    plot_classification_report(cr_val)
    plot_confusion_matrix(cm_val, classes=['glioma', 'meningioma', 'pituitary', 'normal'])

    return model, scaler, pca


def test_model(model, scaler, pca, test_loader):
    """
    Test the trained logistic regression model with PCA on test data.
    """
    print("Preparing test data...")
    # Prepare the test data
    X_test, y_test = [], []
    for images, labels in test_loader:
        images = images.view(images.size(0), -1).numpy()  # Flatten the images
        labels = labels.numpy()
        X_test.extend(images)
        y_test.extend(labels)

    # Convert to numpy arrays
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    print("Standardizing the test data...")
    # Standardize the data
    X_test = scaler.transform(X_test)

    print("Applying PCA to test data...")
    # Apply PCA
    X_test_pca = pca.transform(X_test)

    print("Testing the model...")
    # Test the model
    y_test_pred = model.predict(X_test_pca)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f'Test Accuracy: {test_accuracy:.3f}')

    cr_test = classification_report(y_test, y_test_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)
    print("Testing Classification Report:\n", cr_test)
    print("Testing Confusion Matrix:\n", cm_test)

    plot_classification_report(cr_test)
    plot_confusion_matrix(cm_test, classes=['glioma', 'meningioma', 'pituitary', 'normal'])


def main():
    print("Loading dataset...")
    # Handles our Dataset
    train_loader, val_loader, test_loader = utils.get_loaders("dataset", "extracted_dataset", BATCH_SIZE)

    print("Visualizing dataset...")
    # Visualizes our Dataset - For debugging/confirmation
    visualize_dataset(train_loader, val_loader, test_loader)
    show_batch(train_loader)

    print("Training model...")
    # Train the model
    model, scaler, pca = train_model(train_loader, val_loader, n_components=50)

    print("Testing model...")
    # Test the model
    test_model(model, scaler, pca, test_loader)

    input("Press Enter to close program and close plotted data/images...")


if __name__ == '__main__':
    main()
