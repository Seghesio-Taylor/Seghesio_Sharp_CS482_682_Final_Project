# Authors: Taylor Seghesio & Garrett Sharp
# Date: 05DEC2024
# Course: CS 682
# Final Project

# sources
# https://pandas.pydata.org/docs/user_guide/index.html#user-guide
# https://matplotlib.org/stable/tutorials/pyplot.html
# https://scikit-learn.org/dev/modules/generated/sklearn.metrics.accuracy_score.html
# https://xgboost.readthedocs.io/en/stable/python/python_intro.html

# Acknowledgments: Much of this code was written with the help of provided resources from UNR CS 682 course by Dr. Ankita
# Shukla, and the course textbook: Artificial Intelligence: A Modern Approach by Stuart Russel and Peter Norvig. My
# ability to finish the program was performed from the combined resources provided in the course and the sources listed
# above.

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score


import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
import utils
from utils import get_loaders

import torch

# GLOBALS
BATCH_SIZE = 128


class MyXGBDecisionTree:
    def __init__(self, train_loader=None, val_loader=None, test_loader=None):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.model = None

    def train(self):
        X_train, y_train = [], []

        for inputs, labels in self.train_loader:
            X_train.append(inputs.view(inputs.size(0), -1).numpy())
            y_train.append(labels.numpy())

        X_train = np.vstack(X_train)
        y_train = np.hstack(y_train)
        train_data = xgb.DMatrix(X_train, label=y_train)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        params = {
            'objective': 'multi:softmax',
            'num_class': 4,
            'eval_metric': 'mlogloss',
            'tree_method': 'hist',
            'device': device,
            'learning_rate': 0.1,
            'max_depth': 4,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0
        }

        self.model = xgb.train(params, train_data, num_boost_round=100)
        #self.model.fit(X_train, y_train)

        train_predictions = self.model.predict(train_data)
        train_accuracy = accuracy_score(y_train, train_predictions)

        return train_accuracy

    def evaluate_model(self, dataloader):
        X_test, y_test = [], []

        for inputs, labels in dataloader:
            X_test.append(inputs.view(inputs.size(0), -1).numpy())
            y_test.append(labels.numpy())

        X_test = np.vstack(X_test)
        y_test = np.hstack(y_test)

        test_data = xgb.DMatrix(X_test)

        predictions = self.model.predict(test_data)
        accuracy = accuracy_score(y_test, predictions)

        return accuracy

    def plot_tree(self, num_tree=0):
        #plt.figure(figsize=(40, 20))
        xgb.plot_tree(self.model, num_trees=num_tree)
        plt.show()


if __name__ == '__main__':
    train_loader, val_loader, test_loader = utils.get_loaders("dataset", "extracted_dataset", BATCH_SIZE)

    classifier = MyXGBDecisionTree(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)

    train_acc = classifier.train()
    print("Training Accuracy:", train_acc)

    val_acc = classifier.evaluate_model(val_loader)
    print("Validation Accuracy:", val_acc)

    test_acc = classifier.evaluate_model(test_loader)
    print("Test Accuracy:", test_acc)

    classifier.plot_tree()
