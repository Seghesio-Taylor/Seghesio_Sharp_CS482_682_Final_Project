# Authors: Taylor Seghesio & Garrett Sharp
# Date: 05DEC2024
# Course: CS 482/ CS 682
# Final Project

# Acknowledgements:
# https://pandas.pydata.org/docs/user_guide/index.html#user-guide ### pandas
# https://matplotlib.org/stable/tutorials/pyplot.html ### plots
# https://machinelearningmastery.com/visualize-gradient-boosting-decision-trees-xgboost-python/ ### XGBoost visual
# https://xgboost.readthedocs.io/en/stable/python/python_intro.html ### XGBoost
# https://numpy.org/devdocs/user/index.html#user ###Numpy
# https://scikit-learn.org/dev/modules/generated/sklearn.metrics.precision_recall_fscore_support.html ### Sklearn prec_recall_supp
# https://scikit-learn.org/1.5/modules/generated/sklearn.metrics.accuracy_score.html ### accuracy score

# Acknowledgments: Much of this code was written with the help of provided resources from UNR CS 682 course by Dr. Ankita
# Shukla, and the course textbook: Artificial Intelligence: A Modern Approach by Stuart Russel and Peter Norvig. My
# ability to finish the program was performed from the combined resources provided in the course and the sources listed
# above.

# data handling
import pandas as pd
import numpy as np
# Boosting and model
import xgboost as xgb
import utils
import torch
# metrics and visuals
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


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
        X_val, y_val = [], []

        for inputs, labels in self.train_loader:
            X_train.append(inputs.view(inputs.size(0), -1).numpy())
            y_train.append(labels.numpy())

        X_train = np.vstack(X_train)
        y_train = np.hstack(y_train)
        train_data = xgb.DMatrix(X_train, label=y_train)

        for inputs, labels in self.val_loader:
            X_val.append(inputs.view(inputs.size(0), -1).numpy())
            y_val.append(labels.numpy())

        X_val = np.vstack(X_val)
        y_val = np.hstack(y_val)
        val_data = xgb.DMatrix(X_val, label=y_val)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        params = {
            'objective': 'multi:softmax',
            'num_class': 4,
            'eval_metric': 'mlogloss',
            'tree_method': 'hist',
            'device': device,
            'learning_rate': 0.01,
            'max_depth': 6,
            'reg_alpha': 0.5,
            'reg_lambda': 3.0
        }

        print("Training XGBTree...")
        self.model = xgb.train(params, train_data, num_boost_round=200, evals=[(val_data, 'validation')], early_stopping_rounds=10)
        train_predictions = self.model.predict(train_data)
        train_accuracy = accuracy_score(y_train, train_predictions)
        train_precision, train_recall, train_f1, train_support = precision_recall_fscore_support(y_train, train_predictions, average=None)

        return train_accuracy, train_precision, train_recall, train_f1, train_support

    def evaluate_model(self, dataloader):
        X_test, y_test = [], []

        for inputs, labels in dataloader:
            X_test.append(inputs.view(inputs.size(0), -1).numpy())
            y_test.append(labels.numpy())

        X_test = np.vstack(X_test)
        y_test = np.hstack(y_test)

        test_data = xgb.DMatrix(X_test)

        if dataloader == self.val_loader:
            print("\nEvaluating model...")
        else:
            print("\nTesting model...")

        predictions = self.model.predict(test_data)
        accuracy = accuracy_score(y_test, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(y_test, predictions, average=None)

        return accuracy, precision, recall, f1, support

    def plot_tree(self, num_tree=0):
        xgb.plot_tree(self.model, num_trees=num_tree)
        plt.show()


def metrics_table(precision, recall, f1, support, classes):
    metrics = pd.DataFrame({'Class': classes, 'Precision': precision, 'Recall': recall, 'F1-Score': f1, 'Support': support})
    print(metrics)


def main():
    train_loader, val_loader, test_loader = utils.get_loaders("dataset", "extracted_dataset", BATCH_SIZE)
    classifier = MyXGBDecisionTree(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)
    classes = ['glioma tumor', 'meningioma tumor', 'pituitary tumor', 'no tumor']

    train_acc, train_precision, train_recall, train_f1, train_support = classifier.train()
    print("\nTraining Metrics:")
    print("Training Accuracy:", train_acc*100)
    metrics_table(train_precision, train_recall, train_f1, train_support, classes)

    val_acc, val_precision, val_recall, val_f1, val_support = classifier.evaluate_model(val_loader)
    print("\nValidation Metrics:")
    print("Validation Accuracy:", val_acc*100)
    metrics_table(val_precision, val_recall, val_f1, val_support, classes)

    test_acc, test_precision, test_recall, test_f1, test_support = classifier.evaluate_model(test_loader)
    print("\nTest Metrics:")
    print("Test Accuracy:", test_acc*100)
    metrics_table(test_precision, test_recall, test_f1, test_support, classes)

    classifier.plot_tree()


if __name__ == '__main__':
    main()
