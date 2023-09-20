# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Trains ML model using training dataset. Saves trained model.
"""

import argparse
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import os
import pandas as pd
import mlflow


def parse_args():
    '''Parse input arguments'''

    parser = argparse.ArgumentParser("train")
    parser.add_argument("--train_data", type=str, help="Path to train dataset")
    parser.add_argument("--model_output", type=str, help="Path of output model")

    # classifier specific arguments
    parser.add_argument('--regression_C', type=float, default=1.0,
                        help='C')
    parser.add_argument('--regression_penalty', type=str, default='l1',
                        help='penalty')
    parser.add_argument('--regression_solver', type=str, default='liblinear',
                        help='solver')
    args = parser.parse_args()

    return args

def select_first_file(path):
    """Selects first file in folder, use under assumption there is only one file in folder
    Args:
        path (str): path to directory or file to choose
    Returns:
        str: full path of selected file
    """
    files = os.listdir(path)
    for file in files:
        if file.endswith('.csv'):
            return os.path.join(path, file)

def main(args):
    """Main function of the script."""

    # paths are mounted as folder, therefore, we are selecting the file from folder
    train_df = pd.read_csv(select_first_file(args.train_data))
    # Extracting the label column
    y_train = train_df.pop("is_bug_inc")
    # convert the dataframe values to array
    X_train = train_df.values

    print(f"Training with data of shape {X_train.shape}")

    clf = LogisticRegression(C=args.regression_C, penalty=args.regression_penalty, solver=args.regression_solver)
    mlflow.log_param("C", args.regression_C)
    mlflow.log_param("penalty", args.regression_penalty)
    mlflow.log_param("solver", args.regression_solver)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_train)
    fpr, tpr, thresholds = metrics.roc_curve(y_train, y_pred)
    precision = metrics.precision_score(y_train, y_pred)
    recall = metrics.recall_score(y_train, y_pred)
    f1 = metrics.f1_score(y_train, y_pred)
    auc = metrics.auc(fpr, tpr)
    print(precision, recall, f1, auc)

    mlflow.log_metric("train precision", precision)
    mlflow.log_metric("train recall", recall)
    mlflow.log_metric("train f1", f1)
    mlflow.log_metric("train auc", auc)

    # Save the model
    mlflow.sklearn.save_model(sk_model=clf, path=args.model_output)

if __name__ == "__main__":
    
    mlflow.start_run()

    # ---------- Parse Arguments ----------- #
    # -------------------------------------- #

    args = parse_args()

    lines = [
        f"Train dataset input path: {args.train_data}",
        f"Model output path: {args.model_output}",
        f"C: {args.regression_C}",
        f"penalty: {args.regression_penalty}",
        f"solver: {args.regression_solver}"
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()
    