import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from scoring import load_model
from training import preprocess_data
from diagnostics import model_predictions




with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
reports_path = os.path.join(config['reports_path']) 
test_data_path = os.path.join(config['test_data_path']) 


def plt_confusion_matrix(
    X_test, 
    y_test, 
    preds, 
    model, 
    image_path="confusion_matrix.jpg"
):
    """This function plots a confusion matrix using the test data and the deployed model."""

    cm = confusion_matrix(y_test, preds)
    print(cm)
    plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Blues);
    plt.savefig(
        os.path.join(os.getcwd(), reports_path, image_path),
        bbox_inches='tight',
        dpi=150)
    plt.close()


def plot_classification_report(
    y_train, 
    train_preds, 
    y_test, 
    test_preds, 
    image_path="Classification_Report.jpg"
    ):

    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str(f'Logistic Regression Train'), {
        'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, train_preds)), {
        'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str(f'Logistic Regression Test'), {
        'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, test_preds)), {
        'fontsize': 10}, fontproperties='monospace')  
    plt.axis('off')
    plt.savefig(
        os.path.join(os.getcwd(), reports_path, image_path),
        bbox_inches='tight',
        dpi=150)
    plt.close()


def main():
    model = load_model()
    X_train, y_train, encoder = preprocess_data()
    X_test, y_test, _ = preprocess_data(datapath=test_data_path, training=False, encoder=encoder)
    test_preds = model_predictions(model, X_test)
    train_preds = model_predictions(model, X_train)
    plt_confusion_matrix(
        X_test, 
        y_test, 
        test_preds, 
        model, 
    )
    plot_classification_report(
        y_train, 
        train_preds, 
        y_test, 
        test_preds
    )



if __name__ == '__main__':
    main()
