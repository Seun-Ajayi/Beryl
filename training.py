import pandas as pd
import numpy as np
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
import json
from glob import glob


with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
model_path = os.path.join(config['output_model_path']) 

def preprocess_data(datapath=dataset_csv_path, training=True, encoder=None, dataframe=None, use_datapath=True):

    if use_datapath is True:
        datafiles = glob(f"{os.path.join(os.getcwd(), datapath)}/*.csv")
        data = pd.read_csv(datafiles[0])
    else:
        data = dataframe

    y = data["exited"]
    X = data.drop(columns=["exited"], axis=1)

    X_categorical = X["corporation"].values
    X_continuous = X.drop(["corporation"], axis=1)
    if training is True:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        X_categorical = encoder.fit_transform(X_categorical.reshape(-1, 1))
    else:
        X_categorical = encoder.transform(X_categorical.reshape(-1, 1))

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder

def save_encoder(encoder):

    with open(os.path.join(os.getcwd(), model_path, "encoder.pkl"), "wb") as f:
        pickle.dump(encoder, f)
        



def train_model(X, y, output_model="trainedmodel.pkl"):

    lr = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                    intercept_scaling=1, l1_ratio=None, max_iter=100,
                    multi_class='multinomial', n_jobs=None, penalty='l2',
                    random_state=0, solver='lbfgs', tol=0.0001, verbose=0,
                    warm_start=False)
    
    
    model = lr.fit(X, y)
    score = model.score(X,y)
    print(score)
   
    MODEL_PATH = os.path.join(os.getcwd(), model_path)
    os.makedirs(MODEL_PATH, exist_ok=True)
    with open(f"{MODEL_PATH}/{output_model}", "wb") as f:
        pickle.dump(model, f)

def main():
    X, y, encoder = preprocess_data(dataset_csv_path, training=True)
    train_model(X, y)
    save_encoder(encoder)
    

if __name__ == "__main__":
    main()
    