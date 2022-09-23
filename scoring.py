import pandas as pd
import numpy as np
import pickle
import os
from sklearn import metrics
import json
from training import preprocess_data 



with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path'])
model_path = os.path.join(config["output_model_path"]) 
model_scores_path = os.path.join(config['model_scores_path']) 

def load_model(modelpath=model_path, modelfile="trainedmodel.pkl"):

    with open(os.path.join(os.getcwd(), modelpath, modelfile), "rb") as f:
        model = pickle.load(f)
    return model

def score_model(X, y, model):
    """ this function takes a trained model, load test data, 
        and calculate an F1 score for the model relative to the test data 

        inputs:
            X - features
            y - target
            model - trained model

        outputs:
            metric - f1score
    """

    preds = model.predict(X)

    f1score = metrics.f1_score(preds, y)
    print(f1score)

    return f1score

def write_score_to_file(f1score, csv_file="modelscores.csv"):
    """ it writes the result to the latestscore.txt file
        and also appends the recent score to the modelscores dataframe
    """

    filepath = os.path.join(os.getcwd(), model_scores_path, csv_file)
    previousscores = pd.read_csv(filepath)
    maxversion=previousscores['version'].max()
    thisversion=maxversion+1
    new_row_f1 = {'metric':'f1score', 'version':thisversion, 'score':f1score}
    if f1score<previousscores.loc[previousscores['metric']=='f1score','score'].min():
        previousscores = previousscores.append(new_row_f1, ignore_index=True)
        previousscores.to_csv(filepath,index=False)

    with open(f"{os.getcwd()}/{model_scores_path}/latestscore.txt", "w") as f:
        f.write(str(f1score))

def main():
    model = load_model()
    encoder = load_model(modelfile="encoder.pkl")
    X_test, y_test, _ = preprocess_data(test_data_path, encoder=encoder, training=False)
    f1score = score_model(X_test, y_test, model)
    write_score_to_file(f1score)

    return f1score


if __name__ == "__main__":
    main()
    