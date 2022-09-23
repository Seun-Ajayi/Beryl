
import pandas as pd
import numpy as np
import timeit
import os
import json
import subprocess
from scoring import load_model
from training import preprocess_data


with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 



def model_predictions(model, X):
    """ This functiomread the deployed model and a test dataset, calculate predictions """
    preds = model.predict(X)
    return [float(x) for x in list(preds)]

def dataframe_summary(data_folder=dataset_csv_path, cleaned_datafle="merged_dataset.csv"):
    """This function calculates summary statistics. """
    data = pd.read_csv(os.path.join(os.getcwd(), data_folder, cleaned_datafle))
    stats = data.select_dtypes([np.number]).agg(["mean", "median", "std", "count"])
    return stats.values.tolist()

def ingestion_timing():
    starttime = timeit.default_timer()
    os.system("python ingestion.py")
    timing = timeit.default_timer() - starttime
    return timing

def training_timing():
    starttime = timeit.default_timer()
    os.system("python training.py")
    timing = timeit.default_timer() - starttime
    return timing

def execution_time():
    """ calculate timing of training.py and ingestion.py """
    ingestion_timings = []
    training_timings = []

    for _ in range(20):
        ingestion_timings.append(ingestion_timing())
        training_timings.append(training_timing())

    final_output = []
    final_output.append(np.mean(ingestion_timings))
    final_output.append(np.mean(training_timings))
    
    return {"Ingestion_timing": final_output[0], "Training_timing": final_output[1]}

def check_missing_values(data_folder=dataset_csv_path, cleaned_datafle="merged_dataset.csv"):
    """ This function checks for the % of missing values in the dataset inputed """

    data = pd.read_csv(os.path.join(os.getcwd(), data_folder, cleaned_datafle))
    missing_values = data.isna().sum() / data.shape[0]
    return list(missing_values)



def outdated_packages_list():
    """ Function to check dependencies; outdated packages """
    outdated =  subprocess.run(
        "pip list --outdated --format columns",
        shell=True,
        capture_output=True,
        check=True,
        text=True)
    with open(os.path.join(os.getcwd(), prod_deployment_path, "outdated_packages.txt"), "w") as f:
        f.write(str(outdated.stdout))
    return str(outdated.stdout)



if __name__ == '__main__':
    model = load_model()
    encoder = load_model("encoder.pkl")
    X, *_ = preprocess_data(datapath=test_data_path, training=False, encoder=encoder)
    model_predictions(model, X)
    dataframe_summary()
    check_missing_values()
    execution_time()
    outdated_packages_list()
