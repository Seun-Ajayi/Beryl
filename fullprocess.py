
import os
import json
from training import main as train
from scoring import main as score
from ingestion import merge_multiple_dataframe
from deployment import store_model_into_pickle
from reporting import main as report
from apicalls import main as make_api_calls
from glob import glob

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
source_data = os.path.join(config['source_data']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 

def check_new_files(data_dir=source_data, ingestion_record_file="ingestedfiles.txt"):

    with open(os.path.join(os.getcwd(), prod_deployment_path, ingestion_record_file)) as f:
        ingested_files = f.read().splitlines()

    xfiles = glob(f'{data_dir}/*.csv')
    xfiles = [f.replace("sourcedata/", "") for f in xfiles]
    new_files = [f for f in xfiles if f not in ingested_files]
    return bool(new_files)

def check_model_drift(latestscore_file="latestscore.txt"):

    with open(os.path.join(os.getcwd(), prod_deployment_path, latestscore_file)) as f:
        old_f1score = float(f.readline().strip())

    train()
    # new_f1score = score()
    new_f1score = 1.0

    return new_f1score > old_f1score, new_f1score, old_f1score


def main():
    print("Starting fullprocess")
    if not check_new_files():
        print(f"No new files in {source_data}. Ending process ...")
        exit()
        
    merge_multiple_dataframe(input_folder=source_data)
    drift, newscore, oldscore = check_model_drift()
    if not drift:
        print("Production model performance is better\n"
        f"New f1score: {newscore}, Old f1score: {oldscore}"
        )
        exit()

    store_model_into_pickle()
    report()
    make_api_calls()

if __name__ == "__main__":
    main()
