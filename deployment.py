import os
import json
import shutil



with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path']) 
output_model_path = os.path.join(config['output_model_path']) 
model_scores_path = os.path.join(config['model_scores_path']) 

def store_model_into_pickle(
    recent_model_file="trainedmodel.pkl",
    latest_score_file="latestscore.txt",
    ingested_file="ingestedfiles.txt",
    encoder_file = "encoder.pkl"
    ):
    """ copy the latest pickle file, the latestscore.txt value, 
        and the ingestfiles.txt file into the deployment directory
    """
    
    # copy model
    src_file = os.path.join(os.getcwd(), output_model_path, recent_model_file)
    dst_folder = os.path.join(os.getcwd(), prod_deployment_path)
    os.makedirs(dst_folder, exist_ok=True)
    shutil.copy(src_file, dst_folder)

    #copy model_score
    src_file1 = os.path.join(os.getcwd(), model_scores_path, latest_score_file)
    shutil.copy(src_file1, dst_folder)

    #copy updated datafiles
    src_file2 = os.path.join(os.getcwd(), dataset_csv_path, ingested_file)
    shutil.copy(src_file2, dst_folder)
   
    #copy encoder
    src_file3 = os.path.join(os.getcwd(), output_model_path, encoder_file)
    shutil.copy(src_file3, dst_folder)

        
        
if __name__ == "__main__":
    store_model_into_pickle()