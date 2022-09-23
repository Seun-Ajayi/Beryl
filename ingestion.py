import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from glob import glob



now = datetime.now()
with open('config.json','r') as f:
    config = json.load(f) 

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']



def merge_multiple_dataframe(input_folder=input_folder_path, existing_data="merged_dataset.csv"):
    """ check for datasets, compile them together, and write to an output file """
    if existing_data:
        df_merge = pd.read_csv(os.path.join(os.getcwd(), output_folder_path, existing_data))
    else:
        df_merge = pd.DataFrame(
            columns=[
                "corporation",
                "lastmonth_activity",
                "lastyear_activity",
                "number_of_employees",
                "exited"
                ]
            )

    with open(os.path.join(os.getcwd(), output_folder_path, 'ingestedfiles.txt'), "w") as f:
        for filename in sorted(glob(f'{input_folder}/*.csv')):
            df = pd.read_csv(os.getcwd()+"/"+filename)
            df_merge = df_merge.append(df)
            filename = filename.replace("practicedata/", "")
            filename = filename.replace("sourcedata/", "")
            f.write(filename) 
            f.write("\n")
          
        

    merged_dataset = df_merge.drop_duplicates()
    OUTPUT_DIR = os.path.join(os.getcwd(), output_folder_path)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    merged_dataset.to_csv(f"{OUTPUT_DIR}/merged_dataset.csv", index=False)





if __name__ == '__main__':
    merge_multiple_dataframe()
