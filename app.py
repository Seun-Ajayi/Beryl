from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import json
import os
from scoring import load_model, score_model
from diagnostics import model_predictions
from training import preprocess_data
from diagnostics import dataframe_summary, check_missing_values, outdated_packages_list, execution_time



app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json','r') as f:
    config = json.load(f) 

dataset_csv_path = os.path.join(config['output_folder_path']) 
test_data_path = os.path.join(config['test_data_path']) 
prod_deployment_path = os.path.join(config['prod_deployment_path'])

prediction_model = load_model()
encoder = load_model(modelfile="encoder.pkl")
X, y, _ = preprocess_data(datapath=test_data_path, encoder=encoder, training=False)

def read_data(filename):
    data = pd.read_csv(os.path.join(os.getcwd(), dataset_csv_path, filename))
    return data


@app.route("/prediction", methods=['POST','OPTIONS'])
def predict():        

    data_json = request.get_json()
    data = pd.DataFrame(data_json)
    print(data)
    predictions = model_predictions(prediction_model, data)
    return jsonify(predictions=predictions), 200


@app.route("/scoring", methods=['GET','OPTIONS'])
def score():        
    return jsonify(score=score_model(model=prediction_model, X=X, y=y)), 200


@app.route("/summarystats", methods=['GET','OPTIONS'])
def stats():        
    return jsonify(stats=dataframe_summary()), 200

@app.route("/diagnostics", methods=['GET','OPTIONS'])
def diagnose():        
    missing_values = check_missing_values()
    executiontime = execution_time()
    outdated_packages = outdated_packages_list()

    return jsonify(
        missing_values = missing_values,
        executionTime = executiontime,
        outdated_packages = outdated_packages
    ), 200

if __name__ == "__main__":    
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
