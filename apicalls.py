import os
import requests
from training import preprocess_data
import json
from scoring import load_model
import pandas as pd
from datetime import datetime



URL = "http://127.0.0.1:8000/"
now = datetime.now()


def main():

    with open('config.json', 'r') as f:
            config = json.load(f)

    test_data_path = config['test_data_path']
    api_responses = config['api_responses']

    encoder = load_model(modelfile="encoder.pkl")
    data, *_ = preprocess_data(datapath=test_data_path, training=False, encoder=encoder)
    input_data = json.dumps(data.tolist())
    print(pd.DataFrame(json.loads(input_data)))

    headers = {'Content-Type': 'application/json'}

    predictions = requests.post(URL + 'prediction', input_data, headers=headers)
    print(predictions.json())

    score = requests.get(URL + 'scoring')
    print(score.json())

    stats = requests.get(URL + 'summarystats')
    print(stats.json())

    diagnosis = requests.get(URL + 'diagnostics')
    print(diagnosis.json())

    responses =  {
        "predictions": predictions.json(),
        "score": score.json(),
        "statistics": stats.json(),
        "diagnostics": diagnosis.json()
    }

    output_file = os.path.join(os.getcwd(), api_responses, f'apiresponses_{now.strftime("%Y%m%d%H%M%S")}.txt')
    with open(output_file, "w") as f:
        print(f"Writing API responses to {output_file}")
        f.write(json.dumps(responses, indent=4))


if __name__ == "__main__":
    main()
