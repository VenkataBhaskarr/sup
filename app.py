from flask import Flask, request, jsonify
import pandas as pd
from minisom import MiniSom
import numpy as np

app = Flask(__name__)


# Define a function to preprocess the input case
def preprocess_case(case):
    case = case.drop(['CNR'], axis=1)  # Drop CNR as it's not used in classification
    case['Nature of the case'] = case['Nature of the case'].map({'Criminal': 0, 'Civil': 1})
    case['Disposition time (months)'] = 12
    case['Average hearing time period (days)'] = 12
    print(case)
    return case


def classify_case(case, som):
    # Preprocess the input case
    preprocessed_case = preprocess_case(case)

    # Find the best-matching unit (BMU) for the input case
    bmu = som.winner(preprocessed_case.values.flatten())

    # Determine additional information based on the result
    result_str = '-'.join(map(str, bmu))
    additional_info = {}
    if result_str == '0-0':
        additional_info = {'Average hearing time period': '56-65 days', 'Disposition time': '21 months'}
    elif result_str == '1-0':
        additional_info = {'Average hearing time period': '60-70 days', 'Disposition time': '22 months'}
    elif result_str == '2-0':
        additional_info = {'Average hearing time period': '20-23 days', 'Disposition time': '18 months'}
    elif result_str == '3-0':
        additional_info = {'Average hearing time period': '21-90 days', 'Disposition time': '20 months'}
    elif result_str == '3-1':
        additional_info = {'Average hearing time period': '23-27 days', 'Disposition time': '10 months'}
    elif result_str == '2-1':
        additional_info = {'Average hearing time period': '22-90 days', 'Disposition time': '25 months'}
    elif result_str == '1-1':
        additional_info = {'Average hearing time period': '35-40 days', 'Disposition time': '35 months'}
    elif result_str == '0-1':
        additional_info = {'Average hearing time period': '57-64 days', 'Disposition time': '30 months'}
    elif result_str == '0-2':
        additional_info = {'Average hearing time period': '23-27 days', 'Disposition time': '29 months'}
    elif result_str == '1-2':
        additional_info = {'Average hearing time period': '33-37 days', 'Disposition time': '32 months'}
    elif result_str == '2-2':
        additional_info = {'Average hearing time period': '29-33 days', 'Disposition time': '29 months'}
    elif result_str == '3-2':
        additional_info = {'Average hearing time period': '30-34 days', 'Disposition time': '20 months'}
    elif result_str == '3-3':
        additional_info = {'Average hearing time period': '45-55 days', 'Disposition time': '18 months'}
    elif result_str == '2-3':
        additional_info = {'Average hearing time period': '28-32 days', 'Disposition time': '17 months'}
    elif result_str == '1-3':
        additional_info = {'Average hearing time period': '45-65 days', 'Disposition time': '16 months'}
    elif result_str == '0-3':
        additional_info = {'Average hearing time period': '35-45 days', 'Disposition time': '24 months'}
    else:
        additional_info = {'Average hearing time period': '35-45 days', 'Disposition time': '24 months'}
    # Add more conditions as needed

    # Combine the result and additional information
    result = {'cluster': result_str, **additional_info}

    return result


# Define an API endpoint to handle case classification
@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json()
    case = pd.DataFrame(data, index=[0])  # Assuming the data is sent as JSON

    # Reload the SOM weights for each request
    # Load the trained SOM model
    som = MiniSom(2, 2, 5)  # Adjust the input size according to your data
    som.weights = np.load('soma_weights.npy')  # Load your trained SOM weights

    result = classify_case(case, som)
    # now based on the result i have to also add some other information and send it as a JSON object
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True,port=5000)




