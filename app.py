from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import os

app = Flask(__name__)

# Load models
model_names = [
    'LinearRegression', 'RobustRegression', 'RidgeRegression', 'LassoRegression', 'ElasticNet', 
    'PolynomialRegression', 'SGDRegressor', 'ANN', 'RandomForest', 'SVM', 'LGBM', 
    'XGBoost', 'KNN'
]
models = {name: pickle.load(open(f'{name}.pkl', 'rb')) for name in model_names}

# Load evaluation results
results_df = pd.read_csv('model_evaluation_results.csv')

# Path to store prediction logs
PREDICTION_LOG = 'prediction_logs.csv'

@app.route('/')
def index():
    return render_template('index.html', model_names=model_names)

@app.route('/predict', methods=['POST'])
def predict():
    model_name = request.form['model']
    input_data = {
        'Avg. Area Income': float(request.form['Avg. Area Income']),
        'Avg. Area House Age': float(request.form['Avg. Area House Age']),
        'Avg. Area Number of Rooms': float(request.form['Avg. Area Number of Rooms']),
        'Avg. Area Number of Bedrooms': float(request.form['Avg. Area Number of Bedrooms']),
        'Area Population': float(request.form['Area Population'])
    }
    input_df = pd.DataFrame([input_data])
    
    if model_name in models:
        model = models[model_name]
        prediction = model.predict(input_df)[0]

        # Log the prediction to a CSV file
        log_data = input_data.copy()
        log_data['Model'] = model_name
        log_data['Prediction'] = prediction
        log_df = pd.DataFrame([log_data])

        # Append to the CSV file
        if not os.path.exists(PREDICTION_LOG):
            log_df.to_csv(PREDICTION_LOG, index=False)
        else:
            log_df.to_csv(PREDICTION_LOG, mode='a', header=False, index=False)

        return render_template('results.html', prediction=prediction, model_name=model_name)
    else:
        return jsonify({'error': 'Model not found'}), 400

@app.route('/results')
def results():
    return render_template('model.html', tables=[results_df.to_html(classes='data')], titles=results_df.columns.values)

if __name__ == '__main__':
    app.run(debug=True)
