from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import io

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'Empty file'}), 400
    
    # Leer el archivo CSV en un DataFrame de pandas
    file_content = file.read().decode('utf-8')
    data = pd.read_csv(io.StringIO(file_content))


    # Cargar el modelo y realizar la predicción
    regresion_model = joblib.load('covid.pkl')
    prediction = regresion_model.predict(data)

    # Retornar la predicción en formato JSON
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)

