from flask import Flask, request, jsonify, url_for, redirect, render_template
import joblib
import os
import warnings
import numpy as np
from sklearn.exceptions import InconsistentVersionWarning

# Suppress InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

app = Flask(__name__,template_folder="templates")


# Load the model
model_path = 'model_joblib_pcos1.pkl'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
model = joblib.load(model_path)
@app.route('/')
def home():
    return render_template("pcos123.html")
@app.route('/pcos123.html', methods=['GET'])
def pcos123():
    return render_template("pcos123.html")
@app.route('/dietplan.html')
def dietplan():
    return render_template("dietplan.html")

@app.route('/excersie plan.html')
def excersie():
    return render_template("excersie plan.html")
@app.route('/detect.html')
def detect():
    return render_template("detect.html")
@app.route('/predict',methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        features = [
            data['age'],
            data['bmi'],
            data['cycleLength'],
            data['cycleValue'],
            data['amh'],
            data['fshlh'],
            data['fsh'],
            data['weightGain'],
            data['follicleNoL'],
            data['follicleNoR'],
            data['avgFollicleSize']
        ]

        # Convert to a 2D array
        final_features = np.array(features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(final_features)

        # Return the result
        result = "Possibility of PCOS" if prediction[0] == 1 else "No PCOS"
        return jsonify({'message': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400
if __name__ == "__main__":
    app.run(debug=True)