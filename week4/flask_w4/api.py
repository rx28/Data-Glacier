X = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]]
y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

from sklearn.linear_model import LinearRegression
import joblib

# Train the model
model = LinearRegression()
model.fit(X, y) 

# Save the model
joblib.dump(model, 'linear_regression_model.pkl')

from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('linear_regression_model.pkl')

# Define a route to accept input and return output
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input value from the request
    input_value = float(request.form['X'])

    # Use the trained model to make a prediction
    output = model.predict([[input_value]])

    # Return the predicted output as a string
    return render_template('index.html', prediction='Prediction of the model is {}'.format(output))

if __name__ == '__main__':
    app.run()
