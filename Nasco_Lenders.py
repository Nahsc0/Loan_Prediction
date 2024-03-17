from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get features from the form
    Gender = request.form['Gender']
    Married = request.form['Married']
    Dependents = request.form['Dependents']
    Education = request.form['Education']
    Self_Employed = request.form['Self_Employed']
    ApplicantIncome = float(request.form['ApplicantIncome'])
    CoapplicantIncome = float(request.form['CoapplicantIncome'])
    LoanAmount = float(request.form['LoanAmount'])
    Loan_Amount_Term = float(request.form['Loan_Amount_Term'])
    Credit_History = float(request.form['Credit_History'])
    Property_Area = request.form['Property_Area']

    # Make prediction
    features = [[Gender, Married, Dependents, Education, Self_Employed,
                 ApplicantIncome, CoapplicantIncome, LoanAmount,
                 Loan_Amount_Term, Credit_History, Property_Area]]
    prediction = model.predict(features)

    # Return the prediction
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
