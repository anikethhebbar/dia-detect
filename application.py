from flask import Flask, request, render_template
import pickle
import numpy as np

application = Flask(__name__)
app = application

scaler = pickle.load(open("Model/scaler.pkl", "rb"))
model = pickle.load(open("Model/LogisticModel.pkl", "rb"))

@app.route('/', methods=['GET', 'POST'])
def predict_datapoint():
    result = ""

    if request.method == 'POST':

        BMI = float(request.form.get('BMI'))
        Age = float(request.form.get('Age'))
        JunkFoodIntake = int(request.form.get('JunkFoodIntake'))
        ExerciseCount = int(request.form.get('ExerciseCount'))
        Smoking = int(request.form.get('Smoking'))
        Drinking = int(request.form.get('Drinking'))
        Weight = float(request.form.get('Weight'))
        ParentDiabetes = int(request.form.get('ParentDiabetes'))
        SiblingDiabetes = int(request.form.get('SiblingDiabetes'))

        new_data = scaler.transform([[BMI, Age, JunkFoodIntake, ExerciseCount, Smoking, Drinking, Weight, ParentDiabetes, SiblingDiabetes]])
        predict = model.predict(new_data)
       
        if predict[0] == 1:
            result = 'High Risk'
        else:
            result = 'Low Risk'
            
        return render_template('single_prediction.html', result=result)

    else:
        return render_template('index.html')

# Route for BMI Calculator
@app.route('/bmi', methods=['GET', 'POST'])

def bmi_calculator():
    if request.method == 'POST':
        weight = float(request.form.get('weight'))
        height = float(request.form.get('height')) / 100  # Convert cm to meters
        bmi = weight / (height ** 2)
        return render_template('bmi_result.html', bmi=bmi)
    else:
        return render_template('bmi_calculator.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0")
