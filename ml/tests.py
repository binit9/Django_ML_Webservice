# file pima_django/pima/ml/diabetes_classification/tests.py
from django.test import TestCase
from diabetes_classification.random_forest import RandomForestClassifier

class MLTests(TestCase):
    def test_rf_algorithm(self):
        input_data = {
            "Pregnancies" : 1.0, 
            "Glucose" : 85.0, 
            "BloodPressure" : 64.0, 
            "SkinThickness" : 29.0, 
            "Insulin" : 0.0, 
            "BMI" : 23.3, 
            "DiabetesPedigreeFunction" : 0.672, 
            "Age" : 32.0
        }
        my_alg = RandomForestClassifier()
        response = my_alg.compute_prediction(input_data)
        self.assertEqual('OK', response['status'])
        self.assertTrue('label' in response)
        self.assertTrue('class' in response)
        print(response)

if __name__ == '__main__':
    test = MLTests()
    test.test_rf_algorithm()