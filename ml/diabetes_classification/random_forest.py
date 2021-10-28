# file pima_django/pima/ml/diabetes_classification/random_forest.py
import os
import joblib
import numpy as np
import pandas as pd

class RandomForestClassifier:
    def __init__(self):
        path_to_artifacts = r"C:\Users\bbhagat\Documents\SingHealth\HPE\pima_django\pima\ml\trained_models"
        # self.values_fill_missing =  joblib.load(path_to_artifacts + "train_mode.joblib")
        # self.encoders = joblib.load(path_to_artifacts + "encoders.joblib")
        self.model = joblib.load(os.path.join(path_to_artifacts, "random_forest_best.pkl"))

    def preprocessing(self, input_data):
        # # JSON to pandas DataFrame
        # input_data = pd.DataFrame(input_data, index=[0])
        # # fill missing values
        # input_data.fillna(self.values_fill_missing)
        # # convert categoricals
        # for column in [
        #     "workclass",
        #     "education",
        #     "marital-status",
        #     "occupation",
        #     "relationship",
        #     "race",
        #     "sex",
        #     "native-country",
        # ]:
        #     categorical_convert = self.encoders[column]
        #     input_data[column] = categorical_convert.transform(input_data[column])
        input_data = np.expand_dims(list(input_data.values()), axis=0)

        return input_data

    def predict(self, input_data):
        return self.model.predict(input_data)

    def postprocessing(self, input_data):
        label = 'Diabetes'
        if input_data==0:
            label = 'No Diabetes'
        return {"class": input_data, "label": label, "status": "OK"}

    def compute_prediction(self, input_data):
        try:
            input_data = self.preprocessing(input_data)
            prediction = self.predict(input_data)[0]  # only one sample
            prediction = self.postprocessing(prediction)
        except Exception as e:
            return {"status": "Error", "message": str(e)}

        return prediction