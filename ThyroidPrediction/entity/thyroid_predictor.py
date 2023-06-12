import os
import sys
from ThyroidPrediction.logger import logging
from ThyroidPrediction.exception import ThyroidException
from ThyroidPrediction.util.util import load_object

import pandas as pd


class ThyroidData:

    def __init__(self, age: float, sex, on_thyroxine, on_antithyroid_medication, sick, pregnant,
                 thyroid_surgery, I131_treatment, query_hypothyroid, query_hyperthyroid, lithium, goitre,
                 tumor, hypopituitary, psych, TSH: float, T3: float, TT4: float, T4U: float, FTI: float,
                 ):

        logging.info(f"{'>>' * 30} ThyroidData log started {'<<' * 30} ")

        try:
            self.age = age
            self.sex = sex
            self.on_thyroxine = on_thyroxine
            #self.query_on_thyroxine = query_on_thyroxine
            self.on_antithyroid_medication = on_antithyroid_medication
            self.sick = sick
            self.pregnant = pregnant
            self.thyroid_surgery = thyroid_surgery
            self.I131_treatment = I131_treatment
            self.query_hypothyroid = query_hypothyroid
            self.query_hyperthyroid = query_hyperthyroid
            self.lithium = lithium
            self.goitre = goitre
            self.tumor = tumor
            self.hypopituitary = hypopituitary
            self.psych = psych
            self.TSH = TSH
            self.T3 = T3
            self.TT4 = TT4
            self.T4U = T4U
            self.FTI = FTI
            #self.referral_source_SVHC = referral_source_SVHC
            #self.referral_source_SVHD = referral_source_SVHD
            #self.referral_source_SVI = referral_source_SVI
            #self.referral_source_other = referral_source_other

        except Exception as e:
            raise ThyroidException(e, sys) from e

    def get_thyroid_input_data_frame(self):

        try:
            logging.info(f"Converting to DataFrame")
            thyroid_input_dict = self.get_thyroid_data_as_dict()

            print(pd.DataFrame(thyroid_input_dict).columns)

            return pd.DataFrame(thyroid_input_dict)

        except Exception as e:
            raise ThyroidException(e, sys) from e

    def get_thyroid_data_as_dict(self):
        try:
            logging.info(f"getting thyroid data as_dict")

            input_data = {"age": [self.age],
                          "sex":[self.sex],
                          "on_thyroxine":[self.on_thyroxine],
                          #"query_on_thyroxine":[self.query_on_thyroxine],
                          "on_antithyroid_medication":[self.on_antithyroid_medication],
                          "sick":[self.sick],
                          "pregnant":[self.pregnant],
                          "thyroid_surgery":[self.thyroid_surgery],
                          "I131_treatment":[self.I131_treatment],
                          "query_hypothyroid":[self.query_hypothyroid],
                          "query_hyperthyroid":[self.query_hyperthyroid],
                          "lithium":[self.lithium],
                          "goitre":[self.goitre],
                          "tumor":[self.tumor],
                          "hypopituitary":[self.hypopituitary],
                          "psych":[self.psych],

                          "TSH": [self.TSH],
                          "T3": [self.T3],
                          "TT4": [self.TT4],
                          "T4U": [self.T4U],
                          "FTI": [self.FTI],

                          #"referral_source_SVHC":[self.referral_source_SVHC],
                          #"referral_source_SVHD":[self.referral_source_SVHD],
                          #"referral_source_SVI":[self.referral_source_SVI],
                          #"referral_source_other":[self.referral_source_other],
                          }
           
            return input_data
        except Exception as e:
            raise ThyroidException(e, sys)


class ThyroidPredictor:

    def __init__(self, model_dir: str):
        try:
            logging.info(f"{'>>' * 30} ThyroidPredictor log started {'<<' * 30} ")

            self.model_dir = model_dir
        except Exception as e:
            raise ThyroidException(e, sys) from e

    def get_latest_model_path(self):
        try:
            logging.info(f"getting latest model path")

            folder_name = list(map(int, os.listdir(self.model_dir)))
            latest_model_dir = os.path.join(self.model_dir, f"{max(folder_name)}")
            file_name = os.listdir(latest_model_dir)[0]
            latest_model_path = os.path.join(latest_model_dir, file_name)

            logging.info(f"latest model path: [ {latest_model_path} ]")
            
            return latest_model_path
        except Exception as e:
            raise ThyroidException(e, sys) from e

    def predict(self, X):
        try:
            logging.info(f"ThyroidPredictor is Making Predictions")

            model_path = self.get_latest_model_path()
            model = load_object(file_path=model_path)
            
            logging.info(f"Model objct loaded from path: [ {model_path} ]")

           #major_class_prediction = model.predict(X)
           ## Verified
           #major_class = ['binding protein','discordant', 'goitre', 'hyperthyroid', 'hypothyroid', 'negative','replacement_therapy', 'sick']
           #major_class_encoded = [0, 1, 2, 3, 4, 5, 6, 7]

           #if major_class_prediction == 0:
           #    major_class_prediction = 'Binding Protein'
           #elif major_class_prediction == 1:
           #    major_class_prediction = "Discordant"
           #elif major_class_prediction == 2:
           #    major_class_prediction == "Goitre"
           #elif major_class_prediction == 3:
           #    major_class_prediction = "Hyperthyroid"
           #elif major_class_prediction == 4:
           #    major_class_prediction == "Hypothyroid"
           #elif major_class_prediction == 5:
           #    major_class_prediction == "Negative"
           #elif major_class_prediction == 6:
           #    major_class_prediction == "Replacement Therapy"
           #elif major_class_prediction == 7:
           #    major_class_prediction == "Sick"

            major_class_mapping = {0: 'Binding Protein',
                                   1: 'Discordant',
                                   2: 'Goitre',
                                   3: 'Hyperthyroid',
                                   4: 'Hypothyroid',
                                   5: 'Negative',
                                   6: 'Replacement Therapy',
                                   7: 'Sick'}
            
            major_class_prediction = major_class_mapping[model.predict(X)[0]]

            logging.info(f"Predictions: [ {major_class_prediction} ]")

            return major_class_prediction

        except Exception as e:
            raise ThyroidException(e, sys) from e
        
    def bulk_prediction(self, X):
        try:
            logging.info(f"ThyroidPredictor is Making Predictions")

            model_path = self.get_latest_model_path()
            model = load_object(file_path=model_path)
            
            logging.info(f"Model objct loaded from path: [ {model_path} ]")

           

            major_class_mapping = {0: 'Binding Protein',
                                   1: 'Discordant',
                                   2: 'Goitre',
                                   3: 'Hyperthyroid',
                                   4: 'Hypothyroid',
                                   5: 'Negative',
                                   6: 'Replacement Therapy',
                                   7: 'Sick'}
            
            major_class_prediction = model.predict(X)
            
            logging.info(f"Predictions: [ {major_class_prediction} ]")
            # returning original prdictions without mapping but it will be don in app.py file
            return major_class_prediction

        except Exception as e:
            raise ThyroidException(e, sys) from e        