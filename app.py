from ast import mod
from email import message
from email.mime import application
from ipaddress import collapse_addresses
from re import S
from numpy import dtype
import pandas as pd
from sklearn.metrics import f1_score
from ThyroidPrediction.logger import get_log_dataframe
from flask import Flask, request
import sys

import pip
from ThyroidPrediction.util.util import read_yaml_file, write_yaml_file
from matplotlib.style import context
from ThyroidPrediction.logger import logging
from ThyroidPrediction.exception import ThyroidException
import os
import sys
import json
from ThyroidPrediction.config.configuration import Configuration
from ThyroidPrediction.constant import CONFIG_DIR, get_current_time_stamp
from ThyroidPrediction.pipeline.pipeline import Pipeline
from ThyroidPrediction.entity.thyroid_predictor import ThyroidPredictor, ThyroidData
from flask import send_file, abort, render_template


ROOT_DIR = os.getcwd()
LOG_FOLDER_NAME = "logs"
PIPELINE_FOLDER_NAME = "ThyroidPrediction"
SAVED_MODELS_DIR_NAME = "saved_models"

MODEL_CONFIG_FILE_PATH = os.path.join(ROOT_DIR, CONFIG_DIR, "model.yaml")
LOG_DIR = os.path.join(ROOT_DIR, LOG_FOLDER_NAME)
PIPELINE_DIR = os.path.join(ROOT_DIR, PIPELINE_FOLDER_NAME)
MODEL_DIR = os.path.join(ROOT_DIR, SAVED_MODELS_DIR_NAME)


THYROID_DATA_KEY = "thyroid_data"
THYROID_PREDICTION_VALUE_KEY = "thyroid_prediction_value"


app = Flask(__name__)



@app.route('/artifact', defaults={'req_path': 'ThyroidPrediction'})
@app.route('/artifact/<path:req_path>')
def render_artifact_dir(req_path):

    logging.info(f"req_path: {req_path}")
    
    os.makedirs(req_path, exist_ok=True)
    # Joining the base and the requested path
    print(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        if ".html" in abs_path:
            with open(abs_path, "r", encoding="utf-8") as file:
                content = ''
                for line in file.readlines():
                    content = f"{content}{line}"
                return content
        return send_file(abs_path)

    # Show directory contents
    files = {os.path.join(abs_path, file_name): file_name for file_name in os.listdir(abs_path) if
             "artifact" in os.path.join(abs_path, file_name)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('files.html', result=result)


@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        #print(MODEL_DIR)
        #print(ThyroidPredictor(MODEL_DIR).get_latest_model_path())
        latest_model_dir = ThyroidPredictor(MODEL_DIR).get_latest_model_path()
        head, _ = os.path.split(latest_model_dir)
        model_score_file_dir = os.path.join(head, "score")
        os.makedirs(model_score_file_dir, exist_ok=True)
        
        print(model_score_file_dir)
        
        model_scores = pd.read_csv(os.path.join(model_score_file_dir,"model_score.csv"))
        model_scores.set_index("models",drop=True, inplace=True)
        print(model_scores.columns)
        
        f1_score_train = round(model_scores.loc["RandomForestClassifier","f1_weighted_train"], 3)
        f1_score_test = round(model_scores.loc["RandomForestClassifier","f1_weighted_test"], 3)
        #print(f1_score_train)
        
        roc_auc_ovr_weighted_train = round(model_scores.loc["RandomForestClassifier","roc_auc_ovr_weighted_train"], 3)
        roc_auc_ovr_weighted_test = round(model_scores.loc["RandomForestClassifier","roc_auc_ovr_weighted_test"], 3)        

        balanced_accuracy_train = round(model_scores.loc["RandomForestClassifier","balanced_accuracy_train"], 3)
        balanced_accuracy_test = round(model_scores.loc["RandomForestClassifier","balanced_accuracy_test"], 3)

        log_loss_train = round(model_scores.loc["RandomForestClassifier","log_loss_train"], 3)
        log_loss_test = round(model_scores.loc["RandomForestClassifier","log_loss_test"], 3)
        

        return render_template('index.html', f1_score_train = f1_score_train, f1_score_test= f1_score_test, roc_auc_ovr_weighted_test = roc_auc_ovr_weighted_test, roc_auc_ovr_weighted_train = roc_auc_ovr_weighted_train, balanced_accuracy_test=balanced_accuracy_test, balanced_accuracy_train= balanced_accuracy_train, log_loss_test= log_loss_test, log_loss_train= log_loss_train)
    
    except Exception as e:
        return str(e)


@app.route('/reports', methods=['GET', 'POST'])
def render_eda():
    try:
        return render_template('eda.html', report1 = "ProfileReport_1.html")
    except Exception as e:
        return str(e)

@app.route('/reports_1', methods=['GET', 'POST'])
def render_profile_report_1():
    try:
        return render_template("ProfileReport_1.html")
    except Exception as e:
        raise ThyroidException(e, sys) from e
    
@app.route('/reports_2', methods=['GET', 'POST'])
def render_profile_report_2():
    try:
        return render_template("ProfileReport_2.html")
    except Exception as e:
        raise ThyroidException(e, sys) from e    
    
@app.route('/reports_3', methods=['GET', 'POST'])
def render_drift_report():
    try:
        return render_template("drift_report.html")
    except Exception as e:
        raise ThyroidException(e, sys) from e        

@app.route('/reports_4', methods=['GET', 'POST'])
def render_performance_report():
    try:
        return render_template("PerformanceReport.html")
    except Exception as e:
        raise ThyroidException(e, sys) from e        

@app.route('/view_experiment_hist', methods=['GET', 'POST'])
def view_experiment_history():
    logging.info(f"req_path: view_experiment_hist")

    experiment_df = Pipeline.get_experiments_status()
    context = {
        "experiment": experiment_df.to_html(classes='table table-striped col-12')
    }
    return render_template('experiment_history.html', context=context)


@app.route('/train', methods=['GET', 'POST'])
def train():
    
    logging.info(f"req_path: train")

    message = ""
    pipeline = Pipeline(config=Configuration(time_stamp=get_current_time_stamp()))

    if not Pipeline.experiment.running_status:
        message = """
                    Training Started....                     
                    """
        pipeline.start()
    else:
        message = """
                    Training is Already in progress...
                    """
    context = {
        "experiment": pipeline.get_experiments_status().to_html(classes='table table-striped col-12'),
        "message": message
    }
    return render_template('train.html', context=context)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    logging.info(f"req_path: predict")

    message = ""
    context = {
        THYROID_DATA_KEY: None,
        THYROID_PREDICTION_VALUE_KEY: None
    }

    if request.method == 'POST':
        age = float(request.form['age'])
        sex = float(request.form['sex'])
        on_thyroxine = float(request.form['on_thyroxine'])
        #query_on_thyroxine = float(request.form['query_on_thyroxine'])
        on_antithyroid_medication = float(request.form['on_antithyroid_medication'])
        sick = float(request.form['sick'])
        pregnant = float(request.form['pregnant'])
        thyroid_surgery = float(request.form['thyroid_surgery'])
        I131_treatment = float(request.form['I131_treatment'])
        query_hypothyroid = float(request.form['query_hypothyroid'])
        query_hyperthyroid = float(request.form['query_hyperthyroid'])
        lithium = float(request.form['lithium'])
        goitre = float(request.form['goitre'])
        tumor = float(request.form['tumor'])
        hypopituitary = float(request.form['hypopituitary'])
        TSH = float(request.form['tsh'])
        T3 = float(request.form["t3"])
        TT4 = float(request.form["tt4"])
        T4U = float(request.form["t4u"])
        FTI = float(request.form["fti"])
        psych = float(request.form["psych"])

        #referral_source_SVHC = float(request.form["referral_source_SVHC"])
        #referral_source_SVHD = float(request.form["referral_source_SVHD"])
        #referral_source_SVI = float(request.form["referral_source_SVI"])
        #referral_source_other = float(request.form["referral_source_other"])

        #referral_source = float(request.form["referral_source"])

        #if referral_source==0:
        #    referral_source_SVHC = 1
        #    referral_source_SVHD = 0
        #    referral_source_SVI = 0
        #    referral_source_other = 0

        #elif referral_source==1:
        #    referral_source_SVHC = 0
        #    referral_source_SVHD = 1
        #    referral_source_SVI = 0
        #    referral_source_other = 0

        #elif referral_source==2:
        #    referral_source_SVHC = 0
        #    referral_source_SVHD = 0
        #    referral_source_SVI = 1
        #    referral_source_other = 0

        #elif referral_source==3:
        #    referral_source_SVHC = 0
        #    referral_source_SVHD = 0
        #    referral_source_SVI = 0
        #    referral_source_other = 1
            

            

        thyroid_data = ThyroidData(age=age,
                                   TSH= TSH,
                                   T3= T3,
                                   TT4= TT4,
                                   T4U=T4U,
                                   FTI=FTI,
                                   sex= sex,
                                   on_thyroxine= on_thyroxine,
                                   #query_on_thyroxine= query_on_thyroxine,
                                   on_antithyroid_medication= on_antithyroid_medication,
                                   sick= sick,
                                   pregnant= pregnant,
                                   thyroid_surgery= thyroid_surgery,
                                   I131_treatment= I131_treatment,
                                   query_hypothyroid=query_hypothyroid,
                                   query_hyperthyroid=query_hyperthyroid,
                                   lithium= lithium,
                                   goitre=goitre,
                                   tumor=tumor,
                                   hypopituitary=hypopituitary,
                                   psych=psych,
                                   #referral_source_SVHC= referral_source_SVHC,
                                   #referral_source_SVHD=referral_source_SVHD,
                                   #referral_source_SVI=referral_source_SVI,
                                   #referral_source_other=referral_source_other

                                   )
        
        thyroid_df = thyroid_data.get_thyroid_input_data_frame()
        thyroid_predictor = ThyroidPredictor(model_dir=MODEL_DIR)
        thyroid_prediction = thyroid_predictor.predict(X=thyroid_df)
        
        context = {
            THYROID_DATA_KEY: thyroid_data.get_thyroid_data_as_dict(),
            THYROID_PREDICTION_VALUE_KEY: thyroid_prediction
        }
        
        return render_template('predict.html', context=context)
    return render_template("predict.html", context=context)


@app.route('/predict_bulk', methods=['POST'])
def predict_bulk():
    logging.info(f"req_path: predict_bulk")
    
    try:
        if request.method == 'POST':
            file = request.files['file']
            thyroid_df = pd.read_csv(file)
            try:
                thyroid_df = thyroid_df.drop("major_class_encoded", axis=1)
                #print(thyroid_df.columns)
                #print(thyroid_df.dtypes )
            except:
                pass
                
            thyroid_predictor = ThyroidPredictor(model_dir=MODEL_DIR)
            try:
                thyroid_prediction = thyroid_predictor.bulk_prediction(X=thyroid_df)
            except Exception as e:
                raise ThyroidException(e, sys) from e

            major_class_mapping = {0: 'Binding Protein',
                                   1: 'Discordant',
                                   2: 'Goitre',
                                   3: 'Hyperthyroid',
                                   4: 'Hypothyroid',
                                   5: 'Negative',
                                   6: 'Replacement Therapy',
                                   7: 'Sick'}
            
            thyroid_df["Predictions"] = thyroid_prediction
            thyroid_df["Predictions"] = thyroid_df["Predictions"].map(major_class_mapping)  # Major class mapping and replacing numerical rpresentation of major classes with string values
            
            result = thyroid_df.to_html(show_dimensions=True,
                                         justify= "center",
                                         classes=['table', 'table-striped'],
                                         border=0.5,
                                         escape=True)

            message = "PREDICTION RESULT"
            
            return render_template("bulk_prediction.html", result=result, message= message)
        
        message = 'Something Is Wrong!'
        return render_template("bulk_prediction.html", result="", message= message)
    
    except Exception as e:
        raise ThyroidException(e, sys) from e


@app.route('/saved_models', defaults={'req_path': 'saved_models'})
@app.route('/saved_models/<path:req_path>')
def saved_models_dir(req_path):
    os.makedirs("saved_models", exist_ok=True)
    # Joining the base and the requested path
    print(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        return send_file(abs_path)

    # Show directory contents
    files = {os.path.join(abs_path, file): file for file in os.listdir(abs_path)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('saved_models_files.html', result=result)


@app.route("/update_model_config", methods=['GET', 'POST'])
def update_model_config():
    try:
        if request.method == 'POST':
            model_config = request.form['new_model_config']
            model_config = model_config.replace("'", '"')
            print(model_config)
            model_config = json.loads(model_config)

            write_yaml_file(file_path=MODEL_CONFIG_FILE_PATH,
                            data=model_config)

        model_config = read_yaml_file(file_path=MODEL_CONFIG_FILE_PATH)
        return render_template('update_model.html', result={"model_config": model_config})

    except Exception as e:
        logging.exception(e)
        return str(e)


@app.route(f'/logs', defaults={'req_path': f'{LOG_FOLDER_NAME}'})
@app.route(f'/{LOG_FOLDER_NAME}/<path:req_path>')
def render_log_dir(req_path):
    os.makedirs(LOG_FOLDER_NAME, exist_ok=True)

    # Joining the base and the requested path
    logging.info(f"req_path: {req_path}")
    abs_path = os.path.join(req_path)
    print(abs_path)

    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        log_df = get_log_dataframe(abs_path)
        context = {"log": log_df.to_html(classes="table-striped", index=False, max_rows=None)}
        return render_template('log.html', context=context)

    # Show directory contents
    files = {os.path.join(abs_path, file): file for file in os.listdir(abs_path)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('log_files.html', result=result)


if __name__ == "__main__":
    app.run(debug=True)