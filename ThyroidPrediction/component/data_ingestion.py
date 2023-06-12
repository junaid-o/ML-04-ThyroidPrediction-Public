import shutil
from sqlite3 import Timestamp

from ThyroidPrediction.constant import CURRENT_TIME_STAMP, get_current_time_stamp

from ThyroidPrediction.entity.config_entity import DataIngestionConfig, BaseDataIngestionConfig
from ThyroidPrediction.exception import ThyroidException
from ThyroidPrediction.logger import logging
from ThyroidPrediction.entity.artifact_entity import DataIngestionArtifact, BaseDataIngestionArtifact
import os, sys

import tarfile
from six.moves import urllib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import LabelEncoder

import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTENC, RandomOverSampler, KMeansSMOTE
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.subplots as sp
import plotly.express as px
import plotly.io as pio
import pandas as pd
import os
import natsort

import ydata_profiling
from ydata_profiling import ProfileReport
from ydata_profiling.config import Settings
import matplotlib.pyplot as plt
import boto3
from ThyroidPrediction.secrets.secret import AWS_S3_ACCESS_KEY_ID, AWS_S3_SECRET_ACCESS_KEY, AWS_REGION, AWS_BUCKET_NAME, AWS_FOLDER_NAME


class DataIngestion:
    def __init__(self, data_ingestion_config: BaseDataIngestionConfig):
        try:
            logging.info(f"{'='*20} DATA INGESTION LOG STARTED.{'='*20}")

            # self.data_ingestion_config = data_ingestion_config
            self.current_time_stamp = CURRENT_TIME_STAMP
            self.base_data_ingestion_config = data_ingestion_config
            self.base_dataset_path = r"ThyroidPrediction\dataset_base"

            self.dataset_s3_bucket_download_dir = os.path.join("ThyroidPrediction", "s3_bucket", self.current_time_stamp)

            self.profiling_dir_part_1 = os.path.join("ThyroidPrediction\\artifact", "Profiling", self.current_time_stamp, "Part_1")
            os.makedirs(self.profiling_dir_part_1, exist_ok=True)

            self.profiling_dir_part_2 = os.path.join("ThyroidPrediction\\artifact","Profiling",self.current_time_stamp,"Part_2")
            os.makedirs(self.profiling_dir_part_2, exist_ok=True)

        except Exception as e:
            raise ThyroidException(e, sys) from e

    def download_s3_bucket_data(self):
        try:
            logging.info(f"{'='*20} S3 BUCKET DATA DOWNLOAD LOG STARTED.{'='*20}")

            # import boto3
            # import os

            # Set up the S3 client

            logging.info(f" Creating s3 client using boto3")
            s3 = boto3.client("s3",
                              aws_access_key_id=AWS_S3_ACCESS_KEY_ID,
                              aws_secret_access_key=AWS_S3_SECRET_ACCESS_KEY,
                              region_name=AWS_REGION
                              )

            # Define the bucket and folder name
            bucket_name = AWS_BUCKET_NAME
            folder_name = AWS_FOLDER_NAME

            # List all the objects in the folder

            logging.info(f"Getting list of objcts in s3 bucket")
            objects = s3.list_objects(Bucket=bucket_name, Prefix=folder_name)["Contents"]

            # Loop through each object and download it
            for obj in objects:
                # Create the local file path
                logging.info(f"{ obj['Key'] }")

                print("======= OBJECT KEY IN S3 BUCKET ===" * 2)
                print(obj["Key"])
                print("====================================" * 2)

                head, tail = os.path.split(obj["Key"])

                s3_download_dir = os.path.join(self.dataset_s3_bucket_download_dir, head)
                os.makedirs(s3_download_dir, exist_ok=True)

                s3_download_file_path = os.path.join(s3_download_dir, tail)

                # Download the file
                logging.info(f"Downloading object: [ {tail} ] in path [ {s3_download_file_path} ]")

                s3.download_file(bucket_name, obj["Key"], s3_download_file_path)

        except Exception as e:
            raise ThyroidException(e, sys) from e

    def get_base_data(self):
        try:
            pd.set_option("display.max_columns", None)

            # Path to the top-level directory
            # dir_path = "ThyroidPrediction/dataset_base/Raw_Dataset"
            try:
                s3_bucket_download_dir, _ = os.path.split(self.dataset_s3_bucket_download_dir)
                # s3_bucket_download_dir = s3_bucket_download_dir

                print("============ S3 Bucket Download Dir=====" * 3)
                print(s3_bucket_download_dir)
            except:
                pass

            def is_dir_empty(path):

                logging.info(f"Checking If S3 Bucket Local Storage [ {path} ] is empty or not")
                if not os.path.isdir(path):
                    return False
                
                if len(os.listdir(path)) != 0:
                        return False
                
                for dirpath, dirnames, filenames in os.walk(path):
                    if dirnames or filenames:
                        return False                                
                return True
                

            print(f"=========== s3 bucket is empty: {is_dir_empty(s3_bucket_download_dir)}")
            print("====================================================================")

            logging.info(f"S3 Local Bucket: [ {s3_bucket_download_dir} ] \n EMPTY: {is_dir_empty(s3_bucket_download_dir)}")
            
            try:
                if not is_dir_empty(s3_bucket_download_dir):
                    
                    print("DIR NOT EMPTY")

                    ordered_dir_list = natsort.natsorted(os.listdir(s3_bucket_download_dir))
                    most_recent_dir = ordered_dir_list[-1]

                    raw_data_dir = self.base_data_ingestion_config.raw_data_dir
                    raw_data_dir_path = os.path.join(s3_bucket_download_dir, most_recent_dir, raw_data_dir)
                    
                    logging.info("=============== DATA FETCHING FROM S3 BUCKET ==================")
                    print("=============== DATA FETCHING FROM S3 BUCKET ==================")
                else:
                    logging.info("=============== DATA FETCHING FROM LOCAL_DATASET_BASE ==================")
                    print("=============== DATA FETCHING FROM LOCAL_DATASET_BASE  ==================")
                    dataset_base = self.base_dataset_path
                    raw_data_dir = self.base_data_ingestion_config.raw_data_dir
                    raw_data_dir_path = os.path.join(dataset_base, raw_data_dir)                    
            except:
                logging.info("=============== DATA FETCHING FROM LOCAL_DATASET_BASE ==================")
                print("=============== DATA FETCHING FROM LOCAL_DATASET_BASE  ==================")
                dataset_base = self.base_dataset_path
                raw_data_dir = self.base_data_ingestion_config.raw_data_dir
                raw_data_dir_path = os.path.join(dataset_base, raw_data_dir)

            # dataset_base = self.base_dataset_path
            # raw_data_dir = self.base_data_ingestion_config.raw_data_dir
            # raw_data_dir_path = os.path.join(dataset_base, raw_data_dir)

            print("=== raw_data_dir_path ==" * 20)
            print("\n\n", raw_data_dir_path)
            print("==" * 20)

            # os.makedirs(raw_data_dir, exist_ok=True)

            logging.info(f"raw_data_dir_path: [ {raw_data_dir_path} ]")
            csv_files = []
            columns_list = [ "age", 
                            "sex",
                            "on_thyroxine",
                            "query_on_thyroxine",
                            "on_antithyroid_medication",
                            "sick",
                            "pregnant",
                            "thyroid_surgery",
                            "I131_treatment",
                            "query_hypothyroid",
                            "query_hyperthyroid",
                            "lithium",
                            "goitre",
                            "tumor",
                            "hypopituitary",
                            "psych",
                            "TSH_measured",
                            "TSH",
                            "T3_measured",
                            "T3",
                            "TT4_measured",
                            "TT4",
                            "T4U_measured",
                            "T4U",
                            "FTI_measured",
                            "FTI",
                            "TBG_measured",
                            "TBG",
                            "referral_source",
                            "Class"]

            logging.info(f"{'='*20} READING BASE DATASET {'='*20} \n\n Walking Through All Dirs In [ {raw_data_dir_path} ] for all .data and .test files")

            # Traverse the directory structure recursively
            for root, dirs, files in os.walk(raw_data_dir_path):
                for file in files:
                    # print(files)
                    # Check if the file is a CSV file
                    if file.endswith(".data") or file.endswith(".test"):
                        file_path = os.path.join(root, file)
                        # print(file_path)

                        # Read the CSV file into a pandas DataFrame
                        df = pd.read_csv(file_path, header=None)
                        df.columns = columns_list
                        
                        print("Unique value per file",
                              file_path,
                              df["hypopituitary"].unique()
                              )
                        
                        # print(file_path, df.columns)
                        csv_files.append(df)
                        

            logging.info(f"Total [ {len(csv_files)} ] files read in all dirs in [ {raw_data_dir_path} ]")

            print("Number of csv files", len(csv_files))

            df_combined = pd.DataFrame()
            for i in range(len(csv_files)):
                # print(len(csv_files[i].columns))
                # df_name = f"df_{i}"
                df_next = csv_files[i]
                df_combined = pd.concat([df_combined, df_next], axis=0)

                print("Unique value per file", file_path, df_combined["hypopituitary"].unique())

            df_combined.columns = columns_list

            # print("Hypopituitory unique values before cleaning",df_combined['hypopituitary'].unique())
            ###############################################################################################

            logging.info(f"Handling Duplicats and Missing Values")

            df_combined.drop_duplicates(inplace=True)
            df_combined["Class"].replace(to_replace=r"[.|0-9]", value="", regex=True, inplace=True)
            df_combined = df_combined.drop(["TSH_measured", "T3_measured", "TT4_measured", "T4U_measured", "FTI_measured","TBG_measured", "query_on_thyroxine"], axis=1)

            print("Missing Value('?') Count After Replaing '?' symbols with NaN Revealing:\n")
            for column in df_combined.columns:
                missing_count = df_combined[column][df_combined[column] == "?"].count()

                if missing_count != 0:
                    print(column, missing_count)
                    df_combined[column] = df_combined[column].replace("?", np.nan)

            # The entir TBG column have missing values so dropping it
            df_combined = df_combined.drop(["TBG"], axis=1)

            columns_float = ["age", "TSH", "T3", "TT4", "T4U", "FTI"]

            for column in columns_float:
                df_combined[column] = df_combined[column].astype(float)

            # print("Hypopituitory Uniqu valus",df_combined["hypopituitary"].unique())

            # processed_dataset_dir = os.path.join(self.base_dataset_path,"Processed_Dataset","Cleaned_Data")
            # os.makedirs(processed_dataset_dir, exist_ok=True)

            ################## EXPORT PROCESSED FILE ################

            logging.info(f"Exporting Combined and semi-Cleaned Data to path: [{os.path.join(self.base_dataset_path,self.base_data_ingestion_config.processed_data_dir,self.base_data_ingestion_config.cleaned_data_dir)}]")

            processed_data_dir = os.path.join(self.base_dataset_path,
                                              self.base_data_ingestion_config.processed_data_dir,
                                              self.base_data_ingestion_config.cleaned_data_dir,
                                              "raw_data_merged")
            
            os.makedirs(processed_data_dir, exist_ok=True)

            # processed_data_file_path = os.path.join(processed_dataset_dir,"df_combined_cleaned.csv")
            # df_combined.to_csv(processed_data_file_path,index=False)

            processed_data_file_path = os.path.join(processed_data_dir, "df_combined.csv")
            df_combined.to_csv(processed_data_file_path, index=False)

            logging.info("Merged Data Export Done!")
            logging.info(f"Processed data file path: [ {processed_data_file_path} ]")

            print("====== processed_data_file_path====" * 5)
            print(processed_data_file_path)
            print("====================================" * 5)

            return df_combined

        except Exception as e:
            raise ThyroidException(e, sys)

    def get_data_transformer_object(self):
        try:
            # processesd_data_dir_path = self.processed_data_dir_path
            # cleaned_data_file_path = os.path.join(self.cleaned_data_dir,"df_combined_cleaned.csv")

            # print("===== Cleand Data File Path ======"*20)
            # print("\n\n",cleaned_data_file_path)

            # df_combined = pd.read_csv(cleaned_data_file_path)

            df_combined = self.get_base_data()

            #######################################    MISSING VALUE IMPUATION    ##########################################################
            logging.info(f"Missing Value Imputation and Handling Categorical Variables")

            df_combined["sex"] = SimpleImputer(missing_values=np.nan, strategy="most_frequent").fit_transform(df_combined[["sex"]].values)
            df_combined["age"] = SimpleImputer(missing_values=np.nan, strategy="most_frequent").fit_transform(df_combined[["age"]].values)

            df_combined["TSH"] = SimpleImputer(missing_values=np.nan, strategy="median").fit_transform(df_combined[["TSH"]].values)
            df_combined["T3"] = SimpleImputer(missing_values=np.nan, strategy="median").fit_transform(df_combined[["T3"]].values)
            df_combined["TT4"] = SimpleImputer(missing_values=np.nan, strategy="median").fit_transform(df_combined[["TT4"]].values)
            df_combined["T4U"] = SimpleImputer(missing_values=np.nan, strategy="median").fit_transform(df_combined[["T4U"]].values)
            df_combined["FTI"] = SimpleImputer(missing_values=np.nan, strategy="median").fit_transform(df_combined[["FTI"]].values)

            ###################### HANDLINNG CATEGOICAL VARIABLES ###########################

            df_combined_plot = df_combined.copy()

            columns_list = df_combined.columns.to_list()

            for feature in columns_list:
                if len(df_combined[feature].unique()) <= 3:
                    # print(df_combined[feature].unique() )
                    value1 = df_combined[feature].unique()[0]
                    value2 = df_combined[feature].unique()[1]

                    df_combined[feature] = df_combined[feature].map({f"{value1}": 0, f"{value2}": 1})

                    print(feature, df_combined[feature].unique())

            df_combined = pd.get_dummies(data=df_combined, columns=["referral_source"], drop_first=True)
            df_combined["Class_encoded"] = LabelEncoder().fit_transform(df_combined["Class"])

            return df_combined, df_combined_plot
        except Exception as e:
            raise ThyroidException(e, sys) from e

    def outliers_handling(self):
        try:
            ############################## OUTLIERS HANDLING ###############################

            df_combined, df_combined_plot = self.get_data_transformer_object()

            logging.info(f"Handling Outliers")

            def outliers_fence(col):
                Q1 = df_combined[col].quantile(q=0.25)
                Q3 = df_combined[col].quantile(q=0.75)
                IQR = Q3 - Q1

                lower_fence = Q1 - 1.5 * IQR
                upper_fence = Q3 + 1.5 * IQR
                return lower_fence, upper_fence

            lower_fence1, upper_fence1 = outliers_fence(col="TSH")
            lower_fence2, upper_fence2 = outliers_fence(col="T3")
            lower_fence3, upper_fence3 = outliers_fence(col="TT4")
            lower_fence4, upper_fence4 = outliers_fence(col="T4U")
            lower_fence5, upper_fence5 = outliers_fence(col="FTI")

            # Winsorize the data just replace outliers with corresponding fence

            df_combined["TSH"] = np.where(
                df_combined["TSH"] < lower_fence1, lower_fence1, df_combined["TSH"]
            )
            df_combined["TSH"] = np.where(
                df_combined["TSH"] > upper_fence1, upper_fence1, df_combined["TSH"]
            )

            df_combined["T3"] = np.where(
                df_combined["T3"] < lower_fence2, lower_fence2, df_combined["T3"]
            )
            df_combined["T3"] = np.where(
                df_combined["T3"] > upper_fence2, upper_fence2, df_combined["T3"]
            )

            df_combined["TT4"] = np.where(
                df_combined["TT4"] < lower_fence3, lower_fence3, df_combined["TT4"]
            )
            df_combined["TT4"] = np.where(
                df_combined["TT4"] > upper_fence3, upper_fence3, df_combined["TT4"]
            )

            df_combined["T4U"] = np.where(
                df_combined["T4U"] < lower_fence4, lower_fence4, df_combined["T4U"]
            )
            df_combined["T4U"] = np.where(
                df_combined["T4U"] > upper_fence4, upper_fence4, df_combined["T4U"]
            )

            df_combined["FTI"] = np.where(
                df_combined["FTI"] < lower_fence5, lower_fence5, df_combined["FTI"]
            )
            df_combined["FTI"] = np.where(
                df_combined["FTI"] > upper_fence5, upper_fence5, df_combined["FTI"]
            )

            logging.info(f"Outliers Handling DOne")

            return df_combined, df_combined_plot

        except Exception as e:
            raise ThyroidException(e, sys) from e

    def get_target_by_major_class(self):
        try:
            ##################################### MAJOR CLASS CREATION   ############################################################

            df_combined_class_labels, df_combined_plot = self.outliers_handling()
            df_combined_class_labels["Class_label"] = df_combined_plot["Class"]

            logging.info(f"Grouping Class lablels into major categories")

            df = df_combined_class_labels
            # Define the major class conditions
            conditions = [
                df["Class_label"].isin(
                    [
                        "compensated hypothyroid",
                        "primary hypothyroid",
                        "secondary hypothyroid",
                    ]
                ),
                df["Class_label"].isin(["hyperthyroid", "T toxic", "secondary toxic"]),
                df["Class_label"].isin(
                    ["replacement therapy", "underreplacement", "overreplacement"]
                ),
                df["Class_label"].isin(["goitre"]),
                df["Class_label"].isin(
                    ["increased binding protein", "decreased binding protein"]
                ),
                df["Class_label"].isin(["sick"]),
                df["Class_label"].isin(["discordant"]),
            ]

            # Define the major class labels
            class_labels = [
                "hypothyroid",
                "hyperthyroid",
                "replacement therapy",
                "goitre",
                "binding protein",
                "sick",
                "discordant",
            ]

            # Add the major class column to the dataframe based on the conditions
            df["major_class"] = np.select(conditions, class_labels, default="negative")
            df.drop("Class_label", axis=1, inplace=True)

            df_combined_grouped = df.copy()

            df_combined_grouped["major_class_encoded"] = LabelEncoder().fit_transform(
                df_combined_grouped["major_class"]
            )

            transformed_data_dir = os.path.join(
                self.base_dataset_path,
                self.base_data_ingestion_config.processed_data_dir,
                self.base_data_ingestion_config.cleaned_data_dir,
                "processed_data",
                "Cleaned_transformed",
            )
            os.makedirs(transformed_data_dir, exist_ok=True)

            transformed_data_file_path = os.path.join(
                transformed_data_dir, "df_transformed_major_class.csv"
            )

            logging.info(
                f"Exporting Grouped adn Cleaned Data to path: [ {transformed_data_file_path} ]"
            )

            df_combined_grouped.to_csv(transformed_data_file_path, index=False)

            return df_combined_grouped

        except Exception as e:
            raise ThyroidException(e, sys) from e

    def profiling_report(self):
        def get_missing_value_fig():
            try:
                df_combined = self.get_base_data()

                plt.figure(figsize=(14, 6), layout="tight")

                # plt.subplot(1,2,1)
                # sns.heatmap(df_combined_orig.isnull(), cbar=False, cmap="viridis", yticklabels=False)
                # plt.title('Missing Values Before', fontdict={'fontsize':20},pad=12)

                # plt.subplot(1,2,2)
                sns.heatmap(
                    df_combined.isnull(), cbar=False, cmap="viridis", yticklabels=False
                )
                plt.title("Revealed Missing Values", fontdict={"fontsize": 20}, pad=12)
                # plt.show()

                missing_value_fig_path = os.path.join(
                    self.profiling_dir_part_1, "1_missing_values.svg"
                )
                plt.savefig(missing_value_fig_path, dpi=300, bbox_inches="tight")

            except Exception as e:
                raise ThyroidException(e, sys) from e

        def get_outlier_before_fig():
            try:
                df_combined_orig = self.get_base_data()
                df_combined_orig[df_combined_orig["TSH"] != "?"][["TSH"]].astype(float)

                # Define the number of rows and columns for the subplot grid
                num_rows = 2
                num_cols = 3

                # Create a subplot grid with the specified number of rows and columns
                fig = sp.make_subplots(
                    rows=num_rows,
                    cols=num_cols,
                    subplot_titles=["age", "TSH", "T3", "TT4", "T4U", "FTI"],
                )

                # Loop through each column in the dataframe and add a box plot to the subplot grid
                for idx, col_name in enumerate(
                    ["age", "TSH", "T3", "TT4", "T4U", "FTI"]
                ):
                    row_num = (idx // num_cols) + 1
                    col_num = (idx % num_cols) + 1

                    fig.add_trace(
                        px.box(
                            df_combined_orig[df_combined_orig[col_name] != "?"][
                                [col_name]
                            ].astype(float)
                        ).data[0],
                        row=row_num,
                        col=col_num,
                    )

                    # Set the title of the subplot grid
                    fig.update_layout(
                        height=500,
                        width=1100,
                        title="Before Handling Outliers",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                    )
                    fig.update_traces(marker_color="green")
                # Show the plot

                ###############################
                fig.update_yaxes(showline=False, showgrid=False)
                fig.update_xaxes(showline=False, showgrid=False)
                # fig.show()
                ##########################################
                outlier_fig_before_path = os.path.join(
                    self.profiling_dir_part_1, "2_outliers_before.html"
                )
                pio.write_html(fig, file=outlier_fig_before_path, auto_play=False)

            except Exception as e:
                raise ThyroidException(e, sys) from e

        def get_outlier_after_outlier_handling():
            try:
                df_combined, _ = self.outliers_handling()
                # Define the number of rows and columns for the subplot grid
                num_rows = 2
                num_cols = 3

                # Create a subplot grid with the specified number of rows and columns
                fig = sp.make_subplots(
                    rows=num_rows,
                    cols=num_cols,
                    subplot_titles=["age", "TSH", "T3", "TT4", "T4U", "FTI"],
                )

                # Loop through each column in the dataframe and add a box plot to the subplot grid
                for idx, col_name in enumerate(
                    ["age", "TSH", "T3", "TT4", "T4U", "FTI"]
                ):
                    row_num = (idx // num_cols) + 1
                    col_num = (idx % num_cols) + 1
                    fig.add_trace(
                        px.box(df_combined[col_name]).data[0],
                        row=row_num,
                        col=col_num,
                    )

                    # Set the title of the subplot grid
                    fig.update_layout(
                        height=500,
                        width=1100,
                        title="After Handling Outliers",
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                    )
                    fig.update_traces(marker_color="green")
                # Show the plot

                ###############################
                fig.update_yaxes(showline=False, showgrid=False)
                fig.update_xaxes(showline=False, showgrid=False)
                # fig.show()
                ##########################################
                outliers_fig_after_path = os.path.join(
                    self.profiling_dir_part_1, "3_outliers_after.html"
                )
                pio.write_html(fig, file=outliers_fig_after_path, auto_play=False)
            except Exception as e:
                raise ThyroidException(e, sys) from e

        def get_class_pecentage_share():
            try:
                df_combined, _ = self.outliers_handling()
                fig = px.pie(df_combined, names="Class", hole=0.3)

                fig.update_layout(
                    title="Percentage Share of Class Labels",
                    height=800,
                    width=800,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    # annotations = [dict(text="Class".title(), showarrow=False)],
                    margin_autoexpand=True,
                    legend=dict(
                        yanchor="bottom",
                        y=-0.5,
                        xanchor="center",
                        x=0.5,
                        orientation="h",
                    ),
                    autosize=True,
                )
                # fig.show()
                class_percentage_share_path = os.path.join(
                    self.profiling_dir_part_1, "4_class_share.html"
                )
                pio.write_html(fig, file=class_percentage_share_path, auto_play=False)

            except Exception as e:
                raise ThyroidException(e, sys) from e

        def get_major_class_pecentage_share():
            try:
                df_combined = self.get_target_by_major_class()

                fig = px.pie(df_combined, names="major_class", hole=0.3)

                fig.update_layout(
                    title="Percentage Share of Major Class Labels",
                    height=800,
                    width=800,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    # annotations = [dict(text="Class".title(), showarrow=False)],
                    margin_autoexpand=True,
                    legend=dict(
                        yanchor="bottom",
                        y=-0.5,
                        xanchor="center",
                        x=0.5,
                        orientation="h",
                    ),
                    autosize=True,
                )
                # fig.show()
                major_class_percentage_share_path = os.path.join(
                    self.profiling_dir_part_1, "5_major_class_share.html"
                )
                pio.write_html(
                    fig, file=major_class_percentage_share_path, auto_play=False
                )

            except Exception as e:
                raise ThyroidException(e, sys) from e

        def get_gender_share():
            try:
                _, df_combined_plot = self.outliers_handling()

                fig = px.histogram(
                    df_combined_plot, x="sex", color="sex", histfunc="count"
                )
                fig.update_layout(
                    height=400,
                    width=500,
                    title="Count Plot For Gender",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                # fig.show()

                gender_share_path = os.path.join(
                    self.profiling_dir_part_1, "6_gender_share.html"
                )
                pio.write_html(fig, file=gender_share_path, auto_play=False)

            except Exception as e:
                raise ThyroidException(e, sys) from e

        def get_comparative_impact():
            try:
                _, df_combined_plot = self.outliers_handling()

                print("=" * 20)
                print(df_combined_plot.columns)

                col1 = [
                    "sex",
                    "on_thyroxine",
                    "on_antithyroid_medication",
                    "pregnant",
                    "thyroid_surgery",
                ]
                col2 = ["I131_treatment", "lithium", "goitre", "tumor", "hypopituitary"]

                # df = df_combined[cols]

                for col in df_combined_plot[col1].columns:
                    for row in df_combined_plot[col2].columns:
                        if col != row:
                            fig = px.pie(
                                df_combined_plot,
                                names="Class",
                                facet_col=col,
                                facet_row=row,
                                hole=0.3,
                            )

                            fig.update_layout(
                                height=1000,
                                width=1000,
                                paper_bgcolor="rgba(0,0,0,0)",
                                plot_bgcolor="rgba(0,0,0,0)",
                                # annotations = [dict(text="Class".title(), showarrow=False)],
                                margin_autoexpand=True,
                                legend=dict(
                                    yanchor="bottom",
                                    y=-0.5,
                                    xanchor="center",
                                    x=0.5,
                                    orientation="h",
                                ),
                                autosize=True,
                            )
                            # fig.show()
                            # comparative_impact_path = os.path.join(self.profiling_dir,"7_comparative_impact.html")
                            # pio.write_html(fig,file = comparative_impact_path, auto_play=False)

                            relational_separate_fig_dir = os.path.join(
                                self.profiling_dir_part_2, "relational"
                            )
                            os.makedirs(relational_separate_fig_dir, exist_ok=True)

                            relational_separate_fig_path = os.path.join(
                                relational_separate_fig_dir, f"{col}_vs_{row}.html"
                            )

                            pio.write_html(
                                fig,
                                file=relational_separate_fig_path,
                                auto_play=False,
                                full_html=False,
                            )

                ###########################################################

                # Set the path to the directory containing the sufolders or HTML files
                dir_path = relational_separate_fig_dir

                # Create a string to hold the HTML code
                html_code = ""

                # Loop through all directories and files in the directory tree

                for root, dirs, files in os.walk(dir_path):
                    files = natsort.natsorted(files)

                    print(files)

                    for file in files:
                        # Check if the file is an HTML file
                        ###########################################################

                        if file.endswith(".html") or file.endswith(".svg"):
                            # file_list.append(file)
                            # Read the contents of the file

                            with open(
                                os.path.join(root, file), "r", encoding="utf-8"
                            ) as f:
                                file_contents = f.read()

                            # Add the contents of the file to the HTML code string
                            html_code += file_contents
                            ##########################################

                # Write the HTML code to a new file
                with open(
                    os.path.join(self.profiling_dir_part_2, "7_relational_mrged.html"),
                    "a",
                    encoding="utf-8",
                ) as f:
                    # with open('comparative_impact.html', 'a') as f:

                    f.write(html_code)

                shutil.rmtree(dir_path)

            except Exception as e:
                raise ThyroidException(e, sys) from e

        def get_kde_plot():
            try:
                _, df_combined_plot = self.outliers_handling()

                col1 = [
                    "sex",
                    "on_thyroxine",
                    "on_antithyroid_medication",
                    "sick",
                    "pregnant",
                    "thyroid_surgery",
                    "I131_treatment",
                    "query_hypothyroid",
                    "query_hyperthyroid",
                    "lithium",
                    "goitre",
                    "tumor",
                    "hypopituitary",
                    "psych",
                    "referral_source",
                    "Class",
                ]
                # col2 = [ 'I131_treatment','query_hypothyroid','query_hyperthyroid','lithium','goitre','tumor','hypopituitary','psych']

                # generate a sample dataframe
                df = df_combined_plot.drop(col1, axis=1)

                # create figure and axes objects
                fig, axes = plt.subplots(
                    nrows=5, ncols=3, squeeze=True, figsize=(12, 12)
                )

                # flatten the axes array for easy indexing
                axes = axes.flatten()

                # loop through each column and plot the kde on a separate axis
                for i, col in enumerate(df.columns):
                    sns.kdeplot(df[col], ax=axes[i], fill=True)
                    axes[i].set_title(col)

                # remove any unused axes and add a main title
                for i in range(len(df.columns), len(axes)):
                    fig.delaxes(axes[i])
                fig.suptitle("KDE Plot for All Independent Features", fontsize=14)

                # adjust the spacing between the subplots and show the figure
                fig.tight_layout(pad=2)
                # plt.show()

                #############################################
                kde_plot_path = os.path.join(
                    self.profiling_dir_part_1, "8_kde_plot.svg"
                )
                fig.figure.savefig(kde_plot_path, transparent=True, dpi=300)

            except Exception as e:
                raise ThyroidException(e, sys) from e

        def get_yDataprofile():
            try:
                _, df = self.outliers_handling()
                df_major_class = self.get_target_by_major_class()
                df["major_class"] = df_major_class["major_class"]
                yDataprofile = ProfileReport(
                    df=df,
                    explorative=True,
                    infer_dtypes=True,
                    orange_mode=True,
                    dark_mode=True,
                    tsmode=False,
                    plot={"dpi": 200, "image_format": "svg"},
                    title="Profiling Report",
                    progress_bar=True,
                    html={
                        "style": {"full_width": True, "primary_color": "#000000"},
                        "minify": True,
                    },
                    correlations={
                        "pearson": {"calculate": True},
                        "spearman": {"calculate": True},
                        "kendall": {"calculate": True},
                        "phi_k": {"calculate": True},
                    },
                    missing_diagrams={"bar": True, "matrix": True, "hatmap": False},
                    interactions=None,
                )

                yDataprofile_path = os.path.join(
                    self.profiling_dir_part_1, "0_yDataprofile.html"
                )
                yDataprofile.to_file(yDataprofile_path, silent=True)

            except Exception as e:
                raise ThyroidException(e, sys) from e

        def get_profile_report_1():
            try:
                # Set the path to the directory containing the sufolders or HTML files
                dir_path = self.profiling_dir_part_1

                # Create a string to hold the HTML code
                html_code = ""

                # Loop through all directories and files in the directory tree

                for root, dirs, files in os.walk(dir_path):
                    files = natsort.natsorted(files)

                    print(files)

                    for file in files:
                        # Check if the file is an HTML file
                        ###########################################################

                        if file.endswith(".html") or file.endswith(".svg"):
                            # file_list.append(file)
                            # Read the contents of the file

                            with open(
                                os.path.join(root, file), "r", encoding="utf-8"
                            ) as f:
                                file_contents = f.read()

                            # Add the contents of the file to the HTML code string
                            html_code += file_contents
                            ##########################################

                # Write the HTML code to a new file
                with open(
                    os.path.join(self.profiling_dir_part_1, "ProfileReport_1.html"),
                    "a",
                    encoding="utf-8",
                ) as f:
                    # with open('comparative_impact.html', 'a') as f:

                    f.write(html_code)
                ######################### CLEARING ALL FILES Other THAN ONE SPECIFID FILE ###########################
                # shutil.rmtree(dir_path)    # Clear all the files and folder irrespective to that if they contain data or not

                dir_path = self.profiling_dir_part_1
                except_file = "ProfileReport_1.html"

                for file_name in os.listdir(dir_path):
                    if file_name != except_file:
                        os.remove(os.path.join(dir_path, file_name))

                ####################### COPYING PROFILE_REPORT_1 TO TEMPLATE #######################################

                src_dir = os.path.join("ThyroidPrediction\\artifact", "Profiling")
                dest_dir = os.path.join("templates")

                # Get the list of folders in the source directory
                folders = natsort.natsorted(os.listdir(src_dir))

                # Get the most recent folder
                most_recent_folder = folders[-1]

                # Construct the path to the most recent folder
                most_recent_folder_path = os.path.join(
                    src_dir, most_recent_folder, "Part_1"
                )

                # Get the list of files in the most recent folder
                files = natsort.natsorted(os.listdir(most_recent_folder_path))

                # Get the most recent file
                most_recent_file = files[-1]

                # Construct the path to the most recent file
                most_recent_file_path = os.path.join(
                    most_recent_folder_path, most_recent_file
                )

                # Copy the most recent file to the destination directory
                shutil.copy(most_recent_file_path, dest_dir)

            except Exception as e:
                raise ThyroidException(e, sys) from e

        def get_profile_report_2():
            try:
                # Set the path to the directory containing the sufolders or HTML files
                dir_path = self.profiling_dir_part_2

                # Create a string to hold the HTML code
                html_code = ""

                # Loop through all directories and files in the directory tree

                for root, dirs, files in os.walk(dir_path):
                    files = natsort.natsorted(files)

                    print(files)

                    for file in files:
                        # Check if the file is an HTML file
                        ###########################################################

                        if file.endswith(".html") or file.endswith(".svg"):
                            # file_list.append(file)
                            # Read the contents of the file

                            with open(
                                os.path.join(root, file), "r", encoding="utf-8"
                            ) as f:
                                file_contents = f.read()

                            # Add the contents of the file to the HTML code string
                            html_code += file_contents
                            ##########################################

                # Write the HTML code to a new file
                with open(
                    os.path.join(self.profiling_dir_part_2, "ProfileReport_2.html"),
                    "a",
                    encoding="utf-8",
                ) as f:
                    # with open('comparative_impact.html', 'a') as f:

                    f.write(html_code)
                ######################### CLEARING ALL FILES Other THAN ONE SPECIFID FILE ###########################
                # shutil.rmtree(dir_path)    # Clear all the files and folder irrespective to that if they contain data or not

                dir_path = self.profiling_dir_part_2
                except_file = "ProfileReport_2.html"

                for file_name in os.listdir(dir_path):
                    if file_name != except_file:
                        os.remove(os.path.join(dir_path, file_name))

                ####################### COPYING PROFILE_REPORT_2 TO TEMPLATE #######################################

                src_dir = os.path.join("ThyroidPrediction\\artifact", "Profiling")
                dest_dir = os.path.join("templates")

                # Get the list of folders in the source directory
                folders = natsort.natsorted(os.listdir(src_dir))

                # Get the most recent folder
                most_recent_folder = folders[-1]

                # Construct the path to the most recent folder
                most_recent_folder_path = os.path.join(
                    src_dir, most_recent_folder, "Part_2"
                )

                # Get the list of files in the most recent folder
                files = natsort.natsorted(os.listdir(most_recent_folder_path))

                # Get the most recent file
                most_recent_file = files[-1]

                # Construct the path to the most recent file
                most_recent_file_path = os.path.join(
                    most_recent_folder_path, most_recent_file
                )

                # Copy the most recent file to the destination directory
                shutil.copy(most_recent_file_path, dest_dir)

            except Exception as e:
                raise ThyroidException(e, sys) from e

        ############ Calling Sub Functions  ############

        get_yDataprofile()
        get_missing_value_fig()
        get_outlier_before_fig()
        get_outlier_after_outlier_handling()
        get_class_pecentage_share()
        get_major_class_pecentage_share()
        get_gender_share()
        get_comparative_impact()
        get_kde_plot()
        get_profile_report_1()
        get_profile_report_2()

    def split_data(self):
        try:
            # file_path = os.path.join(self.base_dataset_path,"Resampled_Dataset","ResampleData_major.csv")

            # logging.info(f"Reading CSV file for Base dataset [{file_path}]")
            # df = pd.read_csv(file_path)

            # logging.info(f"Encoding Major_Class Categories")
            # df["major_class_endcoded"] = LabelEncoder().fit_transform(df["major_class"])

            # X = df.drop(['major_class', 'major_class_endcoded'],axis=1)
            # y = df.drop(['major_class'],axis=1)['major_class_endcoded']

            # logging.info(f"Spliting Dataset into train and test set")
            # X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, shuffle=True, stratify= y, random_state=2023 )

            df_combined_grouped = self.get_target_by_major_class()

            logging.info(f"Splitting Data into train and test set")

            #################################   DATA SPLIT BEFORE RESAMPLING  #################################################
            X = df_combined_grouped.drop(
                ["Class", "Class_encoded", "major_class", "major_class_encoded"], axis=1
            )
            y = df_combined_grouped["major_class_encoded"]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, shuffle=True, stratify=y, random_state=2023
            )

            start_train_set = None
            start_test_set = None
            start_train_set = pd.concat([X_train, y_train], axis=1)
            start_test_set = pd.concat([X_test, y_test], axis=1)

            print(start_train_set.columns)

            split_data_dir = os.path.join(
                self.base_dataset_path,
                self.base_data_ingestion_config.processed_data_dir,
                self.base_data_ingestion_config.cleaned_data_dir,
                "processed_data",
                "split_data",
            )
            train_file_dir = os.path.join(split_data_dir, "train_set")
            test_file_dir = os.path.join(split_data_dir, "test_set")

            print("==== train_file_dir ======" * 20)
            print(train_file_dir)
            print("==========================" * 20)

            if start_train_set is not None:
                os.makedirs(train_file_dir, exist_ok=True)

                logging.info(f"Exporting training data to file:[{train_file_dir}]")

                train_file_path = os.path.join(train_file_dir, "train.csv")
                start_train_set.to_csv(train_file_path, index=False)

            if start_test_set is not None:
                os.makedirs(test_file_dir, exist_ok=True)
                logging.info(f"Exporting test data to file:[{test_file_dir}]")
                test_file_path = os.path.join(test_file_dir, "test.csv")

                start_test_set.to_csv(test_file_path, index=False)

            logging.info(f"Data Split Done!")

            #################################   Returning DataIngestionArtifact#################################################
            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path=train_file_path,
                test_file_path=test_file_path,
                is_ingested=True,
                message=f"DataIngestion Completed Successfully",
            )

            logging.info(f"Data Ingestion Artifact:[{data_ingestion_artifact}]")
            return data_ingestion_artifact

        except Exception as e:
            raise ThyroidException(e, sys)

    def initiate_data_ingestion(self):
        try:
            
            try:
                self.download_s3_bucket_data()
            except:
                pass

            self.get_base_data()
            self.get_data_transformer_object()
            self.outliers_handling()
            self.get_target_by_major_class()

            self.profiling_report()

            return self.split_data()

        except Exception as e:
            raise ThyroidException(e, sys) from e

    def __del__(self):
        logging.info(f"{'='*20} Ingestion log completed {'='*20}\n\n")
