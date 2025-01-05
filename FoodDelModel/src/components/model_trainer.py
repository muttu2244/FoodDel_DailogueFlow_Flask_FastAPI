from src.constants import *
from src.logger import logging
from src.exception import CustomException
import os, sys
from src.config.configuration import *
from dataclasses import dataclass
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.utils import evaluate_model
#from src.utils.save import save_obj

@dataclass
class ModelTrainerConfig:
    train_model_file_path = MODEL_FILE_PATH

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig

    def initiate_model_training(self, train_array, test_array):
        try:
            X_train, y_train, X_test, y_test = (train_array[:,:-1],train_array[:,-1],
                                                test_array[:,:-1],test_array[:,-1])
            
            
            
            models = {
                "XGBRegressor":XGBRegressor(),
                "DecisionTreeRegressor":DecisionTreeRegressor(),
                "GradientBoostingRegressor":GradientBoostingRegressor(),
                "RandomForestRegressor":RandomForestRegressor(),
                "SVR":SVR()
            }

            # model_report: dict = evaluate_model(X_train, y_train, X_test, y_test,models)
            # print(model_report)

            # best_model_score = max(sorted(model_report.values()))
            # best_model_name = list(model_report.keys())[
            #     list(model_report.values().index(best_model_score))
            # ]
            # best_model = models[best_model_name]
            model_report: dict = evaluate_model(X_train, y_train, X_test, y_test, models)
            #print(model_report)

            # Find the best model based on R² Score
            best_model_name = max(model_report, key=lambda k: model_report[k]["R2"])
            best_model = models[best_model_name]
            best_model_score = model_report[best_model_name]["R2"]

            
            from src.utils.save import save_obj
            save_obj(file_path=self.model_trainer_config.train_model_file_path, obj=best_model)

            # Print metrics for all models
            for model_name, metrics in model_report.items():
                print(f"\nModel: {model_name}")
                for metric, value in metrics.items():
                    print(f"{metric}: {value}")
                    
            print(f"Best Model found, Model name is : {best_model_name}, R2 Score is : {best_model_score}")
            logging.info(f"Best Model found, Model name is : {best_model_name}, R2 Score is : {best_model_score}")


        except Exception as e:
            #raise CustomException(e, sys)
            print(f"Error occurred: {e}")
                  