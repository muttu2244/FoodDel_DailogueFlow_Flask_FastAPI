from src.constants import *
from src.logger import logging
from src.exception import CustomException
import os, sys
from src.config.configuration import *
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer
from src.components.data_ingestion import DataIngestion

class Train:
    def __init__(self):
        self.c = 0
        print(f"**********{self.c}***************")
    def main(self):
        obj = DataIngestion()
        print(f"DataIng Obje {obj}")
        train_data,test_data = obj.initiate_data_ingestion()
        print("AFTER DATA INGESTION")
        data_transformation = DataTransformation()
        train_arr,test_arr = data_transformation.initiate_data_transformation(train_data,test_data)
        print(train_arr)
        print(test_arr)
        model_trainer = ModelTrainer()
        print(model_trainer.initiate_model_training(train_arr,test_arr))
        