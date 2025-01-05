from src.logger import logging
from src.exception import CustomException
import os, sys
import pickle
#from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import numpy as np
from src.components.data_transformation import Feature_Engineering
import pandas as pd


    

"""
def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
            
            # Calculate evaluation metrics
            mse = mean_squared_error(y_test, y_test_pred)
            rmse = np.sqrt(mse)
            mape = mean_absolute_percentage_error(y_test, y_test_pred)
            r2 = r2_score(y_test, y_test_pred)
            
            # Store the metrics in a dictionary for each model
            report[model_name] = {
                "MSE": mse,
                "RMSE": rmse,
                "MAPE": mape,
                "R2": r2
            }
        return report
    except Exception as e:
        raise CustomException(e, sys)
"""     

def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        
        #feature_engineer = Feature_Engineering()  # Initialize the feature engineering class
        
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_test_pred = model.predict(X_test)
            
            # Calculate evaluation metrics
            mse = mean_squared_error(y_test, y_test_pred)
            rmse = np.sqrt(mse)
            mape = mean_absolute_percentage_error(y_test, y_test_pred)
            r2 = r2_score(y_test, y_test_pred)
            
            X_test_features = X_test[:, :-1]
            y_test = X_test[:, -1]  # The last column is the target variable 
            

            # converting ndarray to panda df
            X_test_df = pd.DataFrame(X_test_features, columns=['Delivery_person_ID','Delivery_person_Age', 'Delivery_person_Ratings', 'Restaurant_latitude',
                        'Restaurant_longitude', 'Delivery_location_latitude','Delivery_location_longitude', 'Order_Date', 'Time_Orderd',
                        'Time_Order_picked', 'Weatherconditions', 'Road_traffic_density','Vehicle_condition', 'Type_of_order', 'Type_of_vehicle',
                        'multiple_deliveries', 'Festival', 'City'])

            '''
            # Apply feature engineering for times and distance
            X_test_with_features = feature_engineer.calculate_times(
                X_test_df, 
                pickup_time='Time_Order_picked', ordered_time='Time_Orderd',
                restaurant_lat='Restaurant_latitude', restaurant_lon='Restaurant_longitude',
                customer_lat='Delivery_location_latitude', customer_lon='Delivery_location_longitude'
            )
            '''
            # Store the metrics and time-related features in a dictionary for each model
            report[model_name] = {
                "MSE": mse,
                "RMSE": rmse,
                "MAPE": mape,
                "R2": r2,
                #"FM_time": X_test_with_features['FM_time'].iloc[0],
                #"LM_time": X_test_with_features['LM_time'].iloc[0],
                #"O2A_time": X_test_with_features['O2A_time'].iloc[0],
                #"WT_time": X_test_with_features['WT_time'].iloc[0],
                #"O2R_time": X_test_with_features['O2R_time'].iloc[0],
                #"Distance": X_test_with_features['distance'].iloc[0],  # Add distance feature
                #"DeliveryTime": X_test_with_features['Delivery Time'].iloc[0]
            }
        return report
    except Exception as e:
        raise CustomException(e, sys)

def load_model(file_path):
    try:
        with open(file_path,'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logging.info("Exception occured while loading an object")
        raise CustomException(e,sys)