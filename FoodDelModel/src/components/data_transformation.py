from src.constants import *
from src.logger import logging
from src.exception import CustomException
import os, sys
from src.config.configuration import *
from dataclasses import dataclass
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder,OneHotEncoder
from sklearn.pipeline import Pipeline
from src.utils.save import save_obj
from src.config.configuration import PREPROCESING_OBJ_FILE,TRANSFORM_TRAIn_FILE_PATH,TRANSFORM_TEST_FILE_PATH,FEATURE_ENGG_OBJ_FILE_PATH




# FE
# Data transformation

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


"""
class Feature_Engineering(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.average_speed_kmh = 40  # Average speed of delivery executive in km/h
    
    
    def distance_numpy(self, df, lat1, lon1, lat2, lon2):
        '''
        Calculate distance using latitude and longitude from the DataFrame.
        '''
        try:
            p = np.pi / 180
            a = 0.5 - np.cos((df[lat2] - df[lat1]) * p) / 2 + np.cos(df[lat1] * p) * np.cos(df[lat2] * p) * (1 - np.cos((df[lon2] - df[lon1]) * p)) / 2
            df['distance'] = 12734 * np.arccos(np.sort(a))
        except Exception as e:
            raise ValueError("Error in distance calculation") from e
    
        
    def haversine(self, lat1, lon1, lat2, lon2):
        '''
        Calculate the Haversine distance between two points on the Earth.
        '''
        try:
            p = np.pi / 180  # Convert degrees to radians
            a = 0.5 - np.cos((lat2 - lat1) * p) / 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
            distance_km = 12734 * np.arccos(np.sqrt(a))  # Haversine formula in km
            return distance_km
        except Exception as e:
            raise ValueError("Error in Haversine distance calculation") from e

    def calculate_times(self, df, pickup_time, ordered_time, restaurant_lat, restaurant_lon, customer_lat, customer_lon):
        '''
        Calculate delivery times based on locations (DE, restaurant, customer).
        '''
        #print("pickup_time dtype:", df[pickup_time].dtype)
        #print("ordered_time dtype:", df[ordered_time].dtype)
        #print("restaurant_lat dtype:", df[restaurant_lat].dtype)
        #print("restaurant_lon dtype:", df[restaurant_lon].dtype)
        #print("customer_lat dtype:", df[customer_lat].dtype)
        #print("customer_lon dtype:", df[customer_lon].dtype)
        #print(df[[pickup_time, ordered_time, restaurant_lat, restaurant_lon, customer_lat, customer_lon]].head())
        #print(f"INSIDE calculat times {df.head}")

        
        try:
            # First Mile Time (FM) - time from driver to restaurant
            fm_distance = self.haversine(df[pickup_time], df[ordered_time], df[restaurant_lat], df[restaurant_lon])
            df['FM_time'] = fm_distance / self.average_speed_kmh   
            df['FM_time'] = df['FM_time'] / 60  # Convert to minutes

            # Last Mile Time (LM) - time from restaurant to customer
            lm_distance = self.haversine(df[restaurant_lat], df[restaurant_lon], df[customer_lat], df[customer_lon])
            df['LM_time'] = lm_distance / self.average_speed_kmh  
            df['LM_time'] = df['LM_time'] / 60  # Convert to minutes

            # Order to Assignment Time (O2A) - Can be a fixed value (e.g., 1 minutes for assignment)
            df['O2A_time'] = 1  # 1 minutes for assignment (almost immediately order is assigned -- can be adjusted)

            # Wait Time (WT) - time spent at the restaurant (can be fixed or estimated, e.g., assuming 2 minutes for all foods)
            df['WT_time'] = 2  # 2 minutes for waiting (can be adjusted)

            # Order to Reached Time (O2R) - total time from order to customer reached
            df['O2R_time'] = df['O2A_time'] + df['FM_time'] + df['WT_time'] + df['LM_time']
            
            
            df['distance'] = lm_distance / 1000   ## Convert to kms
            
            df['Delivery Time'] = df['O2R_time']

            return df
        except Exception as e:
            raise ValueError("Error in time calculations") from e




    def transform_data(self, df):
        '''
        Transform data: checking required columns and calculating distance.
        '''
        try:
            required_columns = ['Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude', 'Delivery_location_longitude']
            missing_columns = [col for col in required_columns if col not in df.columns]
            #print(f" missing colums are : {missing_columns}")
            if missing_columns:
                raise KeyError(f"Missing columns: {missing_columns}")
            
            self.distance_numpy(df, 'Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude', 'Delivery_location_longitude')
            # You can add other transformations here
        except Exception as e:
            raise ValueError("Error in data transformation") from e

    def fit(self, X, y=None):
        '''
        Fit method for compatibility with sklearn pipelines.
        '''
        return self

    def transform(self, X: pd.DataFrame, y=None):
        '''
        Transform method to apply the feature engineering steps.
        '''
        self.transform_data(X)
        return X
""" 

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class Feature_Engineering(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.average_speed_kmh = 40  # Average speed of delivery executive in km/h
        print(f"Initialized with average_speed_kmh: {self.average_speed_kmh}")
    def haversine(self, lat1, lon1, lat2, lon2):
        """
        Calculate the Haversine distance between two points in kilometers.
        """
        p = np.pi / 180  # Convert degrees to radians
        a = 0.5 - np.cos((lat2 - lat1) * p) / 2 + np.cos(lat1 * p) * np.cos(lat2 * p) * (1 - np.cos((lon2 - lon1) * p)) / 2
        return 12734 * np.arccos(np.clip(1 - a, -1, 1))  # Clip values to handle numerical errors

    def calculate_times(self, df, restaurant_lat, restaurant_lon, customer_lat, customer_lon):
        """
        Calculate delivery-related times and distances.
        """
        try:
            # Calculate distances
            fm_distance = self.haversine(df[restaurant_lat], df[restaurant_lon], df[customer_lat], df[customer_lon])
            df['distance'] = fm_distance
            
            # Delivery time components
            df['FM_time'] = fm_distance / self.average_speed_kmh * 60  # First Mile Time (minutes)
            df['LM_time'] = df['FM_time']  # Assuming similar time for last mile
            df['O2A_time'] = 1  # Fixed value for order assignment time
            df['WT_time'] = 2  # Fixed wait time
            
            # Total time
            df['O2R_time'] = df['FM_time'] + df['LM_time'] + df['O2A_time'] + df['WT_time']
            df['Delivery Time'] = df['FM_time']
            print(f"Delivery Time in Calculate Times Fun is : {df['Delivery Time'].iloc[0]}")
            DelTime = df['Delivery Time'].iloc[0]
            dist = df['distance'].iloc[0]
            return (df, DelTime, dist)
        except Exception as e:
            raise ValueError("Error in time calculations") from e

    def transform_data(self, df):
        """
        Transform data by ensuring required columns and calculating distances.
        """
        required_columns = ['Restaurant_latitude', 'Restaurant_longitude', 'Delivery_location_latitude', 'Delivery_location_longitude']
        missing_columns = [col for col in required_columns if col not in df.columns]
        self.average_speed_kmh = 40 # Average speed of delivery executive in km/h
        if missing_columns:
            raise KeyError(f"Missing columns: {missing_columns}")

        return self.calculate_times(
            df, 
            'Restaurant_latitude', 
            'Restaurant_longitude', 
            'Delivery_location_latitude', 
            'Delivery_location_longitude'
        )

    def fit(self, X, y=None):
        """
        Fit method for compatibility with sklearn pipelines.
        """
        self.feature_names_in_ = X.columns 
        return self

    def transform(self, X: pd.DataFrame, y=None):
        """
        Transform method to apply feature engineering steps.
        """
        return self.transform_data(X)



@dataclass 
class DataTransformationConfig():
    proccessed_obj_file_path = PREPROCESING_OBJ_FILE
    transform_train_path = TRANSFORM_TRAIn_FILE_PATH
    transform_test_path = TRANSFORM_TEST_FILE_PATH
    feature_engg_obj_path = FEATURE_ENGG_OBJ_FILE_PATH



class DataTransformation:
    
    def __init__(self):
        pass
    
    def drop_unnecessary_columns(self, df):
        """
        Drops unnecessary columns like 'Unnamed: 0', 'ID', and 'Delivery_person_ID'
        """
        columns_to_drop = ['Unnamed: 0', 'ID', 'Delivery_person_ID','time_taken(min)','FM_time',
                           'LM_time', 'O2A_time', 'WT_time', 'O2R_time','Delivery Time']
        df = df.drop(columns=columns_to_drop, errors='ignore')
        return df


    def get_data_transformation_obj(self):
        """
        Returns a pre-processing pipeline that transforms data.
        """
        try:
            # Categories for encoding
            Road_traffic_density = ['Low', 'Medium', 'High', 'Jam']
            Weatherconditions = ['Sunny', 'Cloudy', 'Fog', 'Sandstorms', 'Windy', 'Stormy']

            categorical_columns = ['Type_of_order', 'Type_of_vehicle', 'Festival', 'City']
            ordinal_encoder = ['Road_traffic_density', 'Weatherconditions']
            numerical_column = ['Delivery_person_Age', 'Delivery_person_Ratings', 'Vehicle_condition',
                                'multiple_deliveries', 'distance']

            # Numerical pipeline
            numerical_pipeline = Pipeline(steps=[
                ('impute', SimpleImputer(strategy='constant', fill_value=0)),
                ('scaler', StandardScaler(with_mean=False))
            ])

            # Categorical Pipeline
            categorical_pipeline = Pipeline(steps=[
                ('impute', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore')),
                ('scaler', StandardScaler(with_mean=False))
            ])

            # Ordinal Pipeline
            ordinal_pipeline = Pipeline(steps=[
                ('impute', SimpleImputer(strategy='most_frequent')),
                ('ordinal', OrdinalEncoder(categories=[Road_traffic_density, Weatherconditions])),
                ('scaler', StandardScaler(with_mean=False))
            ])

            # Column Transformer
            preprocessor = ColumnTransformer([
                ('numerical_pipeline', numerical_pipeline, numerical_column),
                ('categorical_pipeline', categorical_pipeline, categorical_columns),
                ('ordinal_pipeline', ordinal_pipeline, ordinal_encoder)
            ])

            logging.info("Pipeline Steps Completed")
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def get_feature_engineering_object(self):
        """
        Returns the feature engineering object for transformation.
        """
        try:
            feature_engineering = Pipeline(steps=[("fe", Feature_Engineering())])
            return feature_engineering
        except Exception as e:
            raise CustomException(e, sys)
         
         
    def initiate_data_transformation(self, train_path, test_path):
        """
        This method initiates the transformation process for both train and test data.
        """
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)
            
            
            # Apply feature engineering to both train and test datasets
            feature_engineering = self.get_feature_engineering_object()
            
            
            
            print("AFTER FEATURE ENGG BEFORE FeatEnggFitTrans")
            print(train_data.columns)
            print(test_data.columns)
            
            columns_to_drop = ['Unnamed: 0', 'ID', 'Delivery_person_ID','time_taken(min)']
            
            train_data = train_data.drop(columns=columns_to_drop, errors='ignore')
            test_data = test_data.drop(columns=columns_to_drop, errors='ignore')
            
            train_data = feature_engineering.fit_transform(train_data)
            test_data = feature_engineering.transform(test_data)
            
            print("AFTER FEATURE ENGG AFTER FeatEnggFitTrans")
            print(type(train_data))
            print(type(test_data))
            
            print(train_data)
            print(test_data)
            
            #converting the tuple back to dataframe
            train_data = train_data[0]
            test_data = test_data[0]
            
            print(type(train_data))
            print(type(test_data))
            
            print(train_data)
            print(test_data)

            
            train_data = self.drop_unnecessary_columns(train_data)
            test_data = self.drop_unnecessary_columns(test_data)
            
            # Apply pre-processing transformations to both datasets
            preprocessor = self.get_data_transformation_obj()
            expected_columns = ['Delivery_person_Age', 'Delivery_person_Ratings', 'Vehicle_condition', 
                    'multiple_deliveries', 'Type_of_order', 'Type_of_vehicle', 
                    'Festival', 'City', 'Road_traffic_density', 'Weatherconditions','Restaurant_latitude',
                    'Restaurant_longitude', 'Delivery_location_latitude','distance',
                    'Delivery_location_longitude', 'Order_Date', 'Time_Orderd','Time_Order_picked']
            #print("DataFrame columns:", train_data.columns)

            print(f"**************Before fit_transform {preprocessor}********************")
            
            for col in expected_columns:
                if col not in train_data.columns:
                    print(f"Column '{col}' is missing from the DataFrame.")
                    
            train_data_transformed = preprocessor.fit_transform(train_data)
            print(f"********************Transformed TRAIN Data is {train_data_transformed}********************")
            #print("**************After fit_transform Before Transform********************")
            test_data_transformed = preprocessor.transform(test_data)
            print(f"********************Transformed TEST Data is {test_data_transformed}********************")
            print("**************After fit_transform After Transform********************")
            return train_data_transformed, test_data_transformed
        except Exception as e:
            raise CustomException(e, sys)
    
    
