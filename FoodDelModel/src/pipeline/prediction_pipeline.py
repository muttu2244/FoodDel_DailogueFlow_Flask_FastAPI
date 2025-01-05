import os 
import sys
import pandas as pd

from src.config.configuration import *
from src.exception import CustomException
from src.logger import logging
from src.utils import load_model
from src.components.data_transformation import Feature_Engineering


class PredictPipeline:
    def __init__(self):
        pass
    
    """
    def predict(self,features):
        try:
            preprocessor_path = PREPROCESING_OBJ_FILE
            model_path = MODEL_FILE_PATH
            
            preprocessor = load_model(preprocessor_path)
            model = load_model(model_path)
            
            
            
            data_encoded = pd.get_dummies(features, columns=['Weatherconditions', 'Road_traffic_density', 'Type_of_order', 'Type_of_vehicle', 'Festival', 'City'])
            data_scaled = preprocessor.transform(data_encoded)
            
            print(f"the data to be sent to predict is : {data_scaled.columns}")
            
            '''
            expected_columns = ['Delivery_person_Age', 'Delivery_person_Ratings', 'Vehicle_condition', 
                    'multiple_deliveries', 'distance', 'hour','Type_of_order', 'Type_of_vehicle', 
                    'Festival', 'City', 'Order_Date', 'Time_Orderd','Time_Order_picked', 'Road_traffic_density', 
                    'Weatherconditions','Restaurant_latitude', 'Restaurant_longitude',
                    'Delivery_location_latitude', 'Delivery_location_longitude']
            '''
            
            pred = model.predict(data_scaled)
            
            
            return pred 
        
        except Exception as e:
            logging.info("Error occured in prediction pipeline")
            raise CustomException(e,sys)
    """
    
    

    def predict(self, features):
        try:
            preprocessor_path = PREPROCESING_OBJ_FILE
            model_path = MODEL_FILE_PATH
            
            # Load preprocessor and model
            preprocessor = load_model(preprocessor_path)
            model = load_model(model_path)
            
            #print(type(preprocessor))
            #fe_step_name = list(preprocessor.named_steps.keys())[0]
            #fe_step = preprocessor.named_steps[fe_step_name]

            # Print its attributes
            #print(fe_step.__dict__)
            #print(dir(fe_step))
            #print(fe_step.__class__.__dict__)


            
            
            # Feature engineering: Calculate distance
            fe = Feature_Engineering()
            features['distance'] = fe.haversine(
                features['Restaurant_latitude'], features['Restaurant_longitude'],
                features['Delivery_location_latitude'], features['Delivery_location_longitude']
            )
            print(f"distance is : {features['distance']}")
            # Feature engineering: Extract hour from time
            #features['hour'] = features['Time_Orderd'].apply(extract_hour)
            
            # One-hot encoding for categorical features
            data_encoded = pd.get_dummies(features, columns=['Weatherconditions', 'Road_traffic_density', 
                                                            'Type_of_order', 'Type_of_vehicle', 
                                                            'Festival', 'City'])
            preprocessor.fit(data_encoded)
            # Align columns with the preprocessor
            missing_columns = [col for col in preprocessor.feature_names_in_ if col not in data_encoded.columns]
            for col in missing_columns:
                data_encoded[col] = 0  # Add missing columns
            
            # Ensure column order matches the preprocessor’s expectations
            data_encoded = data_encoded[preprocessor.feature_names_in_]
            
            #print(f"DATA ENCODED IS: {data_encoded.columns}")
            if 'distance' in data_encoded.columns:
                data_encoded = data_encoded.drop(columns=['distance'])

            # Transform data
            data_scaled = preprocessor.transform(data_encoded)
            DelTime = data_scaled[1]
            Distance = data_scaled[2]
            
            print(f"DATA SCALED Delivery Time AFTER transform IS: {DelTime}")
            print(f"DATA SCALED Distance AFTER transform IS: {Distance}")
            data_scaled = data_scaled[0].drop(columns=['distance', 'Delivery Time'], errors='ignore')
            # Predict
            pred = model.predict(data_scaled)
            return (DelTime,Distance,pred)

        except Exception as e:
            logging.info("Error occurred in prediction pipeline")
            raise CustomException(e, sys)




        
class CustomData:
    def __init__(self,
                 Delivery_person_Age:int,  
                 Delivery_person_Ratings:float, 
                 Weatherconditions:str, 
                 Road_traffic_density:str,  
                 Vehicle_condition:int,  
                 multiple_deliveries:int,
                 Restaurant_latitude:float,
                 Restaurant_longitude:float,
                 Delivery_location_latitude:float,
                 Delivery_location_longitude:float,
                 Type_of_order:str,
                 Type_of_vehicle:str,
                 Festival:str,
                 City:str):
        
        self.Delivery_person_Age = Delivery_person_Age
        self.Delivery_person_Ratings = Delivery_person_Ratings
        self.Weatherconditions = Weatherconditions
        self.Road_traffic_density = Road_traffic_density
        self.Vehicle_condition = Vehicle_condition
        self.multiple_deliveries = multiple_deliveries
        self.Restaurant_latitude = Restaurant_latitude
        self.Restaurant_longitude = Restaurant_longitude
        self.Delivery_location_latitude = Delivery_location_latitude
        self.Delivery_location_longitude = Delivery_location_longitude
        self.Type_of_order=Type_of_order
        self.Type_of_vehicle=Type_of_vehicle
        self.Festival=Festival
        self.City=City

        
    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Delivery_person_Age':[self.Delivery_person_Age],
                'Delivery_person_Ratings':[self.Delivery_person_Ratings],
                'Weatherconditions':[self.Weatherconditions],
                'Road_traffic_density':[self.Road_traffic_density],
                'Vehicle_condition':[self.Vehicle_condition],
                'multiple_deliveries':[self.multiple_deliveries],
                'Restaurant_latitude' :[self.Restaurant_latitude],
                'Restaurant_longitude':[self.Restaurant_longitude],
                'Delivery_location_latitude':[self.Delivery_location_latitude],
                'Delivery_location_longitude':[self.Delivery_location_longitude],
                'Type_of_order':[self.Type_of_order],
                'Type_of_vehicle':[self.Type_of_vehicle],
                'Festival':[self.Festival],
                'City':[self.City]


            }
            
            df = pd.DataFrame(custom_data_input_dict)
            logging.info("DataFrame gatherd")
            
            return df
        except Exception as e:
            logging.info("Exception occured in Custom data")
            raise CustomException(e,sys)