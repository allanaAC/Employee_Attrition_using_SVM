import pandas as pd
import logging
from sklearn.discriminant_analysis import StandardScaler

def create_dummies(df, to_get_dummies_for): 
    try:
        df = pd.get_dummies(data=df, columns=to_get_dummies_for, drop_first=True)
        return df 
    except Exception as e:
        logging.error(" Error in create_dummies data: {}". format(e)) 
        
def map_columns(df):
    try:
        dict_OverTime = {'Yes': 1, 'No': 0}
        dict_attrition = {'Yes': 1, 'No': 0}
        df['OverTime'] = df.OverTime.map(dict_OverTime)
        df['Attrition'] = df.Attrition.map(dict_attrition)
        return df
    except Exception as e:
        logging.error(" Error in map_columns data: {}". format(e)) 
        
def scale_data(X):
    try:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return pd.DataFrame(X_scaled, columns=X.columns)
    except Exception as e:
        logging.error(" Error in scale_data data: {}". format(e)) 
        
def separate_features_target(df, target_column):
    try:
        Y = df[target_column]
        X = df.drop(columns=[target_column])
        return X, Y
    except Exception as e:
        logging.error(" Error in separate_features_target data: {}". format(e)) 
        
