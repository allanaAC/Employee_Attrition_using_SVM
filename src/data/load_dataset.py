#this function is to load data 
import pandas as pd
import numpy as np

import logging

#data_path = "/data/real_estate.csv"
def load_and_preprocess_data(file_path):
    
    try:
        df = pd.read_excel(file_path)
        print(df.head())
        return df
    except Exception as e:
        logging.error(" Error in processing data: {}". format(e))
        
def preprocess_data(df):
    try:
        df = df.drop(['EmployeeNumber', 'Over18', 'StandardHours'], axis=1)
        num_cols = ['DailyRate', 'Age', 'DistanceFromHome', 'MonthlyIncome', 'MonthlyRate', 'PercentSalaryHike', 'TotalWorkingYears',
                'YearsAtCompany', 'NumCompaniesWorked', 'HourlyRate', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager', 'TrainingTimesLastYear']
        cat_cols = ['Attrition', 'OverTime', 'BusinessTravel', 'Department', 'Education', 'EducationField', 'JobSatisfaction', 'EnvironmentSatisfaction', 'WorkLifeBalance',
                'StockOptionLevel', 'Gender', 'PerformanceRating', 'JobInvolvement', 'JobLevel', 'JobRole', 'MaritalStatus', 'RelationshipSatisfaction']
        return df, num_cols, cat_cols
    except Exception as e:
        logging.error(" Error in preprocess_data: {}". format(e))

        


