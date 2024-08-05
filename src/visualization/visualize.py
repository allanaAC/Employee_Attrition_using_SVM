import logging
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def plot_histograms(df, num_cols):
    try:
        df[num_cols].hist(figsize=(14, 14))
        plt.show()
    except Exception as e:
        logging.error(" Error in plot_histograms data: {}". format(e))
        
def univariate_analysis(df, cat_cols):
    try:
        for col in cat_cols:
            print(df[col].value_counts(normalize=True))
            print('*' * 40)
    except Exception as e:
        logging.error(" Error in univariate_analysis: {}". format(e))

def bivariate_analysis(df, cat_cols):
    try:
        for col in cat_cols:
            if col != 'Attrition':
                (pd.crosstab(df[col], df['Attrition'], normalize='index') * 100).plot(kind='bar', figsize=(8, 4), stacked=True)
                plt.ylabel('Percentage Attrition %')
                plt.show()
    except Exception as e:
        logging.error(" Error in bivariate_analysis data: {}". format(e))
        
def plot_correlation(df, num_cols): 
    try:
        plt.figure(figsize=(15, 8))
        sns.heatmap(df[num_cols].corr(), annot=True, fmt='0.2f', cmap='YlGnBu')
        plt.show()
    except Exception as e:
        logging.error(" Error in plot_correlation data: {}". format(e))   
        
def metrics_score(actual, predicted):
    try:
        print(classification_report(actual, predicted))
        cm = confusion_matrix(actual, predicted)
        plt.figure(figsize=(8, 5))
        sns.heatmap(cm, annot=True, fmt='.2f', xticklabels=['Not Attrite', 'Attrite'], yticklabels=['Not Attrite', 'Attrite'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()
    except Exception as e:
        logging.error(" Error in metrics_score data: {}". format(e))
        
