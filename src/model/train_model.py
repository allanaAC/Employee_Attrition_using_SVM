import logging
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# Function to train the model
def split_data(X, Y): 
    try:
        return train_test_split(X, Y, test_size=0.2, random_state=1, stratify=Y) 
    except Exception as e:
        logging.error(" Error in scale_data data: {}". format(e)) 

def train_logistic_regression(x_train, y_train):
    try:
        lg = LogisticRegression()
        lg.fit(x_train, y_train)
        return lg 
    except Exception as e:
        logging.error(" Error in scale_data data: {}". format(e))
        
def train_svm(x_train, y_train, kernel='poly', degree=3):   
    try:
        model_SVC = svm.SVC()
        tsvm = SVC(kernel=kernel, degree=degree)  
        tsvm.fit(x_train, y_train)
        return tsvm
    except Exception as e:  
        logging.error(" Error in train_svm data: {}". format(e))   
        
