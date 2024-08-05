from src.visualization.visualize import metrics_score
import logging

# Function to predict and evaluate model 
def predict_model(model, X):
    try:
        return model.predict(X)
    except Exception as e: 
        logging.error(" Error in predict data: {}". format(e))   
        
def evaluate_model(model, x_train, y_train, x_test, y_test, model_name): 
    try:    
        print(f"{model_name} - Training Data")
        y_pred_train = predict_model(model, x_train)
        metrics_score(y_train, y_pred_train)

        print(f"{model_name} - Test Data")
        y_pred_test = predict_model(model, x_test)
        metrics_score(y_test, y_pred_test)
    except Exception as e:
        logging.error(" Error in evaluate_model data: {}". format(e))
