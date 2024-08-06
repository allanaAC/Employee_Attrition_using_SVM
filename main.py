import logging
import warnings
from src.data.load_dataset import load_and_preprocess_data, preprocess_data
from src.visualization.visualize import plot_histograms, univariate_analysis, bivariate_analysis, plot_correlation
from src.feature.build_features import create_dummies, map_columns, scale_data, separate_features_target
from src.model.train_model import split_data, train_logistic_regression, train_svm
from src.model.predict_model import evaluate_model
import openpyxl

# Configure logging to write to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

if __name__ == "__main__":
    # Load and preprocess the data
    warnings.filterwarnings("ignore")
    print(openpyxl.__version__)

    # Load data
    df = load_and_preprocess_data('src/data/HR_Employee_Attrition.xlsx')
    #print(df.head())

    # Preprocess data
    df, num_cols, cat_cols = preprocess_data(df)

    # Plot histograms
    plot_histograms(df, num_cols)

    # Univariate analysis
    univariate_analysis(df, cat_cols)

    # Bivariate analysis
    bivariate_analysis(df, cat_cols)

    # Plot correlation
    plot_correlation(df, num_cols)
    
     # Create dummies
    to_get_dummies_for = ['BusinessTravel', 'Department', 'Education', 'EducationField', 'EnvironmentSatisfaction', 'Gender', 'JobInvolvement', 'JobLevel', 'JobRole', 'MaritalStatus']
    df = create_dummies(df, to_get_dummies_for)

    # Map columns
    df = map_columns(df)
    
    # Separate target variable and other variables
    X, Y = separate_features_target(df, 'Attrition')

    # Scale data
    X_scaled = scale_data(X)

    # Split data
    x_train, x_test, y_train, y_test = split_data(X_scaled, Y)
    
    # Train and evaluate Logistic Regression model
    lg_model = train_logistic_regression(x_train, y_train)
    evaluate_model(lg_model, x_train, y_train, x_test, y_test, "Logistic Regression")

    # Train and evaluate SVM model
    svm_model = train_svm(x_train, y_train)
    evaluate_model(svm_model, x_train, y_train, x_test, y_test, "SVM")    

    