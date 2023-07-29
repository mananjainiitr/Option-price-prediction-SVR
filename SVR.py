import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Load and preprocess the dataset
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Split the dataset into training and testing sets
def split_data(data):
    X = data[['Strike_Price' ,'Spot_Price'	, 'Time' , 'BS_Call'	,'MC_Call'	,'FD_Call' , 'close']]
    y = data['C(T+1)']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

# Create and train the SVR model
def train_svr(X_train, y_train):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    svr = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr.fit(X_train_scaled, y_train)
    
    return svr, scaler

# Evaluate the model's performance
def evaluate_model(svr, scaler, X_test, y_test):
    X_test_scaled = scaler.transform(X_test)
    y_pred = svr.predict(X_test_scaled)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return X_test , y_pred, mse, r2

# Main function
def main():
    file_path = 'D6.csv'  # Replace with the path to your dataset
    data = load_and_preprocess_data("dataset/D6.csv")
    X_train, X_test, y_train, y_test = split_data(data)
    svr, scaler = train_svr(X_train, y_train)
    X_test , y_pred, mse, r2 = evaluate_model(svr, scaler, X_test, y_test)
    
    print(f'Mean Squared Error: {mse:.2f}')
    print(f'R2 Score: {r2:.2f}')

    X_test["pred"] = y_pred
    X_test["C(T+1)"] = y_test

    results = pd.DataFrame(X_test)
    results.to_csv('dataset/DataSet-P06.csv', index=False)

        

main()
