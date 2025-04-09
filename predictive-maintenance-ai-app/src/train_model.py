import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def train_predictive_model():
    """Train a predictive maintenance model"""
    # Load sample data (replace with real data)
    data = pd.read_csv('../data/sample_turbine_data.csv')
    
    # Feature engineering
    X = data.drop(['timestamp', 'turbine_id', 'failure_status'], axis=1)
    y = data['failure_status']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    print(f"Model accuracy: {accuracy_score(y_test, predictions)}")
    
    # Save model
    joblib.dump(model, '../models/predictive_model.pkl')

if __name__ == '__main__':
    train_predictive_model()
