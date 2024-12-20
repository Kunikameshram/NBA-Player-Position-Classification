import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data(main_filepath, dummy_filepath=None, min_minutes=5):
    """
    Load and prepare the NBA dataset(s)
    Returns prepared main data and dummy data (if provided)
    """
    # Load main dataset
    main_data = pd.read_csv(main_filepath)
    main_data = main_data[main_data['MP'] > min_minutes]
    
    # Load dummy dataset if provided
    dummy_data = None
    if dummy_filepath:
        dummy_data = pd.read_csv(dummy_filepath)
    
    # Encode positions
    position_encoder = LabelEncoder()
    if dummy_data is not None:
        combined_positions = np.concatenate([main_data['Pos'].unique(), dummy_data['Pos'].unique()])
        position_encoder.fit(combined_positions)
        dummy_data['Pos'] = position_encoder.transform(dummy_data['Pos'])
    else:
        position_encoder.fit(main_data['Pos'].unique())
    
    main_data['Pos'] = position_encoder.transform(main_data['Pos'])
    
    return main_data, dummy_data, position_encoder

def select_features(data, correlation_threshold=0.08):
    """
    Select features based on correlation with target
    Returns list of selected feature names
    """
    # Convert categorical variables to numeric
    data_encoded = data.copy()
    for col in data_encoded.select_dtypes(np.object_):
        data_encoded[col] = LabelEncoder().fit_transform(data_encoded[col])
    
    # Calculate correlations
    correlation_matrix = data_encoded.corr()
    target_correlations = correlation_matrix['Pos'].abs().sort_values(ascending=False)
    
    # Select features above threshold
    selected_features = target_correlations[target_correlations >= correlation_threshold].index.tolist()
    selected_features.remove('Pos')
    
    print("Selected features:", selected_features)
    return selected_features

def prepare_features(main_data, dummy_data, selected_features):
    """
    Prepare and scale features for training
    Returns training, testing, and dummy datasets
    """
    X_main = main_data[selected_features]
    y_main = main_data['Pos']
    
    # Scale features
    scaler = StandardScaler()
    X_main_scaled = scaler.fit_transform(X_main)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_main_scaled, y_main, test_size=0.2, random_state=42, stratify=y_main
    )
    
    # Prepare dummy dataset if available
    X_dummy_scaled = None
    y_dummy = None
    if dummy_data is not None:
        X_dummy = dummy_data[selected_features]
        X_dummy_scaled = scaler.transform(X_dummy)
        y_dummy = dummy_data['Pos']
    
    return X_train, X_test, y_train, y_test, X_dummy_scaled, y_dummy, scaler

def train_models(X_train, y_train):
    """
    Train multiple models with optimized hyperparameters
    Returns dictionary of trained models
    """
    model_params = {
        'neural_network': (MLPClassifier(random_state=42), {
            'hidden_layer_sizes': [(100,), (100, 50), (100, 100)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001],
            'learning_rate': ['constant', 'adaptive']
        }),
        'random_forest': (RandomForestClassifier(random_state=42), {
            'n_estimators': [100, 200],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5]
        }),
        'gradient_boosting': (GradientBoostingClassifier(random_state=42), {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        }),
        'svm': (SVC(random_state=42), {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear']
        })
    }
    
    trained_models = {}
    for name, (model, params) in model_params.items():
        print(f"\nTraining {name}...")
        grid_search = GridSearchCV(model, params, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        trained_models[name] = grid_search.best_estimator_
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.3f}")
    
    return trained_models

def evaluate_models(models, X_train, X_test, y_train, y_test, X_dummy=None, y_dummy=None):
    """
    Evaluate all trained models and return results
    """
    results = {}
    
    for name, model in models.items():
        print(f"\n{name.upper()} Results:")
        
        # Test set evaluation
        y_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {test_accuracy:.3f}")
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        print(f"CV Mean Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Store results
        results[name] = {
            'test_accuracy': test_accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        # Evaluate on dummy dataset if available
        if X_dummy is not None and y_dummy is not None:
            dummy_pred = model.predict(X_dummy)
            dummy_accuracy = accuracy_score(y_dummy, dummy_pred)
            results[name]['dummy_accuracy'] = dummy_accuracy
            print(f"Dummy Set Accuracy: {dummy_accuracy:.3f}")
    
    return results

def predict_position(stats, models, scaler, selected_features, position_encoder, X_test, y_test):
    """
    Predict position for new player stats using the best model
    """
    # Find best model based on test accuracy
    best_model_name = max(models.items(), 
                         key=lambda x: accuracy_score(y_test, x[1].predict(X_test)))[0]
    best_model = models[best_model_name]
    
    # Scale features
    scaled_stats = scaler.transform(stats[selected_features].reshape(1, -1))
    
    # Make prediction
    prediction = best_model.predict(scaled_stats)
    return position_encoder.inverse_transform(prediction)

# Main execution
def main():
    # Load and prepare data
    main_data, dummy_data, position_encoder = load_and_prepare_data('nba_stats.csv', 'dummy_test.csv')
    
    # Select features
    selected_features = select_features(main_data, correlation_threshold=0.1)
    
    # Prepare features
    X_train, X_test, y_train, y_test, X_dummy, y_dummy, scaler = prepare_features(
        main_data, dummy_data, selected_features
    )
    
    # Train models
    models = train_models(X_train, y_train)
    
    # Evaluate models
    results = evaluate_models(models, X_train, X_test, y_train, y_test, X_dummy, y_dummy)
    
    return models, scaler, selected_features, position_encoder, X_test, y_test

if __name__ == "__main__":
    models, scaler, selected_features, position_encoder, X_test, y_test = main()