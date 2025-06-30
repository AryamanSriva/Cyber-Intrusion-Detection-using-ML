"""
Model Training Module for Cyber Intrusion Detection
Implements Random Forest, Decision Tree, and Naive Bayes classifiers
"""

import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import joblib


class CyberIntrusionDetector:
    """
    A class to handle training and evaluation of multiple machine learning models
    for cyber intrusion detection
    """
    
    def __init__(self):
        self.models = {}
        self.training_times = {}
        self.performance_metrics = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def prepare_data(self, data_frame, test_size=0.3, random_state=42):
        """
        Separate features and labels, then split into train/test sets
        
        Args:
            data_frame (pd.DataFrame): Processed dataframe
            test_size (float): Proportion of data for testing
            random_state (int): Random state for reproducibility
        """
        print("Preparing data for training...")
        
        # Separate features and labels
        X = data_frame.drop(columns=['LABEL'], axis=1)
        y = data_frame['LABEL']
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training set shape: {self.X_train.shape}")
        print(f"Test set shape: {self.X_test.shape}")
        print("Data preparation complete!")
    
    def train_random_forest(self, n_estimators=30, random_state=42):
        """
        Train Random Forest classifier
        
        Args:
            n_estimators (int): Number of trees in the forest
            random_state (int): Random state for reproducibility
        """
        print("Training Random Forest Classifier...")
        
        # Create classifier
        rf_classifier = RandomForestClassifier(
            n_estimators=n_estimators, 
            random_state=random_state
        )
        
        # Measure training time
        start_time = time.time()
        rf_classifier.fit(self.X_train, self.y_train)
        end_time = time.time()
        
        # Store model and training time
        self.models['RandomForest'] = rf_classifier
        self.training_times['RandomForest'] = end_time - start_time
        
        print(f"Random Forest training completed in {self.training_times['RandomForest']:.4f} seconds")
    
    def train_decision_tree(self, criterion='entropy', max_depth=4, random_state=42):
        """
        Train Decision Tree classifier
        
        Args:
            criterion (str): The function to measure the quality of a split
            max_depth (int): Maximum depth of the tree
            random_state (int): Random state for reproducibility
        """
        print("Training Decision Tree Classifier...")
        
        # Create classifier
        dt_classifier = DecisionTreeClassifier(
            criterion=criterion, 
            max_depth=max_depth, 
            random_state=random_state
        )
        
        # Measure training time
        start_time = time.time()
        dt_classifier.fit(self.X_train, self.y_train)
        end_time = time.time()
        
        # Store model and training time
        self.models['DecisionTree'] = dt_classifier
        self.training_times['DecisionTree'] = end_time - start_time
        
        print(f"Decision Tree training completed in {self.training_times['DecisionTree']:.4f} seconds")
    
    def train_naive_bayes(self):
        """
        Train Gaussian Naive Bayes classifier
        """
        print("Training Gaussian Naive Bayes Classifier...")
        
        # Create classifier
        nb_classifier = GaussianNB()
        
        # Measure training time
        start_time = time.time()
        nb_classifier.fit(self.X_train, self.y_train)
        end_time = time.time()
        
        # Store model and training time
        self.models['NaiveBayes'] = nb_classifier
        self.training_times['NaiveBayes'] = end_time - start_time
        
        print(f"Naive Bayes training completed in {self.training_times['NaiveBayes']:.4f} seconds")
    
    def evaluate_model(self, model_name):
        """
        Evaluate a trained model and store performance metrics
        
        Args:
            model_name (str): Name of the model to evaluate
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found. Please train it first.")
            return
        
        print(f"Evaluating {model_name}...")
        
        # Make predictions
        model = self.models[model_name]
        y_pred = model.predict(self.X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'f1_score': f1_score(self.y_test, y_pred, average='weighted'),
            'precision': precision_score(self.y_test, y_pred, average='weighted'),
            'recall': recall_score(self.y_test, y_pred, average='weighted'),
            'training_time': self.training_times[model_name]
        }
        
        # Store metrics
        self.performance_metrics[model_name] = metrics
        
        # Print results
        print(f"\n{model_name} Performance:")
        print(f"Training Time: {metrics['training_time']:.4f} seconds")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print("-" * 50)
    
    def train_all_models(self):
        """
        Train all three models
        """
        print("Starting training of all models...")
        print("=" * 60)
        
        self.train_random_forest()
        self.train_decision_tree()
        self.train_naive_bayes()
        
        print("\nAll models trained successfully!")
    
    def evaluate_all_models(self):
        """
        Evaluate all trained models
        """
        print("\nEvaluating all models...")
        print("=" * 60)
        
        for model_name in self.models.keys():
            self.evaluate_model(model_name)
    
    def get_performance_summary(self):
        """
        Get a summary DataFrame of all model performances
        
        Returns:
            pd.DataFrame: Summary of model performances
        """
        if not self.performance_metrics:
            print("No models have been evaluated yet.")
            return None
        
        summary_data = []
        for model_name, metrics in self.performance_metrics.items():
            summary_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'F1 Score': metrics['f1_score'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'Training Time (s)': metrics['training_time']
            })
        
        return pd.DataFrame(summary_data)
    
    def save_models(self, directory='models'):
        """
        Save all trained models to disk
        
        Args:
            directory (str): Directory to save models
        """
        import os
        
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        for model_name, model in self.models.items():
            filename = os.path.join(directory, f'{model_name.lower()}_model.pkl')
            joblib.dump(model, filename)
            print(f"Saved {model_name} to {filename}")
    
    def load_model(self, model_path):
        """
        Load a saved model
        
        Args:
            model_path (str): Path to the saved model
        
        Returns:
            object: Loaded model
        """
        return joblib.load(model_path)


def main():
    """
    Main function to demonstrate the training pipeline
    """
    # Load processed data
    try:
        processed_data = pd.read_csv('processed_data.csv')
        print(f"Loaded processed data with shape: {processed_data.shape}")
    except FileNotFoundError:
        print("Error: processed_data.csv not found.")
        print("Please run data_preprocessing.py first to create the processed dataset.")
        return
    
    # Initialize detector
    detector = CyberIntrusionDetector()
    
    # Prepare data
    detector.prepare_data(processed_data)
    
    # Train all models
    detector.train_all_models()
    
    # Evaluate all models
    detector.evaluate_all_models()
    
    # Get performance summary
    summary = detector.get_performance_summary()
    if summary is not None:
        print("\nPerformance Summary:")
        print("=" * 80)
        print(summary.to_string(index=False))
        
        # Save summary to CSV
        summary.to_csv('model_performance_summary.csv', index=False)
        print("\nPerformance summary saved to 'model_performance_summary.csv'")
    
    # Save trained models
    detector.save_models()
    
    return detector


if __name__ == "__main__":
    main()
