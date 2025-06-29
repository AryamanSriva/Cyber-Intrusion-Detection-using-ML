"""
Main Pipeline for Cyber Intrusion Detection System
Orchestrates the complete machine learning pipeline from data preprocessing to model comparison
"""

import os
import sys
import argparse
from datetime import datetime

# Import custom modules
from data_preprocessing import (
    load_data, select_features, remove_duplicates, 
    visualize_labels, standardize_and_encode
)
from model_training import CyberIntrusionDetector
from model_comparison import ModelComparator


def setup_directories():
    """
    Create necessary directories for outputs
    """
    directories = ['models', 'plots', 'reports']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")


def run_preprocessing_pipeline(data_path='SIMARGL2021.csv'):
    """
    Run the complete data preprocessing pipeline
    
    Args:
        data_path (str): Path to the raw dataset
    
    Returns:
        pd.DataFrame: Processed dataset ready for training
        dict: Preprocessing objects for future use
    """
    print("Starting Data Preprocessing Pipeline...")
    print("=" * 60)
    
    try:
        # Step 1: Load data
        raw_data = load_data(data_path)
        
        # Step 2: Select important features
        selected_data = select_features(raw_data)
        
        # Step 3: Remove duplicates
        clean_data = remove_duplicates(selected_data)
        
        # Step 4: Create visualizations
        print("Creating data visualizations...")
        visualize_labels(clean_data, save_plots=True)
        
        # Step 5: Standardize and encode
        processed_data, preprocessing_objects = standardize_and_encode(clean_data)
        
        # Step 6: Save processed data
        processed_data.to_csv('processed_data.csv', index=False)
        print("Processed data saved to 'processed_data.csv'")
        
        print("\nData Preprocessing Pipeline Completed Successfully!")
        print("=" * 60)
        
        return processed_data, preprocessing_objects
        
    except Exception as e:
        print(f"Error in preprocessing pipeline: {str(e)}")
        return None, None


def run_training_pipeline(processed_data):
    """
    Run the complete model training pipeline
    
    Args:
        processed_data (pd.DataFrame): Preprocessed dataset
    
    Returns:
        CyberIntrusionDetector: Trained detector object
    """
    print("\nStarting Model Training Pipeline...")
    print("=" * 60)
    
    try:
        # Initialize detector
        detector = CyberIntrusionDetector()
        
        # Prepare data
        detector.prepare_data(processed_data)
        
        # Train all models
        detector.train_all_models()
        
        # Evaluate all models
        detector.evaluate_all_models()
        
        # Get and save performance summary
        summary = detector.get_performance_summary()
        if summary is not None:
            summary.to_csv('model_performance_summary.csv', index=False)
            print("Performance summary saved to 'model_performance_summary.csv'")
        
        # Save trained models
        detector.save_models()
        
        print("\nModel Training Pipeline Completed Successfully!")
        print("=" * 60)
        
        return detector
        
    except Exception as e:
        print(f"Error in training pipeline: {str(e)}")
        return None


def run_comparison_pipeline():
    """
    Run the model comparison and visualization pipeline
    """
    print("\nStarting Model Comparison Pipeline...")
    print("=" * 60)
    
    try:
        # Initialize comparator
        comparator = ModelComparator()
        
        # Load performance data
        comparator.load_performance_data()
        
        if comparator.performance_data is not None:
            # Generate performance report
            comparator.generate_performance_report()
            
            # Create all visualizations
            comparator.create_all_visualizations(save_plots=True)
            
            print("\nModel Comparison Pipeline Completed Successfully!")
            print("=" * 60)
            
            return comparator
        else:
            print("Unable to load performance data for comparison.")
            return None
            
    except Exception as e:
        print(f"Error in comparison pipeline: {str(e)}")
        return None


def generate_final_report(detector, comparator):
    """
    Generate a comprehensive final report
    
    Args:
        detector (CyberIntrusionDetector): Trained detector object
        comparator (ModelComparator): Model comparator object
    """
    print("\nGenerating Final Report...")
    print("=" * 60)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report_content = f"""
CYBER INTRUSION DETECTION SYSTEM - FINAL REPORT
================================================================
Generated on: {timestamp}

EXECUTIVE SUMMARY:
This report presents the results of a comprehensive machine learning approach
to cyber intrusion detection using the SIMARGL2021 dataset. Three different
algorithms were trained and evaluated: Random Forest, Decision Tree, and
Gaussian Naive Bayes.

DATASET INFORMATION:
- Original dataset: SIMARGL2021.csv
- Selected features: 15 key network traffic features
- Preprocessing: Standardization and label encoding applied
- Train/Test split: 70/30 ratio

MODELS EVALUATED:
1. Random Forest Classifier (30 estimators)
2. Decision Tree Classifier (entropy criterion, max_depth=4)
3. Gaussian Naive Bayes Classifier

PERFORMANCE SUMMARY:
"""
    
    if detector and detector.performance_metrics:
        for model_name, metrics in detector.performance_metrics.items():
            report_content += f"""
{model_name}:
  - Accuracy: {metrics['accuracy']:.4f}
  - F1 Score: {metrics['f1_score']:.4f}
  - Precision: {metrics['precision']:.4f}
  - Recall: {metrics['recall']:.4f}
  - Training Time: {metrics['training_time']:.4f} seconds
"""
    
    if comparator and comparator.performance_data is not None:
        best_model = comparator.performance_data.loc[comparator.performance_data['Accuracy'].idxmax()]
        report_content += f"""
RECOMMENDATION:
The {best_model['Model']} model achieved the highest accuracy of {best_model['Accuracy']:.4f}
and is recommended for deployment in the cyber intrusion detection system.

FILES GENERATED:
- processed_data.csv: Preprocessed dataset
- model_performance_summary.csv: Performance metrics summary
- Various visualization plots (.png files)
- Trained models saved in /models directory

CONCLUSION:
The machine learning approach successfully demonstrates the feasibility of
automated cyber intrusion detection with high accuracy rates across all
tested algorithms.
"""
    
    # Save report
    with open('final_report.txt', 'w') as f:
        f.write(report_content)
    
    print("Final report saved to 'final_report.txt'")
    print(report_content)


def main():
    """
    Main function to run the complete pipeline
    """
    parser = argparse.ArgumentParser(description='Cyber Intrusion Detection System')
    parser.add_argument('--data-path', default='SIMARGL2021.csv', 
                       help='Path to the dataset file')
    parser.add_argument('--skip-preprocessing', action='store_true',
                       help='Skip preprocessing if processed_data.csv exists')
    parser.add_argument('--skip-training', action='store_true',
                       help='Skip training if models already exist')
    parser.add_argument('--only-comparison', action='store_true',
                       help='Only run comparison pipeline')
    
    args = parser.parse_args()
    
    # Setup directories
    setup_directories()
    
    print("CYBER INTRUSION DETECTION SYSTEM")
    print("=" * 60)
    print(f"Starting pipeline at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    detector = None
    comparator = None
    
    if args.only_comparison:
        # Only run comparison pipeline
        comparator = run_comparison_pipeline()
    else:
        # Check if we should skip preprocessing
        if args.skip_preprocessing and os.path.exists('processed_data.csv'):
            print("Skipping preprocessing - loading existing processed data...")
            try:
                import pandas as pd
                processed_data = pd.read_csv('processed_data.csv')
                preprocessing_objects = None
                print(f"Loaded processed data with shape: {processed_data.shape}")
            except Exception as e:
                print(f"Error loading processed data: {e}")
                processed_data, preprocessing_objects = run_preprocessing_pipeline(args.data_path)
        else:
            # Run preprocessing pipeline
            processed_data, preprocessing_objects = run_preprocessing_pipeline(args.data_path)
        
        if processed_data is not None:
            # Check if we should skip training
            if args.skip_training and os.path.exists('model_performance_summary.csv'):
                print("Skipping training - models already exist...")
                detector = CyberIntrusionDetector()
                # Just run comparison
                comparator = run_comparison_pipeline()
            else:
                # Run training pipeline
                detector = run_training_pipeline(processed_data)
                
                if detector is not None:
                    # Run comparison pipeline
                    comparator = run_comparison_pipeline()
    
    # Generate final report
    if detector or comparator:
        generate_final_report(detector, comparator)
    
    print("\n" + "=" * 60)
    print("PIPELINE EXECUTION COMPLETED")
    print(f"Finished at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
    