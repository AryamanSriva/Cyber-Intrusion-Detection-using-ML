"""
Data Preprocessing Module for Cyber Intrusion Detection
Handles loading, cleaning, and preprocessing of the SIMARGL2021 dataset
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(file_path):
    """
    Load CSV file into a DataFrame
    
    Args:
        file_path (str): Path to the CSV file
    
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    print("Loading dataset...")
    data_frame = pd.read_csv(file_path)
    print(f"Dataset loaded successfully. Shape: {data_frame.shape}")
    return data_frame


def select_features(data_frame):
    """
    Select the 15 important features for analysis
    
    Args:
        data_frame (pd.DataFrame): Original dataframe
    
    Returns:
        pd.DataFrame: Dataframe with selected features
    """
    selected_columns = [
        'DST_TOS', 'SRC_TOS', 'TCP_WIN_SCALE_OUT', 'TCP_WIN_SCALE_IN', 'TCP_FLAGS',
        'TCP_WIN_MAX_OUT', 'PROTOCOL', 'TCP_WIN_MIN_OUT', 'TCP_WIN_MIN_IN',
        'TCP_WIN_MAX_IN', 'LAST_SWITCHED', 'TCP_WIN_MSS_IN', 'TOTAL_FLOWS_EXP',
        'FIRST_SWITCHED', 'FLOW_DURATION_MILLISECONDS', 'LABEL'
    ]
    
    print(f"Selecting {len(selected_columns)} features...")
    filtered_data = data_frame[selected_columns]
    print(f"Feature selection complete. Shape: {filtered_data.shape}")
    return filtered_data


def remove_duplicates(data_frame):
    """
    Remove duplicate rows from the dataframe
    
    Args:
        data_frame (pd.DataFrame): Input dataframe
    
    Returns:
        pd.DataFrame: Dataframe without duplicates
    """
    initial_shape = data_frame.shape
    data_frame.drop_duplicates(inplace=True)
    final_shape = data_frame.shape
    
    print(f"Duplicates removed. Shape changed from {initial_shape} to {final_shape}")
    print(f"Removed {initial_shape[0] - final_shape[0]} duplicate rows")
    
    return data_frame


def visualize_labels(data_frame, save_plots=False):
    """
    Create visualizations for label distribution
    
    Args:
        data_frame (pd.DataFrame): Input dataframe
        save_plots (bool): Whether to save plots to files
    """
    print("Creating label distribution visualizations...")
    
    # Bar graph
    grouped_data = data_frame.groupby('LABEL').size()
    fig, ax = plt.subplots(1, figsize=(10, 6))
    ax.bar(grouped_data.index, grouped_data.values)
    ax.set(xlabel='LABEL', ylabel='Distinct Count')
    ax.set_title('Label Distribution - Bar Chart')
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('label_distribution_bar.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Pie chart
    grouped_data = data_frame['LABEL'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.pie(grouped_data.values, labels=grouped_data.index, autopct='%1.1f%%', 
           textprops={'fontsize': 8})
    ax.set_title('Distribution of LABEL - Pie Chart')
    
    if save_plots:
        plt.savefig('label_distribution_pie.png', dpi=300, bbox_inches='tight')
    plt.show()


def standardize_and_encode(data_frame):
    """
    Standardize numerical features and encode categorical features
    
    Args:
        data_frame (pd.DataFrame): Input dataframe
    
    Returns:
        pd.DataFrame: Processed dataframe
        dict: Preprocessing objects for later use
    """
    print("Starting data standardization and encoding...")
    
    # Create a copy
    normalized_data = data_frame.copy()
    
    # Identify numerical and non-numerical columns
    numerical_columns = normalized_data.select_dtypes(include=['float64', 'int64']).columns
    non_numerical_columns = normalized_data.select_dtypes(exclude=['float64', 'int64']).columns
    
    print(f"Numerical columns: {len(numerical_columns)}")
    print(f"Non-numerical columns: {len(non_numerical_columns)}")
    
    # Label encode non-numerical columns
    label_encoder = LabelEncoder()
    for column in non_numerical_columns:
        normalized_data[column] = label_encoder.fit_transform(normalized_data[column])
    
    # Standardize numerical columns
    scaler = StandardScaler()
    normalized_data[numerical_columns] = scaler.fit_transform(normalized_data[numerical_columns])
    
    # Store preprocessing objects
    preprocessing_objects = {
        'label_encoder': label_encoder,
        'scaler': scaler,
        'numerical_columns': numerical_columns,
        'non_numerical_columns': non_numerical_columns
    }
    
    print("Data preprocessing complete!")
    return normalized_data, preprocessing_objects


def main():
    """
    Main function to demonstrate preprocessing pipeline
    """
    # Note: Update this path to your actual dataset location
    file_path = 'SIMARGL2021.csv'
    
    try:
        # Load and preprocess data
        raw_data = load_data(file_path)
        selected_data = select_features(raw_data)
        clean_data = remove_duplicates(selected_data)
        
        # Visualize label distribution
        visualize_labels(clean_data, save_plots=True)
        
        # Standardize and encode
        processed_data, preprocessing_objects = standardize_and_encode(clean_data)
        
        print(f"\nPreprocessing complete!")
        print(f"Final dataset shape: {processed_data.shape}")
        print(f"First few rows of processed data:")
        print(processed_data.head())
        
        # Save processed data
        processed_data.to_csv('processed_data.csv', index=False)
        print("Processed data saved to 'processed_data.csv'")
        
        return processed_data, preprocessing_objects
        
    except FileNotFoundError:
        print(f"Error: Dataset file '{file_path}' not found.")
        print("Please ensure the SIMARGL2021.csv file is in the correct location.")
        return None, None
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None, None


if __name__ == "__main__":
    main()
