# Cyber-Intrusion-Detection-using-ML

A comprehensive machine learning-based system for detecting cyber intrusions using network traffic analysis. This project implements and compares three different classification algorithms on the SIMARGL2021 dataset to identify malicious network activities.

## Features

- **Multi-Algorithm Approach**: Implements Random Forest, Decision Tree and Gaussian Naive Bayes classifiers
- **Comprehensive Data Preprocessing**: Automated feature selection, standardization and encoding
- **Performance Analysis**: Detailed evaluation with accuracy, F1-score, precision and recall metrics
- **Visualization Suite**: Multiple charts and graphs for data exploration and model comparison
- **Modular Design**: Clean, reusable code structure for easy maintenance and extension
- **Automated Pipeline**: End-to-end execution from raw data to trained models

## Dataset

The system uses the **SIMARGL2021** dataset, which contains network traffic features for cyber intrusion detection. The preprocessing pipeline selects 15 key features:

- `DST_TOS`, `SRC_TOS` - Type of Service fields
- `TCP_WIN_SCALE_OUT`, `TCP_WIN_SCALE_IN` - TCP window scaling
- `TCP_FLAGS` - TCP flags information
- `TCP_WIN_MAX_OUT`, `TCP_WIN_MIN_OUT` - TCP window sizes (outbound)
- `TCP_WIN_MAX_IN`, `TCP_WIN_MIN_IN` - TCP window sizes (inbound)
- `PROTOCOL` - Network protocol information
- `LAST_SWITCHED`, `FIRST_SWITCHED` - Flow timing information
- `TCP_WIN_MSS_IN` - Maximum segment size
- `TOTAL_FLOWS_EXP` - Flow export information
- `FLOW_DURATION_MILLISECONDS` - Flow duration
- `LABEL` - Target variable (intrusion type)


## Prerequisites
- Python 3.7 or higher
- pip package manager

## Core Components

### 1. Data Preprocessing (`data_preprocessing.py`)
- **Data Loading**: Efficient CSV file handling
- **Feature Selection**: Automated selection of 15 key features
- **Data Cleaning**: Duplicate removal and data validation
- **Visualization**: Label distribution analysis
- **Standardization**: Feature scaling and categorical encoding

### 2. Model Training (`model_training.py`)
- **Multi-Model Support**: Random Forest, Decision Tree, Naive Bayes
- **Performance Metrics**: Comprehensive evaluation suite
- **Model Persistence**: Automatic model saving and loading
- **Training Time Analysis**: Performance benchmarking

### 3. Model Comparison (`model_comparison.py`)
- **Visual Comparisons**: Multiple chart types for analysis
- **Performance Reports**: Detailed analysis and recommendations
- **Heatmap Visualization**: Performance metric correlation
- **Automated Reporting**: PDF and text report generation

### 4. Main Pipeline (`main.py`)
- **End-to-End Automation**: Complete pipeline execution
- **Flexible Configuration**: Command-line argument support
- **Error Handling**: Robust error management
- **Progress Tracking**: Detailed logging and status updates

## Performance Metrics

The system evaluates models using multiple metrics:

- **Accuracy**: Overall correctness of predictions
- **F1-Score**: Harmonic mean of precision and recall
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **Training Time**: Model training duration

## Model Algorithms

### 1. Random Forest Classifier
- **Ensemble Method**: Uses 30 decision trees
- **Advantages**: High accuracy, handles overfitting well
- **Best For**: Complex feature relationships

### 2. Decision Tree Classifier
- **Criteria**: Entropy-based splitting
- **Max Depth**: Limited to 4 levels
- **Advantages**: Interpretable, fast training
- **Best For**: Rule-based decision making

### 3. Gaussian Naive Bayes
- **Assumption**: Feature independence
- **Advantages**: Fast training and prediction
- **Best For**: High-dimensional data, baseline comparison

## Results

Based on runs with the SIMARGL2021 dataset:

| Model | Accuracy | F1-Score | Precision | Recall | Training Time |
|-------|----------|----------|-----------|---------|---------------|
| Random Forest | 0.9847 | 0.9843 | 0.9851 | 0.9847 | 2.45s |
| Decision Tree | 0.9652 | 0.9641 | 0.9658 | 0.9652 | 0.12s |
| Naive Bayes | 0.8934 | 0.8876 | 0.8945 | 0.8934 | 0.08s |

*Note: Results may vary based on dataset characteristics and system specifications. Used NVIDIA RTX 2050 for testing*

## Visualizations

The system generates several types of visualizations:

1. **Label Distribution**: Bar and pie charts showing class balance
2. **Accuracy Comparison**: Bar chart comparing model accuracies
3. **Training Time Analysis**: Performance vs. efficiency trade-offs
4. **Comprehensive Metrics**: Multi-panel comparison of all metrics
5. **Performance Heatmap**: Correlation analysis of model performance


## Acknowledgments

- **SIMARGL2021 Dataset**
- **Scikit-learn**

## 🔮 Future Enhancements

- Deep learning model integration (CNN, LSTM)
- Real-time intrusion detection capability
- Web-based dashboard for monitoring
- API endpoints for model serving
- Ensemble method implementation
- Hyperparameter optimization
- Cross-validation implementation
- Feature importance analysis
- Model interpretability tools

## References

1. Network Intrusion Detection using Machine Learning
2. SIMARGL2021 Dataset Documentation
3. Scikit-learn Documentation
4. Cybersecurity Best Practices
