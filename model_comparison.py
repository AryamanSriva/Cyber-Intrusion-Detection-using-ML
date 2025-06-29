"""
Model Comparison Module for Cyber Intrusion Detection
Provides visualization and comparison of different machine learning models
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from model_training import CyberIntrusionDetector


class ModelComparator:
    """
    A class to compare and visualize the performance of different models
    """
    
    def __init__(self, detector=None):
        """
        Initialize the comparator
        
        Args:
            detector (CyberIntrusionDetector): Trained detector object
        """
        self.detector = detector
        self.performance_data = None
    
    def load_performance_data(self, csv_path='model_performance_summary.csv'):
        """
        Load performance data from CSV file
        
        Args:
            csv_path (str): Path to the performance summary CSV
        """
        try:
            self.performance_data = pd.read_csv(csv_path)
            print(f"Loaded performance data for {len(self.performance_data)} models")
        except FileNotFoundError:
            print(f"Error: {csv_path} not found.")
            print("Please run model_training.py first to generate the performance summary.")
    
    def plot_accuracy_comparison(self, save_plot=False):
        """
        Create a bar plot comparing model accuracies
        
        Args:
            save_plot (bool): Whether to save the plot
        """
        if self.performance_data is None:
            print("No performance data available. Please load data first.")
            return
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(self.performance_data['Model'], self.performance_data['Accuracy'], 
                      color=['#FF6B6B', '#4ECDC4', '#45B7D1'], alpha=0.8)
        
        # Add value labels on bars
        for bar, accuracy in zip(bars, self.performance_data['Accuracy']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{accuracy:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.xlabel('Machine Learning Algorithms', fontsize=12, fontweight='bold')
        plt.ylabel('Accuracy Score', fontsize=12, fontweight='bold')
        plt.title('Accuracy Comparison of Cyber Intrusion Detection Models', 
                 fontsize=14, fontweight='bold')
        plt.ylim(0, 1.1)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('accuracy_comparison.png', dpi=300, bbox_inches='tight')
            print("Accuracy comparison plot saved as 'accuracy_comparison.png'")
        
        plt.show()
    
    def plot_training_time_comparison(self, save_plot=False):
        """
        Create a bar plot comparing model training times
        
        Args:
            save_plot (bool): Whether to save the plot
        """
        if self.performance_data is None:
            print("No performance data available. Please load data first.")
            return
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(self.performance_data['Model'], self.performance_data['Training Time (s)'], 
                      color=['#FF9500', '#32CD32', '#9370DB'], alpha=0.8)
        
        # Add value labels on bars
        for bar, time_val in zip(bars, self.performance_data['Training Time (s)']):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{time_val:.4f}s', ha='center', va='bottom', fontweight='bold')
        
        plt.xlabel('Machine Learning Algorithms', fontsize=12, fontweight='bold')
        plt.ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
        plt.title('Training Time Comparison of Cyber Intrusion Detection Models', 
                 fontsize=14, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('training_time_comparison.png', dpi=300, bbox_inches='tight')
            print("Training time comparison plot saved as 'training_time_comparison.png'")
        
        plt.show()
    
    def plot_all_metrics_comparison(self, save_plot=False):
        """
        Create a comprehensive comparison of all metrics
        
        Args:
            save_plot (bool): Whether to save the plot
        """
        if self.performance_data is None:
            print("No performance data available. Please load data first.")
            return
        
        # Prepare data for plotting
        metrics = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
        models = self.performance_data['Model'].tolist()
        
        # Create subplot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Comprehensive Model Performance Comparison', fontsize=16, fontweight='bold')
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for i, metric in enumerate(metrics):
            row = i // 2
            col = i % 2
            ax = axes[row, col]
            
            bars = ax.bar(models, self.performance_data[metric], 
                         color=colors, alpha=0.8)
            
            # Add value labels
            for bar, value in zip(bars, self.performance_data[metric]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                       f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
            
            ax.set_title(f'{metric} Comparison', fontweight='bold')
            ax.set_ylabel(metric, fontweight='bold')
            ax.set_ylim(0, 1.1)
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('all_metrics_comparison.png', dpi=300, bbox_inches='tight')
            print("All metrics comparison plot saved as 'all_metrics_comparison.png'")
        
        plt.show()
    
    def plot_performance_heatmap(self, save_plot=False):
        """
        Create a heatmap of model performance metrics
        
        Args:
            save_plot (bool): Whether to save the plot
        """
        if self.performance_data is None:
            print("No performance data available. Please load data first.")
            return
        
        # Prepare data for heatmap (exclude training time for better scaling)
        heatmap_data = self.performance_data[['Model', 'Accuracy', 'F1 Score', 'Precision', 'Recall']]
        heatmap_data = heatmap_data.set_index('Model')
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(heatmap_data.T, annot=True, cmap='YlOrRd', 
                   cbar_kws={'label': 'Score'}, fmt='.4f')
        plt.title('Model Performance Heatmap', fontsize=14, fontweight='bold')
        plt.xlabel('Models', fontsize=12, fontweight='bold')
        plt.ylabel('Metrics', fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        if save_plot:
            plt.savefig('performance_heatmap.png', dpi=300, bbox_inches='tight')
            print("Performance heatmap saved as 'performance_heatmap.png'")
        
        plt.show()
    
    def generate_performance_report(self):
        """
        Generate a detailed performance report
        """
        if self.performance_data is None:
            print("No performance data available. Please load data first.")
            return
        
        print("=" * 80)
        print("CYBER INTRUSION DETECTION - MODEL PERFORMANCE REPORT")
        print("=" * 80)
        
        # Best performing model for each metric
        best_accuracy = self.performance_data.loc[self.performance_data['Accuracy'].idxmax()]
        best_f1 = self.performance_data.loc[self.performance_data['F1 Score'].idxmax()]
        best_precision = self.performance_data.loc[self.performance_data['Precision'].idxmax()]
        best_recall = self.performance_data.loc[self.performance_data['Recall'].idxmax()]
        fastest_training = self.performance_data.loc[self.performance_data['Training Time (s)'].idxmin()]
        
        print(f"\nBEST PERFORMING MODELS:")
        print(f"• Highest Accuracy: {best_accuracy['Model']} ({best_accuracy['Accuracy']:.4f})")
        print(f"• Highest F1 Score: {best_f1['Model']} ({best_f1['F1 Score']:.4f})")
        print(f"• Highest Precision: {best_precision['Model']} ({best_precision['Precision']:.4f})")
        print(f"• Highest Recall: {best_recall['Model']} ({best_recall['Recall']:.4f})")
        print(f"• Fastest Training: {fastest_training['Model']} ({fastest_training['Training Time (s)']:.4f}s)")
        
        print(f"\nDETAILED PERFORMANCE METRICS:")
        print("-" * 80)
        for _, row in self.performance_data.iterrows():
            print(f"\n{row['Model']}:")
            print(f"  Accuracy:      {row['Accuracy']:.4f}")
            print(f"  F1 Score:      {row['F1 Score']:.4f}")
            print(f"  Precision:     {row['Precision']:.4f}")
            print(f"  Recall:        {row['Recall']:.4f}")
            print(f"  Training Time: {row['Training Time (s)']:.4f} seconds")
        
        # Overall recommendation
        print(f"\nRECOMMENDATION:")
        print("-" * 40)
        overall_best = self.performance_data.loc[self.performance_data['Accuracy'].idxmax()]
        print(f"For cyber intrusion detection, {overall_best['Model']} shows the best overall")
        print(f"performance with {overall_best['Accuracy']:.4f} accuracy.")
        
        if overall_best['Training Time (s)'] > fastest_training['Training Time (s)'] * 2:
            print(f"However, if training time is critical, consider {fastest_training['Model']}")
            print(f"which trains {overall_best['Training Time (s)'] / fastest_training['Training Time (s)']:.1f}x faster.")
        
        print("=" * 80)
    
    def create_all_visualizations(self, save_plots=True):
        """
        Create all comparison visualizations
        
        Args:
            save_plots (bool): Whether to save all plots
        """
        print("Creating all visualization plots...")
        
        self.plot_accuracy_comparison(save_plots)
        self.plot_training_time_comparison(save_plots)
        self.plot_all_metrics_comparison(save_plots)
        self.plot_performance_heatmap(save_plots)
        
        if save_plots:
            print("All visualization plots have been saved!")


def main():
    """
    Main function to demonstrate model comparison
    """
    # Initialize comparator
    comparator = ModelComparator()
    
    # Load performance data
    comparator.load_performance_data()
    
    if comparator.performance_data is not None:
        # Generate performance report
        comparator.generate_performance_report()
        
        # Create all visualizations
        comparator.create_all_visualizations(save_plots=True)
    else:
        print("Unable to load performance data. Please run model_training.py first.")


if __name__ == "__main__":
    main()
