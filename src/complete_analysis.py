"""
Complete Spam Detection Analysis
Progressive evaluation on Small → Medium → Large datasets

Implements requirements from DLBAIPNLP01 guidelines:
- Multiple algorithms (Naive Bayes, SVM)
- Progressive dataset evaluation
- Comprehensive metrics
- Visualizations

Author: Martin Lana Bengut
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

from spam_detector import SpamDetector
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve

def create_visualizations(results, output_dir='visualizations'):
    """Create all required visualizations"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Performance comparison across dataset sizes
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    sizes = list(results.keys())
    metrics_names = ['accuracy', 'precision', 'recall', 'f1_score']
    
    for idx, metric in enumerate(metrics_names):
        ax = axes[idx // 2, idx % 2]
        
        nb_values = [results[size]['naive_bayes'][metric] for size in sizes]
        svm_values = [results[size]['svm'][metric] for size in sizes]
        
        x = np.arange(len(sizes))
        width = 0.35
        
        ax.bar(x - width/2, nb_values, width, label='Naive Bayes', color='#2196F3')
        ax.bar(x + width/2, svm_values, width, label='SVM', color='#4CAF50')
        
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} by Dataset Size', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{s} samples' for s in sizes])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0.85, 1.0])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/performance_by_size.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved performance comparison")
    
    # 2. Confusion matrices
    for size in sizes:
        for model_name in ['naive_bayes', 'svm']:
            cm = results[size][model_name]['confusion_matrix']
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Ham', 'Spam'],
                       yticklabels=['Ham', 'Spam'])
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')
            ax.set_title(f'Confusion Matrix - {model_name.replace("_", " ").title()}\nDataset Size: {size}',
                        fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/confusion_matrix_{model_name}_{size}.png', 
                       dpi=150, bbox_inches='tight')
            plt.close()
    
    print(f"✓ Saved confusion matrices")
    
    # 3. ROC Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for idx, model_name in enumerate(['naive_bayes', 'svm']):
        ax = axes[idx]
        
        for size in sizes:
            fpr = results[size][model_name]['roc_fpr']
            tpr = results[size][model_name]['roc_tpr']
            roc_auc = results[size][model_name]['roc_auc']
            
            ax.plot(fpr, tpr, label=f'{size} samples (AUC={roc_auc:.3f})', linewidth=2)
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curves - {model_name.replace("_", " ").title()}', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/roc_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved ROC curves")

def progressive_evaluation(detector):
    """Evaluate on progressively larger datasets"""
    
    print("\n" + "="*80)
    print("PROGRESSIVE DATASET EVALUATION")
    print("="*80)
    
    # Load full dataset
    df_full = detector.load_data()
    
    # Define dataset sizes (small → medium → large)
    sizes = {
        'Small (1000)': 1000,
        'Medium (3000)': 3000,
        'Large (Full)': len(df_full)
    }
    
    results = {}
    
    for size_name, size in sizes.items():
        print(f"\n{'='*80}")
        print(f"EVALUATING ON {size_name.upper()}")
        print(f"{'='*80}")
        
        # Prepare data
        df_sample = detector.prepare_data(df_full, sample_size=size)
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            df_sample['processed'], df_sample['label'],
            test_size=0.2, random_state=42, stratify=df_sample['label']
        )
        
        print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")
        
        # Vectorize
        X_train_vec, X_test_vec = detector.vectorize(X_train, X_test)
        
        # Train and evaluate both models
        size_results = {}
        
        for model_type, train_func in [('naive_bayes', detector.train_naive_bayes),
                                       ('svm', detector.train_svm)]:
            # Train
            model = train_func(X_train_vec, y_train)
            
            # Evaluate
            metrics, y_pred, y_prob = detector.evaluate_model(
                model, X_test_vec, y_test, model_type.replace('_', ' ').title()
            )
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # ROC curve
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            
            # Store results
            size_results[model_type] = {
                **metrics,
                'confusion_matrix': cm,
                'roc_fpr': fpr,
                'roc_tpr': tpr
            }
        
        results[size] = size_results
    
    return results

def save_results(results, filepath='outputs/results_summary.json'):
    """Save results to file"""
    # Convert arrays to lists for JSON
    results_serializable = {}
    for size, models in results.items():
        results_serializable[str(size)] = {}
        for model_name, metrics in models.items():
            results_serializable[str(size)][model_name] = {
                k: v.tolist() if isinstance(v, np.ndarray) else float(v) if isinstance(v, (np.float32, np.float64)) else v
                for k, v in metrics.items()
                if k not in ['confusion_matrix', 'roc_fpr', 'roc_tpr']
            }
    
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\n✓ Results saved to {filepath}")

def create_results_table(results):
    """Create formatted results table"""
    data = []
    
    for size, models in results.items():
        for model_name, metrics in models.items():
            data.append({
                'Dataset Size': size,
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}",
                'ROC-AUC': f"{metrics['roc_auc']:.4f}"
            })
    
    df_results = pd.DataFrame(data)
    df_results.to_csv('outputs/results_table.csv', index=False)
    
    print("\n" + "="*80)
    print("RESULTS SUMMARY TABLE")
    print("="*80)
    print(df_results.to_string(index=False))
    print("="*80)
    
    return df_results

def main():
    """Execute complete analysis"""
    
    print("\n" + "="*80)
    print("SPAM DETECTION SYSTEM - COMPLETE ANALYSIS")
    print("DLBAIPNLP01 – Project: NLP - Task 2")
    print("="*80)
    
    # Initialize detector
    detector = SpamDetector()
    
    # Progressive evaluation
    results = progressive_evaluation(detector)
    
    # Create visualizations
    create_visualizations(results)
    
    # Save results
    save_results(results)
    
    # Results table
    df_results = create_results_table(results)
    
    # Save best model
    detector.save_model('naive_bayes', 'models/spam_detector_nb.pkl')
    detector.save_model('svm', 'models/spam_detector_svm.pkl')
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\n✓ All models trained and evaluated")
    print("✓ Visualizations created")
    print("✓ Results saved")
    print("\n📁 Check:")
    print("   outputs/ - for CSV results")
    print("   visualizations/ - for PNG plots")
    print("   models/ - for trained models")
    
    return results

if __name__ == "__main__":
    results = main()

