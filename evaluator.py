import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report, 
    roc_curve, confusion_matrix
)
from config import Config

class ProfessionalEvaluator:
    """Professional evaluation system with comprehensive metrics"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = config.get_device()
        self.save_dir = Path(config.results_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate_model(self, model, test_loader):
        """Comprehensive model evaluation"""
        print("🔍 Starting comprehensive evaluation...")
        
        model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for sequences, labels in tqdm(test_loader, desc="Evaluating"):
                sequences = sequences.to(self.device)
                labels = labels.to(self.device)
                
                outputs = model(sequences)
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels, all_predictions)
        auc_score = roc_auc_score(all_labels, [p[1] for p in all_probabilities])
        
        # Classification report
        class_report = classification_report(
            all_labels, all_predictions,
            target_names=['Fake', 'Real'],
            output_dict=True
        )
        
        # Print results
        print("=" * 60)
        print("📊 EVALUATION RESULTS")
        print("=" * 60)
        print(f"🎯 Accuracy: {accuracy:.4f}")
        print(f"📈 AUC Score: {auc_score:.4f}")
        print("\n📋 Classification Report:")
        print(classification_report(all_labels, all_predictions, target_names=['Fake', 'Real']))
        
        # Generate visualizations
        if self.config.save_visualizations:
            self._plot_roc_curve(all_labels, all_probabilities, auc_score)
            self._plot_confusion_matrix(all_labels, all_predictions)
        
        # Save results
        results = {
            'accuracy': accuracy,
            'auc_score': auc_score,
            'classification_report': class_report,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities
        }
        
        # Save to JSON (excluding large arrays)
        results_summary = {
            'accuracy': accuracy,
            'auc_score': auc_score,
            'classification_report': class_report
        }
        
        with open(self.save_dir / 'evaluation_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        return results
    
    def _plot_roc_curve(self, labels, probabilities, auc_score):
        """Plot and save ROC curve"""
        fpr, tpr, _ = roc_curve(labels, [p[1] for p in probabilities])
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - ISTVT Deepfake Detection')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        viz_dir = self.save_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        plt.savefig(viz_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 ROC curve saved to {viz_dir / 'roc_curve.png'}")
    
    def _plot_confusion_matrix(self, labels, predictions):
        """Plot and save confusion matrix"""
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Fake', 'Real'],
                   yticklabels=['Fake', 'Real'])
        plt.title('Confusion Matrix - ISTVT Deepfake Detection')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        viz_dir = self.save_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        plt.savefig(viz_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Confusion matrix saved to {viz_dir / 'confusion_matrix.png'}")
    
    def generate_report(self, results: Dict):
        """Generate comprehensive text report"""
        report_path = self.save_dir / 'evaluation_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("ISTVT Deepfake Detection - Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Overall Performance:\n")
            f.write(f"  Accuracy: {results['accuracy']:.4f}\n")
            f.write(f"  AUC Score: {results['auc_score']:.4f}\n\n")
            
            f.write("Classification Report:\n")
            class_report = results['classification_report']
            
            for class_name in ['0', '1']:  # Fake, Real
                class_label = 'Fake' if class_name == '0' else 'Real'
                if class_name in class_report:
                    f.write(f"  {class_label}:\n")
                    f.write(f"    Precision: {class_report[class_name]['precision']:.4f}\n")
                    f.write(f"    Recall: {class_report[class_name]['recall']:.4f}\n")
                    f.write(f"    F1-Score: {class_report[class_name]['f1-score']:.4f}\n")
            
            # Confusion matrix
            cm = confusion_matrix(results['labels'], results['predictions'])
            f.write(f"\nConfusion Matrix:\n")
            f.write(f"  Predicted:  Fake  Real\n")
            f.write(f"  Fake:      {cm[0,0]:4d}  {cm[0,1]:4d}\n")
            f.write(f"  Real:      {cm[1,0]:4d}  {cm[1,1]:4d}\n")
        
        print(f"📄 Evaluation report saved to {report_path}")
