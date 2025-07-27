#!/usr/bin/env python3
"""
Create clinical decision support mockup for bladder cancer treatment prediction.
Shows how the model would be used in clinical practice.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

def create_clinical_interface_mockup(data_dir, output_dir):
    """Create a professional clinical decision support interface mockup."""
    # Load data
    with open(f'{data_dir}/individual_model_results.json', 'r') as f:
        results = json.load(f)
    
    # Get first patient example
    patient_data = results['clinical_decision_examples'][0]
    
    # Create figure
    fig = plt.figure(figsize=(14, 10))
    
    # Define colors
    colors = {
        'header': '#1e3a8a',
        'positive': '#059669',
        'negative': '#dc2626',
        'neutral': '#6b7280',
        'background': '#f9fafb',
        'card': '#ffffff'
    }
    
    # Add background
    fig.patch.set_facecolor(colors['background'])
    
    # Title section
    ax_title = plt.axes([0, 0.92, 1, 0.08])
    ax_title.axis('off')
    ax_title.text(0.5, 0.5, 'Bladder Cancer Treatment Response Prediction System', 
                 ha='center', va='center', fontsize=20, fontweight='bold', 
                 color=colors['header'])
    
    # Patient info section
    ax_patient = plt.axes([0.02, 0.82, 0.46, 0.08])
    ax_patient.axis('off')
    patient_box = FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02",
                                facecolor=colors['card'], edgecolor='none',
                                transform=ax_patient.transAxes)
    ax_patient.add_patch(patient_box)
    
    ax_patient.text(0.02, 0.7, f"Patient ID: {patient_data['patient_id']}", 
                   fontsize=14, fontweight='bold', transform=ax_patient.transAxes)
    ax_patient.text(0.02, 0.3, f"True Response: {'Responder' if patient_data['true_response'] else 'Non-responder'}", 
                   fontsize=12, transform=ax_patient.transAxes)
    
    # Individual modality predictions
    modalities = ['expression', 'protein', 'methylation', 'mutation']
    modality_names = {
        'expression': 'Gene Expression',
        'protein': 'Protein Expression',
        'methylation': 'DNA Methylation',
        'mutation': 'Mutation Status'
    }
    
    # Create prediction cards for each modality
    for i, modality in enumerate(modalities):
        ax = plt.axes([0.02 + (i % 2) * 0.49, 0.55 - (i // 2) * 0.25, 0.46, 0.22])
        ax.axis('off')
        
        # Card background
        card = FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02",
                            facecolor=colors['card'], 
                            edgecolor='lightgray', linewidth=1,
                            transform=ax.transAxes)
        ax.add_patch(card)
        
        # Get prediction data
        pred = patient_data['predictions'][modality]
        prob = pred['probability']
        is_responder = pred['prediction'] == 1
        
        # Title
        ax.text(0.02, 0.85, modality_names[modality], fontsize=14, fontweight='bold',
               transform=ax.transAxes)
        
        # Probability bar
        bar_width = 0.6
        bar_x = 0.02
        bar_y = 0.5
        
        # Background bar
        bg_bar = FancyBboxPatch((bar_x, bar_y), bar_width, 0.15,
                               boxstyle="round,pad=0.01",
                               facecolor='#e5e7eb', edgecolor='none',
                               transform=ax.transAxes)
        ax.add_patch(bg_bar)
        
        # Probability fill
        prob_color = colors['positive'] if is_responder else colors['negative']
        prob_bar = FancyBboxPatch((bar_x, bar_y), bar_width * prob, 0.15,
                                 boxstyle="round,pad=0.01",
                                 facecolor=prob_color, edgecolor='none',
                                 alpha=0.8, transform=ax.transAxes)
        ax.add_patch(prob_bar)
        
        # Probability text
        ax.text(bar_x + bar_width/2, bar_y + 0.075, f'{prob:.1%}',
               ha='center', va='center', fontsize=12, fontweight='bold',
               color='white' if prob > 0.5 else 'black',
               transform=ax.transAxes)
        
        # Prediction result
        pred_text = 'Predicted: Responder' if is_responder else 'Predicted: Non-responder'
        ax.text(0.02, 0.25, pred_text, fontsize=11,
               color=prob_color, fontweight='bold',
               transform=ax.transAxes)
        
        # Model info
        model_info = patient_data['modality_data'][modality]
        ax.text(0.02, 0.05, f"Model: {model_info['best_model'].upper()} | Features: {model_info['n_features_used']}", 
               fontsize=9, color=colors['neutral'],
               transform=ax.transAxes)
        
        # Confidence indicator
        conf_x = bar_x + bar_width + 0.05
        conf_y = bar_y + 0.075
        
        if prob > 0.7 or prob < 0.3:
            conf_text = 'High confidence'
            conf_color = prob_color
        else:
            conf_text = 'Low confidence'
            conf_color = colors['neutral']
        
        ax.text(conf_x, conf_y, conf_text, fontsize=9,
               color=conf_color, va='center',
               transform=ax.transAxes)
    
    # Fusion prediction section
    ax_fusion = plt.axes([0.52, 0.75, 0.46, 0.15])
    ax_fusion.axis('off')
    
    # Fusion card
    fusion_card = FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02",
                               facecolor=colors['header'], edgecolor='none',
                               transform=ax_fusion.transAxes)
    ax_fusion.add_patch(fusion_card)
    
    # Calculate fusion prediction (simple average for mockup)
    fusion_prob = np.mean([patient_data['predictions'][m]['probability'] 
                          for m in modalities])
    fusion_pred = 1 if fusion_prob > 0.5 else 0
    
    ax_fusion.text(0.5, 0.7, 'FUSION PREDICTION', ha='center', fontsize=16,
                  fontweight='bold', color='white',
                  transform=ax_fusion.transAxes)
    
    fusion_result = 'RESPONDER' if fusion_pred else 'NON-RESPONDER'
    ax_fusion.text(0.5, 0.3, fusion_result, ha='center', fontsize=20,
                  fontweight='bold', color='white',
                  transform=ax_fusion.transAxes)
    
    # Key features section
    ax_features = plt.axes([0.02, 0.05, 0.96, 0.22])
    ax_features.axis('off')
    
    features_card = FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02",
                                  facecolor=colors['card'], 
                                  edgecolor='lightgray', linewidth=1,
                                  transform=ax_features.transAxes)
    ax_features.add_patch(features_card)
    
    ax_features.text(0.02, 0.85, 'Key Molecular Features', fontsize=14, fontweight='bold',
                    transform=ax_features.transAxes)
    
    # Show top features for each modality
    y_pos = 0.65
    for modality in modalities:
        if modality in patient_data['modality_data']:
            features = list(patient_data['modality_data'][modality]['feature_values'].items())[:3]
            feature_text = f"{modality_names[modality]}: "
            feature_text += ", ".join([f"{feat} ({val:.2f})" for feat, val in features])
            ax_features.text(0.02, y_pos, feature_text, fontsize=10,
                           transform=ax_features.transAxes)
            y_pos -= 0.15
    
    # Clinical recommendation
    ax_rec = plt.axes([0.52, 0.55, 0.46, 0.15])
    ax_rec.axis('off')
    
    rec_card = FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02",
                            facecolor='#fef3c7' if fusion_pred else '#fee2e2',
                            edgecolor='#f59e0b' if fusion_pred else '#ef4444',
                            linewidth=2,
                            transform=ax_rec.transAxes)
    ax_rec.add_patch(rec_card)
    
    ax_rec.text(0.02, 0.7, 'Clinical Recommendation:', fontsize=12, fontweight='bold',
               transform=ax_rec.transAxes)
    
    if fusion_pred:
        rec_text = "Consider standard therapy - patient likely to respond"
    else:
        rec_text = "Consider alternative therapy - patient unlikely to respond"
    
    ax_rec.text(0.02, 0.3, rec_text, fontsize=11,
               transform=ax_rec.transAxes, wrap=True)
    
    # Add disclaimer
    ax_disc = plt.axes([0, 0, 1, 0.03])
    ax_disc.axis('off')
    ax_disc.text(0.5, 0.5, 
                'This is a clinical decision support tool. All predictions should be reviewed by qualified medical professionals.',
                ha='center', va='center', fontsize=9, style='italic', color=colors['neutral'])
    
    # Save figure
    plt.savefig(f'{output_dir}/clinical_decision_support_mockup.png', 
                dpi=300, bbox_inches='tight', facecolor=colors['background'])
    plt.savefig(f'{output_dir}/clinical_decision_support_mockup.pdf', 
                bbox_inches='tight', facecolor=colors['background'])
    plt.close()
    
    print("Clinical decision support mockup created successfully!")
    
    # Print summary
    print(f"\nPatient: {patient_data['patient_id']}")
    print(f"True response: {'Responder' if patient_data['true_response'] else 'Non-responder'}")
    print("\nIndividual predictions:")
    for modality in modalities:
        pred = patient_data['predictions'][modality]
        print(f"  {modality_names[modality]}: {pred['probability']:.1%} ({'Responder' if pred['prediction'] else 'Non-responder'})")
    print(f"\nFusion prediction: {fusion_prob:.1%} ({'Responder' if fusion_pred else 'Non-responder'})")


def main():
    """Generate clinical decision support mockup."""
    # Setup paths
    data_dir = '/Users/tobyliu/bladder'
    output_dir = Path(data_dir) / '04_analysis_viz' / 'clinical_mockup'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Creating clinical decision support mockup...")
    print("=" * 60)
    
    create_clinical_interface_mockup(data_dir, output_dir)
    
    print("\n" + "=" * 60)
    print(f"Mockup saved to: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()