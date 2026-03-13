#!/usr/bin/env python3
"""
Visualize AHB validation results across V5, V7, and Minimal Pairs probes.
"""

import matplotlib.pyplot as plt
import numpy as np

# Results from experiments
probes = ['V5\n(style-confounded)', 'V7\n(style-controlled)', 'Minimal Pairs\n(entity swap)']
pearson_r = [0.457, 0.428, -0.422]
spearman_r = [0.365, 0.389, -0.451]

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Colors
colors = ['#2ecc71', '#3498db', '#e74c3c']  # green, blue, red

# Plot 1: Bar chart of correlations
x = np.arange(len(probes))
width = 0.35

bars1 = ax1.bar(x - width/2, pearson_r, width, label='Pearson r', color=colors, alpha=0.8)
bars2 = ax1.bar(x + width/2, spearman_r, width, label='Spearman r', color=colors, alpha=0.5, hatch='//')

ax1.set_ylabel('Correlation with AHB Scores', fontsize=12)
ax1.set_title('AHB Validation: Three Probe Directions', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(probes, fontsize=11)
ax1.legend(loc='upper right')
ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax1.set_ylim(-0.6, 0.6)
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, val in zip(bars1, pearson_r):
    height = bar.get_height()
    ax1.annotate(f'{val:+.3f}',
                xy=(bar.get_x() + bar.get_width()/2, height),
                xytext=(0, 3 if height > 0 else -12),
                textcoords="offset points",
                ha='center', va='bottom' if height > 0 else 'top',
                fontsize=10, fontweight='bold')

# Plot 2: Conceptual diagram of what each measures
ax2.set_xlim(-1.5, 1.5)
ax2.set_ylim(-1.5, 1.5)

# Draw arrows for each direction
arrow_props = dict(arrowstyle='->', lw=2, mutation_scale=15)

# V5/V7 direction (positive correlation - green/blue)
ax2.annotate('', xy=(1, 0.3), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=3))
ax2.annotate('', xy=(0.95, 0.1), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='#3498db', lw=3))

# Minimal pairs direction (negative correlation - red, opposite direction)
ax2.annotate('', xy=(-0.9, -0.3), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=3))

# AHB direction (reference)
ax2.annotate('', xy=(0.8, 0.8), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='gray', lw=2, ls='--'))

# Labels
ax2.text(1.1, 0.35, 'V5 (r=+0.46)', fontsize=11, color='#2ecc71', fontweight='bold')
ax2.text(1.0, 0.0, 'V7 (r=+0.43)', fontsize=11, color='#3498db', fontweight='bold')
ax2.text(-1.4, -0.4, 'Minimal Pairs\n(r=-0.42)', fontsize=11, color='#e74c3c', fontweight='bold', ha='left')
ax2.text(0.6, 1.0, 'AHB\nCompassion', fontsize=11, color='gray', fontweight='bold')

# Add explanation
ax2.text(0, -1.3,
         'V5/V7: Welfare framing → positive correlation with AHB\n'
         'Minimal Pairs: Pet vs livestock language → negative correlation\n'
         '(AHB rewards compassion toward farm animals, not pets)',
         ha='center', fontsize=10, style='italic',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

ax2.set_title('Direction Alignment in Activation Space', fontsize=14, fontweight='bold')
ax2.set_aspect('equal')
ax2.axis('off')

plt.tight_layout()
plt.savefig('/Users/infinitespire/Desktop/ai_dev/caml/caml-research/experiments/linear-probes/outputs/visualizations/ahb_three_probe_comparison.png',
            dpi=150, bbox_inches='tight', facecolor='white')
plt.savefig('/Users/infinitespire/Desktop/ai_dev/caml/caml-research/experiments/linear-probes/outputs/visualizations/ahb_three_probe_comparison.pdf',
            bbox_inches='tight')
print("Saved visualization to outputs/visualizations/ahb_three_probe_comparison.png")
plt.show()
