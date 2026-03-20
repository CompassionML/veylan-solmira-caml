"""Generate a comparison chart showing probe improvement across versions."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Data across versions
versions = ['v8', 'v9', 'v9.1c']

metrics = {
    'Reversed-Context\nAccuracy': [50.0, 90.0, 90.0],
    'Eval Overall\nPass Rate (%)': [50.0, 62.5, 79.2],
    'Must-Score-LOW\nPass Rate (%)': [0.0, 62.5, 100.0],
    'Dynamic Range\n(score std)': [1.0, 23.1, 21.1],
    'Word-Identity\nConfound |cos|': [None, 1.0, 27.5],  # scaled x100 for visibility
}

# Color scheme
colors = ['#e74c3c', '#f39c12', '#27ae60']  # red, orange, green

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle('Compassion Probe Evolution: v8 → v9 → v9.1c\n(meta-llama/Llama-3.1-8B base model)',
             fontsize=14, fontweight='bold', y=0.98)

# Plot 1: Reversed-context accuracy
ax = axes[0, 0]
vals = [50.0, 90.0, 90.0]
bars = ax.bar(versions, vals, color=colors, edgecolor='black', linewidth=0.5)
ax.set_ylabel('%')
ax.set_title('Reversed-Context\nAccuracy', fontsize=11)
ax.set_ylim(0, 105)
ax.axhline(y=60, color='red', linestyle='--', alpha=0.5, label='Fail threshold')
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, val + 2, f'{val:.0f}%', ha='center', fontsize=10, fontweight='bold')
ax.legend(fontsize=8)

# Plot 2: Overall eval pass rate
ax = axes[0, 1]
vals = [50.0, 62.5, 79.2]
bars = ax.bar(versions, vals, color=colors, edgecolor='black', linewidth=0.5)
ax.set_ylabel('%')
ax.set_title('Overall Eval\nPass Rate', fontsize=11)
ax.set_ylim(0, 105)
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, val + 2, f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')

# Plot 3: Must-score-LOW pass rate
ax = axes[0, 2]
vals = [0.0, 62.5, 100.0]
bars = ax.bar(versions, vals, color=colors, edgecolor='black', linewidth=0.5)
ax.set_ylabel('%')
ax.set_title('Must-Score-LOW\nPass Rate', fontsize=11)
ax.set_ylim(0, 115)
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, val + 2, f'{val:.0f}%', ha='center', fontsize=10, fontweight='bold')

# Plot 4: Dynamic range
ax = axes[1, 0]
vals = [1.0, 23.1, 21.1]
bars = ax.bar(versions, vals, color=colors, edgecolor='black', linewidth=0.5)
ax.set_ylabel('Score Std Dev')
ax.set_title('Dynamic Range\n(higher = better)', fontsize=11)
ax.set_ylim(0, 30)
ax.axhline(y=10, color='red', linestyle='--', alpha=0.5, label='Min threshold')
for bar, val in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.8, f'{val:.1f}', ha='center', fontsize=10, fontweight='bold')
ax.legend(fontsize=8)

# Plot 5: Key eval scores comparison
ax = axes[1, 1]
examples = ['Factory\nFarming\nEssay', 'Pet\nSentiment\n(should be LOW)', 'Dismiss\nSuffering\n(should be LOW)']
v8_scores = [59.0, 59.1, 59.0]
v9_scores = [63.1, 85.9, 63.6]
v91_scores = [40.4, 33.6, 27.0]

x = np.arange(len(examples))
w = 0.25
bars1 = ax.bar(x - w, v8_scores, w, label='v8', color='#e74c3c', edgecolor='black', linewidth=0.5)
bars2 = ax.bar(x, v9_scores, w, label='v9', color='#f39c12', edgecolor='black', linewidth=0.5)
bars3 = ax.bar(x + w, v91_scores, w, label='v9.1c', color='#27ae60', edgecolor='black', linewidth=0.5)
ax.set_ylabel('Score (0-100)')
ax.set_title('Key Eval Scores', fontsize=11)
ax.set_xticks(x)
ax.set_xticklabels(examples, fontsize=8)
ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
ax.set_ylim(0, 105)
ax.legend(fontsize=8)

# Plot 6: Summary text
ax = axes[1, 2]
ax.axis('off')
summary = (
    "Key Changes v8→v9.1c:\n\n"
    "1. 2×2 context-crossing design\n"
    "   (removes word-identity shortcut)\n\n"
    "2. Sample-weight balancing\n"
    "   (phases contribute equally)\n\n"
    "3. Percentile-based scoring\n"
    "   (replaces sigmoid compression)\n\n"
    "4. Expanded Phase 3 (+24 pairs)\n"
    "   • Acknowledged-then-dismissed\n"
    "   • Sentimentality vs ethics\n"
    "   • Understated compassion\n\n"
    "Training AUROC = 1.000 in ALL\n"
    "versions (not a useful metric)"
)
ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig('C:/Users/jasmi/Downloads/Extracting-compassion-vectors/v9/output/probe_evolution.png',
            dpi=150, bbox_inches='tight')
print("Saved to v9/output/probe_evolution.png")
