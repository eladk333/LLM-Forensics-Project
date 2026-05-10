import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Your original chart colors
C_LIN = '#78b79f'
C_MLP = '#eb9875'
C_RF  = '#7b9dd4'

# -------------------------------------------------------------
# 1. Comparison Chart Legend (4 Features vs 5 Features)
# -------------------------------------------------------------
handles_comparison = [
    # Solid bars (4 Features)
    Patch(facecolor=C_LIN, edgecolor='white', label='Linear (4 Feat)'),
    Patch(facecolor=C_MLP, edgecolor='white', label='MLP (4 Feat)'),
    Patch(facecolor=C_RF,  edgecolor='white', label='Random Forest (4 Feat)'),
    
    # Hatched bars (5 Features)
    Patch(facecolor=C_LIN, edgecolor='white', hatch='//', alpha=0.85, label='Linear (5 Feat)'),
    Patch(facecolor=C_MLP, edgecolor='white', hatch='//', alpha=0.85, label='MLP (5 Feat)'),
    Patch(facecolor=C_RF,  edgecolor='white', hatch='//', alpha=0.85, label='Random Forest (5 Feat)')
]

fig1, ax1 = plt.subplots(figsize=(6, 2))
ax1.axis('off') # Hide the entire chart area

# Generate the legend in the center of the invisible figure
legend1 = ax1.legend(handles=handles_comparison, loc='center', ncol=2, framealpha=0.9, fontsize=9)

# Save only the tight bounding box of the legend
fig1.savefig("legend_comparison.png", dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.close(fig1)
print("Saved standalone legend: 'legend_comparison.png'")

# -------------------------------------------------------------
# 2. Delta Chart Legend (Impact of removing logit_norm)
# -------------------------------------------------------------
handles_delta = [
    Patch(facecolor=C_LIN, edgecolor='white', label='Linear Delta'),
    Patch(facecolor=C_MLP, edgecolor='white', label='MLP Delta'),
    Patch(facecolor=C_RF,  edgecolor='white', label='Random Forest Delta')
]

fig2, ax2 = plt.subplots(figsize=(4, 1.5))
ax2.axis('off')

legend2 = ax2.legend(handles=handles_delta, loc='center', framealpha=0.9, fontsize=9)

fig2.savefig("legend_delta.png", dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.close(fig2)
print("Saved standalone legend: 'legend_delta.png'")