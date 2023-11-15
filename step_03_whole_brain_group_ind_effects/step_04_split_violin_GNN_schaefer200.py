import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd

dataset_PATH = "/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/step_03_whole_brain_group_ind_effects/pFC_eFC_match_schaefer200.csv"

data = pd.read_csv(dataset_PATH)

sns.set_theme(style="white")
tips = data

palettle_violin = {'Mismatched': '#FCB59C', 'Matched': '#AFCDE9'}
palettle_box = {'Mismatched': '#FB9874', 'Matched': '#8FB9E0'}

# Draw a nested violinplot and split the violins for easier comparison
fig, ax = plt.subplots(figsize=(7, 7))
ax = sns.violinplot(data=tips, x="dataset", y="data", hue="type",
                    split=True, inner="quartiles", linewidth=1.25, width=0.8,
                    dodge=False,
                    palette=palettle_violin)

ax = sns.boxplot(x="dataset", y="data",
                 hue="type",
                 showfliers=False,
                 data=tips, width=0.2, boxprops={'zorder': 2}, ax=ax,
                 palette=palettle_box)

sns.despine(top=True)

ax.legend_.remove()  # remove legend
x_left, x_right = -0.5, 1.5
y_low, y_high = 0.5, 0.9
ratio = 1.2
ax.set_xlim([x_left, x_right])
ax.set_ylim([y_low, y_high])
ax.set_yticks(np.arange(0.55, 0.851, 0.1))
ax.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

fontsize = 24
ax.set_xlabel("", fontsize=fontsize, fontname='Arial')
ax.set_ylabel("Individual coupling (r)", fontsize=fontsize, fontname='Arial')
ax.tick_params(bottom=True, left=True, width=2, length=6)


fig.subplots_adjust(
    top=0.981,
    bottom=0.12,
    left=0.3,
    right=0.9,
    hspace=0.2,
    wspace=0.2
)

for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontsize(fontsize)
    tick.label.set_fontname('Arial')
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontsize(fontsize)
    tick.label.set_fontname('Arial')
for _, s in ax.spines.items():
    s.set_color('black')
    s.set_linewidth(2)

plt.savefig("/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/step_03_whole_brain_group_ind_effects/pFC_eFC_match_schaefer200.png", dpi=300, bbox_inches = 'tight')
plt.show()
