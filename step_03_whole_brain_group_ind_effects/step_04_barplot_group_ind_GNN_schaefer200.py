import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

bar_colors = ['#AFCDE9','#FCB59C']

dataset_PATH = "/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/step_03_whole_brain_group_ind_effects/pFC_eFC_match_schaefer200.csv"
df = pd.read_csv(dataset_PATH)

a0 = np.mean(df[(df['type'] == 'Mismatched') & (df['dataset'] == 'HCP-YA')]['data'])#HCP-YA group
b0 = np.mean(df[(df['type'] == 'Matched') & (df['dataset'] == 'HCP-YA')]['data']) - a0 #HCP-YA individual
a1 = np.mean(df[(df['type'] == 'Mismatched') & (df['dataset'] == 'HCP-D')]['data']) #HCP-D group
b1 = np.mean(df[(df['type'] == 'Matched') & (df['dataset'] == 'HCP-D')]['data']) - a1 #HCP-D individual

fig, ax1  = plt.subplots(figsize=(7,7))
height =0.5
x = [0.5, 1.5]
ax1.bar(x[0], a0, label='Group effect', color=bar_colors[1], width=height, edgecolor='black', linewidth=0.5, alpha=0.8)
ax1.bar(x[0], b0, bottom=a0, label='Individual effect', color=bar_colors[0],width=height, edgecolor='black', linewidth=0.5)

ax1.bar(x[1], a1, label='Group effect', color=bar_colors[1], width=height, edgecolor='black',  linewidth=0.5, alpha=0.8)
ax1.bar(x[1], b1, bottom=a1, label='Individual effect', color=bar_colors[0], width=height, edgecolor='black', linewidth=0.5)

plt.grid(False)
sns.despine(right=True)

x_left, x_right = 0, 2
y_low, y_high = 0, 0.95
ratio = 1.2
ax1.set_xlim([x_left, x_right])
ax1.set_ylim([y_low, y_high])
ax1.set_xticks([0.5, 1.5])
ax1.set_yticks(np.arange(0, 0.81, 0.2))
ax1.set_aspect(abs((x_right - x_left) / (y_low - y_high)) * ratio)

fontsize = 24
ax1.set_xlabel("", fontsize=fontsize, fontname='Arial')
ax1.set_ylabel("Structure-function coupling (r)", fontsize=fontsize, fontname='Arial')
ax1.set_xticklabels([ "HCP-YA", "HCP-D"],rotation=0)
ax1.tick_params(bottom=True, left=True, width=2, length=6)

fig.subplots_adjust(
    top=0.981,
    bottom=0.12,
    left=0.3,
    right=0.9,
    hspace=0.2,
    wspace=0.2
)

for tick in ax1.xaxis.get_major_ticks():
    tick.label.set_fontsize(fontsize)
    tick.label.set_fontname('Arial')
for tick in ax1.yaxis.get_major_ticks():
    tick.label.set_fontsize(fontsize)
    tick.label.set_fontname('Arial')
for _, s in ax1.spines.items():
    s.set_color('black')
    s.set_linewidth(2)
plt.savefig("/Users/chenpeiyu/PycharmProjects/SC_FC_Pred/matlab/step_03_whole_brain_group_ind_effects/pFC_eFC_group_ind_schaefer200.png", dpi=300, bbox_inches = 'tight')
plt.show()
