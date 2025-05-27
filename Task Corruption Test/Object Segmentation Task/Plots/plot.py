# Imports
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors as mcolors
import matplotlib as mpl

mpl.rcParams['font.serif'] = 'Palatino'
mpl.rcParams['text.usetex'] = 'true'
mpl.rcParams['text.latex.preamble'] = r'\usepackage{newtxmath}'
mpl.rcParams['font.size'] = 22
#mpl.rcParams['axes.prop_cycle'] = plt.cycler(color=plt.cm.Set3.colors)


# Change Params Here:
file_string = 'instance_segmentation_map50'
rows_as_group = ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90']
columns_as_bars = ["YOLOv3", "YOLOv8"]
#ytick_label = ['0.0', '0.2', '0.4', '0.6', '0.7']


data_file =  file_string + '.txt'
fig_pdf_file = file_string + '.pdf'
fig_eps_file = file_string + '.eps'


data = np.loadtxt(data_file)
data = data[:, 1:]

column_1 = data[:, 0]
column_2 = data[:, 1]


x = np.arange(len(rows_as_group))
width = 0.3
fig, ax = plt.subplots()
fig.set_figheight(5.5)
fig.set_figwidth(12)

rects1 = ax.bar(x - width, column_1, width, color='lavender', hatch='/', edgecolor="black", linewidth=2)
rects2 = ax.bar(x, column_2, width, color='lightcyan', hatch=' \ ', edgecolor="black", linewidth=2)
#rects3 = ax.bar(x + width, column_3, width, color='lavenderblush', hatch='//', edgecolor="black", linewidth=2)


ax.set_ylabel('mAP$@$0.5', fontsize=35)
ax.set_xlabel("Percent packet loss in each image", fontsize=35)
#ax.set_title('Scores by group and gender', fontsize= 25)
ax.set_yticks([0.0,  0.2, 0.4, 0.6, 0.8], [0.0,  0.2, 0.4, 0.6, 0.8], fontsize=35)

ax.set_xticks(x)
ax.set_xticklabels(rows_as_group, fontsize=35)


ax.set_ylim(0.0, 0.9)
#ax.set_yticklabels(ytick_label, fontsize=30)


ax.legend([rects1, rects2], columns_as_bars, loc='upper center', ncol=2, fontsize=30)
plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.title('mAP$@$0.5 performance of instance segmentation', fontsize=35)

plt.tight_layout()

plt.savefig(fig_pdf_file, dpi=300, format='pdf')
plt.savefig(fig_eps_file, dpi=300, format='eps')

plt.show()
print('1')