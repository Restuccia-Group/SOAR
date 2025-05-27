import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV files
files = ['yolov8_loss_0.csv', 'yolov8_loss_10.csv', 'yolov8_loss_20.csv', 'yolov8_loss_30.csv', 'yolov8_loss_40.csv', 'yolov8_loss_50.csv', 'yolov8_loss_60.csv', 'yolov8_loss_70.csv', 'yolov8_loss_80.csv', 'yolov8_loss_90.csv']
fig_name = 'yolov8_objectPR'


fig_name_png = 'yolov8_objectPR' + '.png'
fig_name_pdf = 'yolov8_objectPR' + '.pdf'

xticks = [0.2, 0.4, 0.6, 0.8, 1.0]
yticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


datasets = [pd.read_csv(file) for file in files]


# Define padding values
padding_values = [((0, 1), (1, 0))]

# Apply padding to each dataset
for data in datasets:
    data.loc[0] = padding_values[0][0]
    data.loc[-1] = padding_values[0][1]


# Create a plot
plt.figure(figsize=(13, 12))

for idx, data in enumerate(datasets):
    x = data.iloc[:, 0]
    y = data.iloc[:, 1]
    plt.plot(x, y, label=f'Packet {files[idx][7:-4]}%', linewidth=3)

plt.xlim(0, 1)
plt.ylim(0, 1)

plt.xlabel('Recall', fontsize=45)
plt.ylabel('Precision', fontsize=45)  # Increase legend font size

plt.xticks(xticks, fontsize=35)
plt.yticks(yticks, fontsize=35)

plt.title('Precision-Recall plots with YOLOv8', fontsize=45)

plt.grid(True)  # Add grid

# Add legend with custom font size
legend = plt.legend(fontsize=35)

plt.tight_layout()
plt.savefig(fig_name_png, dpi=300, format='png')
plt.savefig(fig_name_pdf, dpi=300, format='pdf')

plt.show()
print('1')
