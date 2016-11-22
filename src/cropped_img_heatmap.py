import numpy as np
import numpy.random
import matplotlib.pyplot as plt
import os
from PIL import Image

# the DOL_cropped img dimensions
x = []
y = []

# Load data
train_folder_path = "../input/train";
cropped_folders = []

for label_folder in os.listdir(train_folder_path):
    label_folder_path = train_folder_path + "/" + label_folder
    for cropped_folder in os.listdir(label_folder_path):
        if cropped_folder.lower() == "DOL_cropped":
            cropped_folders.append(label_folder_path + "/" + cropped_folder)

for cropped_folder in cropped_folders:
    for cropped_file in os.listdir(cropped_folder):
        cropped_file_path = cropped_folder + "/" + cropped_file
        im = Image.open(cropped_file_path)
        width, height = im.size
        x.append(width)
        y.append(height)

x = np.array(x)
y = np.array(y)

heatmap, xedges, yedges = np.histogram2d(x, y, bins=25)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

plt.clf()
plt.imshow(heatmap.T, extent=extent, origin='lower')
plt.show()
