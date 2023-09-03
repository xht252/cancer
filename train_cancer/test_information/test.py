from loader_picture import data_loader
import numpy as np
from matplotlib import pyplot as plt

load = data_loader.Loader()
w = 60
h = 40
fig = plt.figure(figsize=(15, 15))
columns = 4
rows = 3

for i in range(1, columns*rows +1):
    ax = fig.add_subplot(rows, columns, i)
    if np.argmax(load.y_train[i]) == 0:
        ax.title.set_text('Benign')
    else:
        ax.title.set_text('Malignant')
    plt.imshow(load.train_of_x[i], interpolation='nearest')
plt.show()