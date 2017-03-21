import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

xedges = [0, 1, 1.5, 3, 5]
yedges = [0, 2, 3, 4, 6]

x = np.random.normal(3, 1, 100)
y = np.random.normal(1, 1, 100)

H, xedges, yedges = np.histogram2d(x, y)
plt.imshow(H)
plt.show()
