import sys

import matplotlib.pyplot as plt

x = []
y = []


for line in sys.stdin:
    tokens = [float(x.strip()) for x in line.strip().split("\t")]
    x.append(tokens[0])
    y.append(tokens[1])

plt.scatter(x, y)
plt.show()
