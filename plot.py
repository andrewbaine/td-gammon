import sys

import matplotlib.pyplot as plt

x = []
y = []


for line in sys.stdin:
    tokens = [x.strip() for x in line.strip().split()]
    x.append(int(tokens[0].split(".")[1]))
    y.append(float(tokens[1]))

plt.scatter(x, y)
plt.show()
