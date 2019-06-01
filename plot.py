from glob import glob
import os
import matplotlib.pyplot as plt

data = []
for file in glob('./results/*.txt'):
    filename = os.path.basename(file)[:-4].split('_')
    density = float(filename[2])
    query = float(filename[5])
    if query != 0.6:
        continue
    with open(file) as f:
        times = f.readlines()[0].split(', ')
        us = float(times[0])
        theirs = float(times[1])
        data.append((density, us, theirs))

data = sorted(data, key=lambda x: x[0])

x = [d[0] for d in data]
y1 = [d[1] for d in data]
y2 = [d[2] for d in data]
print(x)

plt.plot(x, y1, '-o', label='Ours')
plt.plot(x, y2, '-o', label='Baseline')
plt.legend()
plt.title('Running Time vs. Edge Density')
plt.xlabel('Edge Density')
plt.ylabel('Workflow Time (s)')
plt.show()