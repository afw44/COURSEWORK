import numpy as np
import matplotlib.pyplot as plt

x1,y1 = zip(*[[int(q[0]), int(q[1])] if len(q)>1 else [000,431]
        for q in [list(filter(lambda y: y.isdigit(), x.split()))
                  for x in open("plot1.txt", "r")] ])

x2, y2, z2 = zip(*[[int(q[0]), int(q[1]), int(q[2])] if len(q)>2 else [000, 431, 561]
        for q in [tuple(filter(lambda y: y.isdigit(), x.split()))
                  for x in open("plot2.txt", "r")]])

x1arr = np.array(x1)
y1arr = np.array(y1)
x1arr = (x1arr.max() - x1arr)[:-3]
y1arr = (y1arr.max() - y1arr)[:-3]

x2arr = np.array(x2)
y2arr = np.array(y2)
z2arr = np.array(z2)
x2arr = (x2arr.max() - x2arr)[:-3]
y2arr = (y2arr.max() - y2arr)[:-3]
z2arr = (z2arr.max() - z2arr)[:-3]

fig, axs = plt.subplots(2,1)


axs[0].set_xlabel('Frequency (GHz)')
axs[0].set_ylabel('Voltage (mV)')
axs[0].plot(np.linspace(0,x1arr.shape[0],x1arr.shape[0])/100,x1arr)
axs[0].plot(np.linspace(0,y1arr.shape[0],y1arr.shape[0])/100,y1arr)


axs[1].set_xlabel('Frequency (MHz)')
axs[1].set_ylabel('Voltage (mV)')
axs[1].plot(np.linspace(0,x2arr.shape[0],x2arr.shape[0])/100,x2arr)
axs[1].plot(np.linspace(0,y2arr.shape[0],y2arr.shape[0])/100,y2arr)
axs[1].plot(np.linspace(0,z2arr.shape[0],z2arr.shape[0])/100,z2arr)

fig.tight_layout()
fig.savefig('plot.png')
plt.show()