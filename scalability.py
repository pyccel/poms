# -*- coding: UTF-8 -*-
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Data for plotting
proc = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

# Assembly
tass = [222, 111, 56, 33, 15, 7.6, 3.6, 1.6, 0.7, 0.26, 0.1]

# Smoothers
tpre = [408, 234, 112, 60, 30, 24, 8.8, 4.7, 2.6, 1.6, 1.2]
tjac = [20, 11.5, 5.8, 3.0, 1.53, 0.87, 0.43, 0.4, 0.4, 0.40, 0.4]

# GLT
tg2 = [3.12, 2.2, 0.99, 0.5, 0.3, 0.27, 0.18, 0.15, 0.14, 0.14, 0.17]
tg3 = [4.04, 2.4, 1.21, 0.67, 0.3, 0.3, 0.25, 0.2, 0.20, 0.18, 0.14]
tg4 = [5, 3.05, 1.6, 0.84, 0.48, 0.35, 0.25, 0.22, 0.20, 0.18, 0.18]

fig, ax = plt.subplots()

#ax.semilogx(proc, tass, 'g-')
#ax.semilogx(proc, tpre)
#ax.semilogx(proc, tjac, label="$PCG + Jacobi$")
#ax.semilogx(proc, tg3, label="$PCG + GLT$")

ax.semilogx(proc, tg2, label="$p=2$")
ax.semilogx(proc, tg3, label="$p=3$")
ax.semilogx(proc, tg4, label="$p=4$")

ax.set(xlabel='Number of Processors', ylabel='Execution time (s)')
ax.legend()
ax.grid()

fig.savefig("test.png")
plt.show()

