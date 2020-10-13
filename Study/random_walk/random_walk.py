import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure()

# time span
T = 2

mu = 0.1
sigma = 0.4
S0 = 20
dt = 0.01
N = round(T / dt)
t = np.linspace(0, T, N)

# 布朗运动
W = np.random.standard_normal(size=N)
W = np.cumsum(W) * np.sqrt(dt)

X = (mu - 0.5 * sigma ** 2) * t + sigma * W

S = S0 * np.exp(X)

plt.plot(t, S, lw=2)
plt.show()


