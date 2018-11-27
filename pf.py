import numpy as np
import matplotlib.pyplot as plt

# randomy generate particles

# predict next state of the particles

# update based on measurements

# resample every step

# compute estimate

z_t = []
N_iter = 50
N_meas = 10
m_t = []

z_t.append(0.1)
m  = []
for _ in range(N_meas):
    m.append(z_t[-1]**2/20 + np.random.normal(0, np.sqrt(1)))
m_t.append(m)

for i in range(1, N_iter):
    z_t.append(0.5*z_t[-1] + 25*z_t[-1]/(1+z_t[-1]**2) + 8*np.cos(1.2*(i-1)) + np.random.normal(0, np.sqrt(10)))
    m = []
    for _ in range(N_meas):
        m.append(z_t[-1]**2/20 + np.random.normal(0, np.sqrt(1)))
    m_t.append(m)

plt.figure(1)
for t in range(N_iter):
    for i in range(N_meas):
        plt.plot(t, m_t[t][i], 'ro')
plt.plot(range(N_iter), z_t, 'g--', label='$z(t)$')
plt.xlabel('t')
plt.legend()
plt.show()
