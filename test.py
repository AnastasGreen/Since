from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from colorsys import hls_to_rgb

lamb = 0.000532
k = (2 * np.pi) / lamb

alpha = 0.00001
gamma = 1
m = 2

N = 100
a = 1
z = np.zeros((N, N))
h = 2 * a / N

x = np.linspace(-a, a, N)
y = np.linspace(-a, a, N)

x1, y1 = np.meshgrid(x, y)

def r(x: float, y: float) -> float:
    return np.sqrt(np.power(x, 2) + np.power(y, 2))


def phi(x: float, y: float) -> float:
    return np.arctan2(y, x)


# с угловой модуляцией, моя функция
def z(r: float, phi: float) -> float:
    return np.sin(np.power(alpha * k * r, gamma) * m * phi)


def first_exp(r, phi):
    return np.exp(complex(0, 1) * np.power(alpha * k * r, gamma) * m * phi)


def sec_exp(r, phi):
    return np.exp(complex(0, -1) * np.power(alpha * k * r, gamma) * m * phi)


# синус
def f_sin1(r: float, phi: float) -> complex:
    return (first_exp(r, phi) - sec_exp(r, phi)) / (2 * complex(0, 1))


# две экспоненты
def f_exp1(r: float, phi: float) -> complex:
    return first_exp(r, phi) / (2 * complex(0, 1))


def f_exp2(r: float, phi: float) -> complex:
    return sec_exp(r, phi) / (2 * complex(0, 1))


def compose(x):
    return np.abs(np.fft.fftshift(np.fft.fft2(x)))

#эту
def phaza(r, phi):
    return (np.power(alpha * k * r, gamma)* m * phi)



r_vals = r(x[:, None], y[None, :])
phi_vals = phi(x[:, None], y[None, :])

z_vals = np.zeros((N, N))
z_sin = np.zeros((N, N))
z_exp1 = np.zeros((N, N))
z_exp2 = np.zeros((N, N))
z_v = np.zeros((N, N))

for i in range(N):
    for j in range(N):
        if phi_vals[i, j] < 0:
            phi_vals[i, j] += 2 * np.pi
        if r_vals[i, j] < 2:
            z_exp1[i, j] = f_exp1(r_vals[i, j], phi_vals[i, j]).real
            z_exp2[i, j] = f_exp2(r_vals[i, j], phi_vals[i, j]).real
            z_vals[i, j] = z(r_vals[i, j], phi_vals[i, j])
            z_sin[i, j] = f_sin1(r_vals[i, j], phi_vals[i, j]).real
            # посмотреть только функцию phaza 
            z_v[i, j] = phaza(r_vals[i, j], phi_vals[i, j])




# exp_1 = compose(z_exp1)
# plt.imshow(exp_1)
# # plt.show()
#
# exp_2 = compose(z_exp2)
# plt.imshow(exp_2)
# # plt.show()
#
# ft1 = compose(z_sin)
# plt.imshow(ft1)
# # plt.show()
#


ft = compose(z_vals)
# plt.imshow(ft)
# plt.imshow(z_vals)
# plt.show()

#построение графика


fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(x1, y1, z_v)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')


plt.show()

