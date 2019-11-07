# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
pt,eta = 38., -.0
a = -pt*np.sinh(eta)
b = pt*np.cosh(eta)
c = -np.sqrt(b**2-a**2)
Estar = 40. 

X = np.arange(-0.5, 3, 0.01)
Y = np.arange(-10, 10, 0.01)
X, Y = np.meshgrid(X, Y)
R =  ((Estar - c*X - a*Y)/b)**2 - 1 - X**2 - Y**2
R = R.clip(0)
Z = np.sqrt( R )

# Plot the surface.
surf = ax.plot_surface(X, Y, -Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

surf = ax.plot_surface(X, Y, +Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)


# Customize the z axis.
ax.set_zlim(-10, +10)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.01f'))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.01f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.01f'))
plt.ylabel(r'$\gamma\beta_{z}$')
plt.xlabel(r'$\gamma\beta_{x}$')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.savefig('3D.png')

plt.draw()
ax.view_init(30, -45) 
plt.pause(10)

#for angle in range(0, 360):
#    ax.view_init(30, angle)
