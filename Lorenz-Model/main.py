import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from integrator import solver, update_frame

"""
Parameters
sigma :     Prandtl number
beta :      geometric factor
rho :       Rayleigh number

Simulation parameters
N :         number of simulations
t_init :    start time
t_end :     end time
k :         number of time points
tvec :      array of time points
case :      switch for initial conditions
"""

# Parameters (these specific values are from Wikipedia)
sigma   = 10
beta    = 8/3
rho     = 28

# Simulation parameters
N       = 50
t_init  = 0
t_end   = 20
k       = 1501
tvec    = np.linspace(t_init, t_end, k)

# Enable animation?
enable_animation = True



# Pack the parameters
lorenz_pars = [sigma, beta, rho]
sim_pars = [N, t_init, t_end, k]

# Initial conditions
case = 1

if case == 1:
    # Generates initial conditions from random numbers [-10,10)
    IC = 10 * (2 * np.random.rand(N, 3) - 1)
    print('Initial conditions: ', IC)

else:
    # Generates initial conditions close to each other 
    ic = 10 * (2 * np.random.rand(N, 3) - 1)
    IC = [ic]*N
    perturbation = 0.001*(2* np.random.rand(N,1) - 1)

    IC = IC + perturbation
    print('Initial conditions: ', IC)

# Numerical solution
sol = solver(N, tvec, sigma, beta, rho, t_init, t_end, IC)

# Plot setup
fig = plt.figure()
axis = fig.add_subplot(projection='3d')

axis.set_xlabel('x-axis')
axis.set_xlabel('y-axis')
axis.set_xlabel('z-axis')
axis.set_title(f'Lorenz Model: {N} Simulations')

lorenz_plots = []

for i in range(N):
    lorenz_plt, = axis.plot(sol[i][1], sol[i][2], sol[i][3],
                           lw=2, alpha=0.8) # Transparency
    lorenz_plots.append(lorenz_plt)

# Viewing angle
axis.view_init(elev=30, azim=120)

# Animation
if enable_animation:
    animation = FuncAnimation(
        fig, update_frame, frames=len(tvec),
        interval=25, blit=False,
        fargs=(N, sol, lorenz_plots, axis)
    )

plt.show()