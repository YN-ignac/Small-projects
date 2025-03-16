from scipy.integrate import solve_ivp
import numpy as np

# Define system of ODEs
def ode_system(t: float, S: list[float], sigma: float, beta: float, rho: float) -> list[float]:
    """
    Defines system of ODEs (Lorenz model)

    Input
    t (float):              current time
    S (list of float) :     state variables [x y z] in time t
    sigma (float):          Prandtl number
    beta (float):           geometric factor
    rho (float):            Rayleigh number

    Output
    list of float : derivatives (dxdt dydt dzdt)
    """

    # Unpack state variables
    x, y, z = S

    # Model equations
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z

    return [dxdt, dydt, dzdt]

# Solver

# n_simulations, time_points, sigma, beta, rho, t_begin, t_end, initial_conditions
def solver(N: int, tvec: np.ndarray, sigma: float, beta: float, rho: float, 
           t_init: float, t_end: float, initial_conditions: np.ndarray) -> np.ndarray:
    """
    Solves the system of ODEs using scipy solve_ivp

    Input
    N (intiger) :                   number of simulations
    tvec (intiger) :                time array
    sigma, rho, beta (float) :      Lorenz model parameters
    t_init, t_end (float) :         intial and end time of the simulaton
    initial_conditions (ndarray) :  array of initial conditions  

    Output
    ndarray : array of solutions [t x y z] for each simulation
    """

    # Initialize solution array
    solutions = [] * N

    for i in range(N):
        solution = solve_ivp(
            fun=ode_system, t_span=(t_init, t_end),
            y0=initial_conditions[i], t_eval=tvec,
            args=(sigma, beta, rho)
        )
    
        # Solution in current time t
        t_points = solution.t
        x_sol = solution.y[0]
        y_sol = solution.y[1]
        z_sol = solution.y[2]

        # Pack the solution
        solutions.append(np.array([t_points, x_sol, y_sol, z_sol]))


    return np.array(solutions, dtype=np.float32)

def update_frame(frame: int, n_simulations: int, solution_array: np.ndarray, lorenz_plots: list, ax) -> list:
    """
    Updates the 3D plot of every frame in the simulation

    Input
    frame (integer) :           current frame
    n_simulations (float) :     number of simulations
    solution_array (array) :    array of all solutions
    lorenz_plots (list) :       list of plot objects for each simulation
    ax :                        plot axis (3D)

    Output
    updated plot
    """

    lower_lim = max(0, frame - 200)

    for i in range(n_simulations):
        # Update the plot for each simulation
        x_val = solution_array[i, 1][lower_lim:frame+1]
        y_val = solution_array[i, 2][lower_lim:frame+1]
        z_val = solution_array[i, 3][lower_lim:frame+1]

        lorenz_plots[i].set_data(x_val, y_val)
        lorenz_plots[i].set_3d_properties(z_val)

    # Adjust the viewpoint with each frame
    ax.view_init(elev=10, azim=0.25*frame)
    ax.set_title(f"Lorenz Model: {n_simulations} Simulation | Progress: {(frame+1)/len(solution_array[0, 1]):.1%}")

    return lorenz_plots


    
