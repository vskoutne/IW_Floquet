"""
Dedalus script simulating a 2D periodic incompressible shear flow with a passive
tracer field for visualization. This script demonstrates solving a 2D periodic
initial value problem. It can be ran serially or in parallel, and uses the
built-in analysis framework to save data snapshots to HDF5 files. The
`plot_snapshots.py` script can be used to produce plots from the saved data.
The simulation should take about 10 cpu-minutes to run.

The initial flow is in the x-direction and depends only on z. The problem is
non-dimensionalized usign the shear-layer spacing and velocity jump, so the
resulting viscosity and tracer diffusivity are related to the Reynolds and
Schmidt numbers as:

    nu = 1 / Reynolds
    D = nu / Schmidt

To run and plot using e.g. 4 processes:
    $ mpiexec -n 4 python3 shear_flow.py
    $ mpiexec -n 4 python3 plot_snapshots.py snapshots/*.h5
"""

import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)
from mpi4py import MPI
ncpu = MPI.COMM_WORLD.size

log2 = np.log2(ncpu)
if log2 == int(log2):
    mesh = [int(2**np.ceil(log2/2)),int(2**np.floor(log2/2))]
    logger.info("running on processor mesh={}".format(mesh))
# mesh=[2,2]


# Parameters
nmodes = 32
scale = 1
L = 2*np.pi*scale
Nx, Ny, Nz = nmodes, nmodes, nmodes
amp_noise = 1e-4 #initial noise level


#Primary wave
Ek=1e-6                     # Ekman number 
theta_pw=10.0/180.0*np.pi   # Angle between \Omega and \vec{k}
A=0.2                       # Amplitude
s=1                         # Helicity

stop_sim_time = 100.0
num_snapshots = 100
num_slices=25

timestepper = d3.RK222
max_timestep = 1e-2
dtype = np.float64
dealias = 3/2

# Bases
coords = d3.CartesianCoordinates('x', 'y', 'z')
dist = d3.Distributor(coords, dtype=dtype, mesh=mesh)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, L), dealias=dealias)
ybasis = d3.RealFourier(coords['y'], size=Nx, bounds=(0, L), dealias=dealias)
zbasis = d3.RealFourier(coords['z'], size=Nz, bounds=(0, L), dealias=dealias)

# Fields
p = dist.Field(name='p', bases=(xbasis,ybasis,zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis,ybasis,zbasis))
U = dist.VectorField(coords, name='U', bases=(xbasis,ybasis,zbasis))
tau_p = dist.Field(name='tau_p')

# Substitutions
x,y,z = dist.local_grids(xbasis,ybasis,zbasis)
ex,ey,ez = coords.unit_vector_fields(dist)

Omega=-np.tan(theta_pw)*ex+ez
U['g'][0]=A*np.cos(x)
U['g'][1]=s*A*np.sin(x)
U['g'][2]=-s

logger.info('------------------INFO------------------')
logger.info('sim_time         = %0.2e' %(stop_sim_time))
logger.info('Box scale factor = %0.2f' %(scale))
logger.info('A                = %0.2f' %(A))
logger.info('Ekman            = %0.2e' %(Ek))
logger.info('---------------------------------------')

# Problem
problem = d3.IVP([u, p, tau_p], namespace=locals())
problem.add_equation("dt(u) + grad(p)   - Ek*lap(u) = - u@grad(U)- U@grad(u)-cross(Omega,u)")
problem.add_equation("div(u) + tau_p = 0")
problem.add_equation("integ(p) = 0") # Pressure gauge


# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
u.fill_random('g', seed=42, distribution='normal', scale=amp_noise) # Random noise, this doesn't satisfy Div(A)=0 so need to change it! but as a bad hack...

# Analysis

snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=stop_sim_time/num_snapshots, max_writes=100)
snapshots.add_task((ex@u),layout='g', name='ux')
snapshots.add_task((ey@u),layout='g', name='uy')
snapshots.add_task((ez@u),layout='g', name='uz')

series = solver.evaluator.add_file_handler('series', sim_dt=stop_sim_time/2e3)
series.add_task(d3.Average(u@u), name='Urmssq')

slices = solver.evaluator.add_file_handler('slices', sim_dt=stop_sim_time/num_slices, max_writes=100)
slices.add_task((u@ex)(z=0), name='ux_XY')


# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.25, threshold=0.1,
             max_change=1.5, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(u)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property((u@u)**2, name='u_sq')

# Main loop
try:
    logger.info('Starting main loop, stop_sim_time=%0.2f'%(stop_sim_time))
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            max_u = np.sqrt(flow.max('u_sq'))
            logger.info('Iteration=%i, Time=%0.3e, dt=%0.3e, max(u)=%0.3e,' %(solver.iteration, solver.sim_time, timestep, max_u))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

