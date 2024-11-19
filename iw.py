"""
To run and plot using e.g. 4 processes:
    $ mpiexec -n 4 python iw.py
    $ mpiexec -n 1 python save_alphabeta_data.py snapshots/snapshots_s1.h5
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
nmodes = 32  # 16
scale  = 2    # 1
L = 2*np.pi*scale
Nx, Ny, Nz = nmodes, nmodes, nmodes
amp_noise = 1e-4 #initial noise level

#Primary wave
Ek = 1e-6                     # Ekman number 
theta_pw = 10.0/180.0*np.pi   # Angle between \Omega and \vec{k}
A = 0.2                       # Amplitude
s = 1                         # Helicity

stop_sim_time = 200.0
num_snapshots = 200
num_slices = 25

timestepper = d3.RK443
# timestepper = d3.SBDF2
max_timestep = 0.1*A/40
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
U['g'][0]=A*np.cos(z)
U['g'][1]=s*A*np.sin(z)
U['g'][2]=-s

logger.info('------------------INFO------------------')
logger.info('sim_time         = %0.2e' %(stop_sim_time))
logger.info('Box scale factor = %0.2f' %(scale))
logger.info('A                = %0.2f' %(A))
logger.info('theta            = %0.2f' %(theta_pw*180/np.pi))
logger.info('Ekman            = %0.2e' %(Ek))
logger.info('---------------------------------------')

# Problem
problem = d3.IVP([u, p, tau_p], namespace=locals())
problem.add_equation("dt(u) + grad(p)   = - u@grad(U)- U@grad(u)-cross(Omega,u)")# - Ek*lap(u)
problem.add_equation("div(u) + tau_p = 0")
problem.add_equation("integ(p) = 0") # Pressure gauge


# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
u.fill_random('g', seed=43, distribution='normal', scale=amp_noise) # Random noise, this doesn't satisfy Div(A)=0 so need to change it! but as a bad hack...

# for i in range(nmodes):
#     amp=np.random.random()
#     u['g'][0]+=amp_noise*np.sin(x)*amp*np.cos(i*z) 
#     u['g'][1]+=amp_noise*np.cos(x)*amp*np.cos(i*z) 
#     u['g'][2]+=amp_noise*np.cos(x)*amp*np.sin(i*z) 

# Analysis

snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=stop_sim_time/num_snapshots, max_writes=1000)
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

