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
scale = 8
nmodes = int(32*scale)
L = 2*np.pi*scale
Nx, Ny, Nz = nmodes, nmodes, nmodes
amp_noise = 1e-3 #initial noise level


#Primary wave
theta_pw=10/180*np.pi#np.arccos(kz/k)   # Angle between \Omega and \vec{k}
A=0.3                      # Amplitude
s=1                         # Helicity
kbox = 2*np.pi/L*nmodes/2
viscfactor = 0.75
Ek = viscfactor*A/kbox**2

#k=1
#L=2*np.pi*scale/np.sin(theta_pw)


stop_sim_time = 300.0
num_snapshots = 20
num_slices = 100

timestepper = d3.RK443
#timestepper = d3.SBDF2
max_timestep = 0.1/5.0/A
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
Cosz = dist.Field(name='Cosz', bases=zbasis)
Sinz = dist.Field(name='Sinz', bases=zbasis)

# Substitutions
x,y,z = dist.local_grids(xbasis,ybasis,zbasis)
ex,ey,ez = coords.unit_vector_fields(dist)

Omega=-np.tan(theta_pw)*ex+ez
U['g'][0] = A*np.cos(z)
U['g'][1] = s*A*np.sin(z)
U['g'][2] = -s

Cosz['g'] = np.cos(z)
Sinz['g'] = np.sin(z)

logger.info('------------------INFO------------------')
logger.info('sim_time         = %0.2e' %(stop_sim_time))
logger.info('Box scale factor = %0.2f' %(scale))
logger.info('A                = %0.2f' %(A))
logger.info('theta            = %0.2f' %(theta_pw*180/np.pi))
logger.info('Ekman            = %0.2e' %(Ek))
logger.info('---------------------------------------')

# Problem
problem = d3.IVP([u, p, tau_p], namespace=locals())
problem.add_equation("dt(u) + grad(p) - Ek*lap(u) + cross(Omega, u) = -u@grad(u) - cross(Omega, ez)")
#problem.add_equation("dt(u) + grad(p) - Ek*lap(u)  = -u@grad(u) - cross(Omega,u)")
problem.add_equation("div(u) + tau_p = 0")
problem.add_equation("integ(p) = 0") # Pressure gauge


# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
u.fill_random('g', seed=42, distribution='uniform') # Random noise, this doesn't satisfy Div(A)=0 so need to change it! but as a bad hack...
u['g']*=amp_noise
u['g'][0]+=U['g'][0]
u['g'][1]+=U['g'][1]
u['g'][2]+=U['g'][2]

# Analysis

slices = solver.evaluator.add_file_handler('slices', sim_dt=stop_sim_time/num_slices, max_writes=100)
slices.add_task((ex@u)(x=0),layout='g', name='ux_YZ')
slices.add_task((ey@u)(x=0),layout='g', name='uy_YZ')
slices.add_task((ez@u)(x=0),layout='g', name='uz_YZ')
slices.add_task((ex@u)(y=0),layout='g', name='ux_XZ')
slices.add_task((ey@u)(y=0),layout='g', name='uy_XZ')
slices.add_task((ez@u)(y=0),layout='g', name='uz_XZ')
slices.add_task((ex@u)(z=0),layout='g', name='ux_XY')
slices.add_task((ey@u)(z=0),layout='g', name='uy_XY')
slices.add_task((ez@u)(z=0),layout='g', name='uz_XY')
slices.add_task(d3.Average(ex@u,coords['z']),layout='g', name='ux_gs')
slices.add_task(d3.Average(ey@u,coords['z']),layout='g', name='uy_gs')
slices.add_task(d3.Average(ez@u,coords['z']),layout='g', name='uz_gs')
slices.add_task(d3.Average(ez@(d3.curl(u)),coords['z']),layout='g', name='omegaz_gs')
slices.add_task(ez@(d3.curl(u))(z=0),layout='g', name='omegaz_XY')
slices.add_task(ey@(d3.curl(u))(y=0),layout='g', name='omegay_XZ')
slices.add_task(ex@(d3.curl(u))(x=0),layout='g', name='omegax_YZ')

snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=stop_sim_time/num_snapshots, max_writes=10000)
snapshots.add_task((ex@u),layout='g', name='ux')
snapshots.add_task((ey@u),layout='g', name='uy')
snapshots.add_task((ez@u),layout='g', name='uz')
snapshots.add_task((Omega@d3.curl(u))/(Omega@Omega)**0.5,layout='g', name='omegaz')

series = solver.evaluator.add_file_handler('series', sim_dt=stop_sim_time/2e3)
series.add_task(d3.Average(u@u), name='Urmssq')
series.add_task(d3.Average((u@ex)**2), name='Urmssq_x')
series.add_task(d3.Average((u@ey)**2), name='Urmssq_y')
series.add_task(d3.Average((u@ez)**2), name='Urmssq_z')
series.add_task(d3.Average((u@ez))**2, name='Uz_avg_sq')
series.add_task(2*d3.Average((u@ex)*Cosz)**2, name='Usq_IW_x')
series.add_task(2*d3.Average((u@ey)*Sinz)**2, name='Usq_IW_y')
series.add_task(d3.Average(Ek*d3.curl(u)@d3.curl(u)), name='nu_omegasq')
series.add_task(-d3.Average(u@(u@d3.grad(U)+ U@d3.grad(u))), name='P_in')
series.add_task(d3.Average( d3.Average(u@ex,coords['z'])**2 ), name='Ux_rmssq_gs')
series.add_task(d3.Average( d3.Average(u@ey,coords['z'])**2 ), name='Uy_rmssq_gs')
series.add_task(d3.Average( d3.Average(u@ez,coords['z'])**2 ), name='Uz_rmssq_gs')


# CFL
CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.5, threshold=0.1,
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

