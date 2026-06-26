"""
Plot cutaway ball outputs.

Usage:
    plot_ball.py <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames]

"""

import h5py
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
        



def main(filename, start, count, output):
    """Save plot of specified tasks for given range of analysis writes."""
    dpi = 500
    figsize = (8, 8)
    savename_func = lambda write: 'write_{:06}.png'.format(write)
    skip=50
    MHD_ON=1
    colormap=plt.cm.RdBu_r
    fntsz=28
    lnwdth=4
    # chi=1e0
    # mu_eq=4*np.pi*32
    # lambda_mu=mu_eq**2
    # eta=np.sqrt(chi)/mu_eq
    # t_gamma=4/eta/mu_eq**2
    t_gamma=1.0
    # gamma_mu=1/t_gamma
    # R=2e-2#Gamma_f/gamma_mu
    # tgamma_mu_catchup=1/R**(2.0/3.0)*np.log(10**3)
    # print(tgamma_mu_catchup)
    with h5py.File(filename, mode='r') as file:
        print(file)
        print(file.keys())
        print(file['scales'].keys())
        print(file['tasks'].keys())

        Usq_IW = file['tasks']['Usq_IW']
        Urmssq = file['tasks']['Urmssq']
        Urmssq_x = file['tasks']['Urmssq_x']
        Urmssq_y = file['tasks']['Urmssq_y']
        Urmssq_z = file['tasks']['Urmssq_z']
        Ux_rmssq_gs = file['tasks']['Ux_rmssq_gs']
        Uy_rmssq_gs = file['tasks']['Uy_rmssq_gs']
        Uz_rmssq_gs = file['tasks']['Uz_rmssq_gs']
        U_rmssq_gs = Ux_rmssq_gs[:,0,0,0]+Uy_rmssq_gs[:,0,0,0]+Uz_rmssq_gs[:,0,0,0]


        t = Urmssq.dims[0]['sim_time']
        t=np.asarray(t[:])
        
        fig = plt.figure(figsize=figsize)
        plt.plot(t[:], 0.5*Urmssq[:,0,0,0], label='$E_{\\rm total}$',linewidth=lnwdth)
        plt.plot(t[:], 0.5*Usq_IW[:,0,0,0], label='$E_{\\rm IW}$',linewidth=lnwdth)
        plt.plot(t[:], 0.5*U_rmssq_gs[:], label='$E_{\\rm g.s.}$',linewidth=lnwdth)
        plt.legend(fontsize=fntsz)
        #plt.yscale('log')
        #plt.ylim(bottom=1e-6)
        ax=plt.gca()
        ax.tick_params(axis='both', which='major',  width=1.5, length=10)
        ax.tick_params(axis='both', which='minor',  width=0.75, length=5)

        plt.xlabel("Time, $\\omega t$",fontsize=fntsz)
        plt.ylabel("Energy",fontsize=fntsz)
        plt.tick_params(labelsize=fntsz-4)
        fig.savefig('frames/AAA_timeseries.png', dpi=dpi)
        plt.show()
        plt.close()

if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    args = docopt(__doc__)

    output_path = pathlib.Path(args['--output']).absolute()
    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
    post.visit_writes(args['<files>'], main, output=output_path)

