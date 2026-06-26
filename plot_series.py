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
    fntsz=18
    lnwdth=3
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

        Urmssq = file['tasks']['Urmssq']
        Urmssq_x = file['tasks']['Urmssq_x']
        Urmssq_y = file['tasks']['Urmssq_y']
        Urmssq_z = file['tasks']['Urmssq_z']
        Ux_rmssq_gs = file['tasks']['Ux_rmssq_gs']
        Uy_rmssq_gs = file['tasks']['Uy_rmssq_gs']
        Uz_rmssq_gs = file['tasks']['Uz_rmssq_gs']
        P_in = file['tasks']['P_in']
        nu_omegasq = file['tasks']['nu_omegasq']
        t = Urmssq.dims[0]['sim_time']
        t=np.asarray(t[:])
        
        fig = plt.figure(figsize=figsize)
        plt.plot(t[:],np.sqrt(Urmssq[:,0,0,0]),label='$u_{\\rm rms}$',linewidth=lnwdth)
        plt.plot(t[:],np.sqrt(Urmssq_x[:,0,0,0]),label='$u_{x,\\rm rms}$',linewidth=lnwdth)
        plt.plot(t[:],np.sqrt(Urmssq_y[:,0,0,0]),label='$u_{y,\\rm rms}$',linewidth=lnwdth)
        plt.plot(t[:],np.sqrt(Urmssq_z[:,0,0,0]),label='$u_{z,\\rm rms}$',linewidth=lnwdth)
        plt.plot(t[:],np.sqrt(Ux_rmssq_gs[:,0,0,0]),label='$u_{x,\\rm rms,gs}$',linewidth=lnwdth)
        plt.plot(t[:],np.sqrt(Uy_rmssq_gs[:,0,0,0]),label='$u_{y,\\rm rms,gs}$',linewidth=lnwdth)
        plt.plot(t[:],np.sqrt(Uz_rmssq_gs[:,0,0,0]),label='$u_{z,\\rm rms,gs}$',linewidth=lnwdth)
        plt.legend(fontsize=fntsz)
        plt.xlabel("Time, $\\omega t$",fontsize=fntsz)
        plt.tick_params(labelsize=fntsz-4)
        
        plt.yscale('log')
        fig.savefig('frames/AAA_log_timeseries.png', dpi=dpi)

        plt.yscale('linear')
        fig.savefig('frames/AAA_linear_timeseries.png', dpi=dpi)

        plt.close()



       	fig = plt.figure(figsize=figsize)
        plt.plot(t[:],P_in[:,0,0,0],label='$P_{\\rm in}$',linewidth=lnwdth)
        plt.plot(t[:],nu_omegasq[:,0,0,0],label='$\\nu \\omega^2$',linewidth=lnwdth)
        plt.legend(fontsize=fntsz)
        plt.yscale('log')
        #plt.ylim(bottom=1e-6)

       	plt.xlabel("Time, $\\omega t$",fontsize=fntsz)
        plt.tick_params(labelsize=fntsz-4)
        fig.savefig('frames/AAA_power.png', dpi=dpi)
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

