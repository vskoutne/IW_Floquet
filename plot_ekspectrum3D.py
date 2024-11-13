"""
Plot cutaway ball outputs.

Usage:
    plot_cartesian3d.py <files>... [--output=<dir>]

Options:
    --output=<dir>  Output directory [default: ./frames]
cmap='viridis' #see sequential colormaps for pylab on google for options


"""

import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
        
def main(filename, start, count, output):
    """Save plot of specified tasks for given range of analysis writes."""
    # Plot settings
    # r: outer radial face
    # s: starting meridional face
    # e:ending meridional face
    cmap = plt.cm.inferno
    fntsz=16
    dpi = 100
    figsize = (12, 8)
    # mueq=(4*np.pi)*32
    # lambda_mu=mueq**2
    # chi=1
    # eta=chi*0.5/mueq
    # gamma_mueq=mueq**2*eta/4
    gamma_mueq=1.0
    with h5py.File(filename, mode='r') as file:
        ux = np.asarray(file['tasks']['ux'])#real
        uy = np.asarray(file['tasks']['uy'])
        uz = np.asarray(file['tasks']['uz'])
        N=np.shape(ux)[-1]
        t=np.asarray(file['tasks']['ux'].dims[0]['sim_time'])*gamma_mueq

        k_scale=1  
          
        fft_ux=np.fft.rfftn(ux[:,:,:,:],axes=(1,2,3))
        fft_uy=np.fft.rfftn(uy[:,:,:,:],axes=(1,2,3))
        fft_uz=np.fft.rfftn(uz[:,:,:,:],axes=(1,2,3))
        uiui=(fft_ux*np.conj(fft_ux)+fft_uy*np.conj(fft_uy)+fft_uz*np.conj(fft_uz))/N**6
        
                
        kx = np.fft.fftfreq(N, 1/N)[:, None, None]
        ky = np.fft.fftfreq(N, 1/N)[None, :, None]
        kz = np.fft.rfftfreq(N, 1/N)[None, None, :]
        k = (kx**2 + ky**2+kz**2)**0.5
        kmax = int(np.ceil(np.max(k)))
        bins = np.arange(-0.5, N/2+1, 1)
        kcen = bins[:-1] + np.diff(bins)/2
        kcen[0]=0.5
        kcen=kcen*k_scale
        i_len=np.shape(uiui)[0]
        norm = matplotlib.colors.Normalize(vmin=np.min(t),vmax=np.max(t))
        s_m = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
        s_m.set_array([])


        fig,ax=plt.subplots(figsize=figsize)
        # ax[0].axvline(x=mueq/2/2/np.pi,color='green',linestyle='--',linewidth=4)
        for i in range(i_len):
            E_k_u = np.abs(uiui[i,:,:,:])
            # Build histogram over modes, weighted by energy
            spectrum_u, _ = np.histogram(k, bins=bins, weights=E_k_u)
            ax.plot(kcen,spectrum_u,color=s_m.to_rgba(t[i]) )

        cbar=plt.colorbar(s_m,ax=plt.gca())
        cbar.set_label('$\\omega t$',fontsize=fntsz+10,rotation=0,labelpad=20)
        cbar.ax.tick_params(labelsize=fntsz)
        # ax.set_ylim(bottom=1e-12)
        #plt.legend()
        ax.xaxis.set_tick_params(labelsize=fntsz-2)
        ax.yaxis.set_tick_params(labelsize=fntsz-2)
        #ax.set_xticks(fontsize=fntsz-2)
        #ax.set_yticks(fontsize=fntsz-2)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Wavenumber $k$',fontsize=fntsz+10)
        ax.set_ylabel('$E_u(k)$',fontsize=fntsz+10)
        plt.savefig('frames/Energy_spectrum.png')
        

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

