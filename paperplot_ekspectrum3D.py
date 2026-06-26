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
    cmap = plt.cm.viridis
    fntsz=24
    lnwdth=3
    dpi = 100
    figsize = (10, 8)
    expkcomp=3
    expgray=-3
    expblack=-1.6
    with h5py.File(filename, mode='r') as file:
        ux = np.asarray(file['tasks']['ux'])#real
        uy = np.asarray(file['tasks']['uy'])
        uz = np.asarray(file['tasks']['uz'])
        N=np.shape(ux)[-1]
        t=np.asarray(file['tasks']['ux'].dims[0]['sim_time'])
        #t_list=[100,180,210,300,400,600]
        t_list=[100,200,300,400,600]
        I_list=[]
        colors=['tab:purple','tab:blue','tab:green','tab:orange','tab:red']
        print('time:')
        print(t)
        for T in t_list:
            I_list=I_list+[max(np.where(t<T)[0])]
        print(I_list)
        k_scale=1  
          
        fft_ux=np.fft.rfftn(ux[I_list,:,:,:],axes=(1,2,3))
        fft_uy=np.fft.rfftn(uy[I_list,:,:,:],axes=(1,2,3))
        fft_uz=np.fft.rfftn(uz[I_list,:,:,:],axes=(1,2,3))
        uiui=(fft_ux*np.conj(fft_ux)+fft_uy*np.conj(fft_uy)+fft_uz*np.conj(fft_uz))/N**6
        
                
        kx = np.fft.fftfreq(N, 1/N)[:, None, None]
        ky = np.fft.fftfreq(N, 1/N)[None, :, None]
        kz = np.fft.rfftfreq(N, 1/N)[None, None, :]
        k = (kx**2 + ky**2+kz**2)**0.5
        kcyl = (kx**2 + ky**2+0*kz**2)**0.5
        kvert = (0*kx**2 + 0*ky**2+kz**2)**0.5
        kmax = int(np.ceil(np.max(k)))
        bins = np.arange(-0.5, N/2+1, 1)
        print(k)
        kcen = bins[:-1] + np.diff(bins)/2
        kcen[0]=0.5
        kcen=kcen*k_scale
        i_len=np.shape(uiui)[0]
        #norm = matplotlib.colors.Normalize(vmin=np.min(t),vmax=np.max(t))
        norm = matplotlib.colors.Normalize(vmin=0,vmax=len(t_list))
        s_m = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
        s_m.set_array([])


        fig,ax=plt.subplots(figsize=figsize)
        # ax[0].axvline(x=mueq/2/2/np.pi,color='green',linestyle='--',linewidth=4)
        for i in range(len(I_list)):
            E_k_u = np.abs(uiui[i,:,:,:])
            # Build histogram over modes, weighted by energy
            spectrum_u, _ = np.histogram(k, bins=bins, weights=E_k_u)
            #ax.plot(kcen,spectrum_u,color=s_m.to_rgba(t[i]),linewidth=lnwdth )
            ax.plot(kcen,spectrum_u*kcen**expkcomp,color=s_m.to_rgba(i),linewidth=lnwdth )
            #ax.plot(kcen,spectrum_u*kcen**expkcomp,linewidth=lnwdth,color=colors[i],label='t=%d'%(t_list[i]) )
        I_pick=10
        ax.plot(kcen,(kcen/kcen[I_pick])**(expgray)*spectrum_u[I_pick]*kcen[I_pick]**expkcomp,color='gray',linestyle="--" ,linewidth=lnwdth-1)
        ax.plot(kcen,(kcen/kcen[I_pick])**(expblack)*spectrum_u[I_pick]*kcen[I_pick]**expkcomp,color='black',linestyle="--" ,linewidth=lnwdth-1)

        # cbar=plt.colorbar(s_m,ax=plt.gca())
        # cbar.set_label('$\\omega t$',fontsize=fntsz+10,rotation=0,labelpad=30)
        # cbar.ax.tick_params(labelsize=fntsz)
        # ax.set_ylim(bottom=1e-12)
        #plt.legend()
        ax.xaxis.set_tick_params(labelsize=fntsz-2)
        ax.yaxis.set_tick_params(labelsize=fntsz-2)
        ax.tick_params(axis='both', which='major',  width=1.5, length=10)
        ax.tick_params(axis='both', which='minor',  width=0.75, length=5)
        
        #ax.set_xticks(fontsize=fntsz-2)
        #ax.set_yticks(fontsize=fntsz-2)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Wavenumber $k$',fontsize=fntsz)
        ax.set_ylabel('$E_u(k)$',fontsize=fntsz)
        plt.legend(fontsize=fntsz)
        plt.xlim(0.5,120)
        plt.savefig('frames/AAA_Energy_spectrum_spherical.png')

        fig,ax=plt.subplots(figsize=figsize)
        # ax[0].axvline(x=mueq/2/2/np.pi,color='green',linestyle='--',linewidth=4)
        for i in range(len(I_list)):
            E_k_u = np.abs(uiui[i,:,:,:])
            # Build histogram over modes, weighted by energy
            spectrum_u, _ = np.histogram(kcyl, bins=bins, weights=E_k_u)
            ax.plot(kcen,spectrum_u*kcen**expkcomp,color=s_m.to_rgba(i),linewidth=lnwdth, label='t=%d'%(t_list[i]) )
            #ax.plot(kcen,spectrum_u*kcen**expkcomp,linewidth=lnwdth,color=colors[i],label='t=%d'%(t_list[i]) )
            
        ax.plot(kcen,(kcen/kcen[I_pick])**(expgray+expkcomp)*spectrum_u[I_pick]*kcen[I_pick]**expkcomp,color='gray',linestyle="--" ,linewidth=lnwdth-1)
        ax.plot(kcen,(kcen/kcen[I_pick])**(expblack+expkcomp)*spectrum_u[I_pick]*kcen[I_pick]**expkcomp,color='black',linestyle="--" ,linewidth=lnwdth-1)
        # cbar=plt.colorbar(s_m,ax=plt.gca())
        # cbar.set_label('$\\omega t$',fontsize=fntsz+10,rotation=0,labelpad=30)
        # cbar.ax.tick_params(labelsize=fntsz)
        # ax.set_ylim(bottom=1e-12)
        #plt.legend()
        ax.xaxis.set_tick_params(labelsize=fntsz-2)
        ax.yaxis.set_tick_params(labelsize=fntsz-2)
        ax.tick_params(axis='both', which='major',  width=1.5, length=10)
        ax.tick_params(axis='both', which='minor',  width=0.75, length=5)
        
        #ax.set_xticks(fontsize=fntsz-2)
        #ax.set_yticks(fontsize=fntsz-2)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Wavenumber $k$',fontsize=fntsz)
        ax.set_ylabel('$E_u(k)$',fontsize=fntsz)
        plt.legend(fontsize=fntsz)
        plt.xlim(0.5,120)
        plt.savefig('frames/AAA_Energy_spectrum_horizontal.png')
        

        fig,ax=plt.subplots(figsize=figsize)
        # ax[0].axvline(x=mueq/2/2/np.pi,color='green',linestyle='--',linewidth=4)
        for i in range(len(I_list)):
            E_k_u = np.abs(uiui[i,:,:,:])
            # Build histogram over modes, weighted by energy
            spectrum_u, _ = np.histogram(kvert, bins=bins, weights=E_k_u)
            ax.plot(kcen,spectrum_u*kcen**expkcomp,color=s_m.to_rgba(i),linewidth=lnwdth ,label='t=%d'%(t_list[i]))
            #ax.plot(kcen,spectrum_u*kcen**expkcomp,linewidth=lnwdth,color=colors[i],label='t=%d'%(t_list[i]) )
        ax.plot(kcen,(kcen/kcen[I_pick])**(expgray+expkcomp)*spectrum_u[I_pick]*kcen[I_pick]**expkcomp,color='gray',linestyle="--" ,linewidth=lnwdth-1)
        ax.plot(kcen,(kcen/kcen[I_pick])**(expblack+expkcomp)*spectrum_u[I_pick]*kcen[I_pick]**expkcomp,color='black',linestyle="--" ,linewidth=lnwdth-1)
        # cbar=plt.colorbar(s_m,ax=plt.gca())
        # cbar.set_label('$\\omega t$',fontsize=fntsz+10,rotation=0,labelpad=30)
        # cbar.ax.tick_params(labelsize=fntsz)
        # ax.set_ylim(bottom=1e-12)
        #plt.legend()
        ax.xaxis.set_tick_params(labelsize=fntsz-2)
        ax.yaxis.set_tick_params(labelsize=fntsz-2)
        ax.tick_params(axis='both', which='major',  width=1.5, length=10)
        ax.tick_params(axis='both', which='minor',  width=0.75, length=5)
        
        #ax.set_xticks(fontsize=fntsz-2)
        #ax.set_yticks(fontsize=fntsz-2)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Wavenumber $k$',fontsize=fntsz)
        ax.set_ylabel('$E_u(k)$',fontsize=fntsz)
        plt.legend(fontsize=fntsz)
        plt.xlim(0.5,120)
        plt.savefig('frames/AAA_Energy_spectrum_vertical.png')
        

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

