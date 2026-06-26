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
    save_alphabeta_plane=1
    scale=8
    alpha_max=4
    A=0.3
    numsnaps=10#average over these last snapshosts
    # Plot settings
    cmap = plt.cm.inferno
    fntsz=24
    dpi = 100
    figsize = (12, 8)
    with h5py.File(filename, mode='r') as file:
        ux = np.asarray(file['tasks']['ux'])#real
        uy = np.asarray(file['tasks']['uy'])
        uz = np.asarray(file['tasks']['uz'])
        N=np.shape(ux)[-1]
        t=np.asarray(file['tasks']['ux'].dims[0]['sim_time'])
        print(t)
        print('N='+str(N))
        kx = np.fft.fftfreq(N, 1/N)[:, None, None]
        ky = np.fft.fftfreq(N, 1/N)[None, :, None]
        print(kx[:,0,0])
        if save_alphabeta_plane:

            k_scale=1  
              
            fft_ux=np.fft.rfftn(ux[:,:,:,:],axes=(1,2,3))
            fft_uy=np.fft.rfftn(uy[:,:,:,:],axes=(1,2,3))
            fft_uz=np.fft.rfftn(uz[:,:,:,:],axes=(1,2,3))
            uiui=(fft_ux*np.conj(fft_ux)+fft_uy*np.conj(fft_uy)+fft_uz*np.conj(fft_uz))/N**6

            alpha_extent=int(N/2)
            #alpha_extent=alpha_max*scale+1
            kxrange=np.arange(alpha_extent)
            kyrange=np.arange(alpha_extent)
            for i in range(len(uiui[:,0,0,0])):
                fig, ax = plt.subplots(1, 1,figsize=(12,9))
                uiui_real=np.abs(uiui[i,:alpha_extent,:alpha_extent,:])
                uiui_real=np.sum(uiui_real,axis=-1)
                uiui_real[scale,0]=0
                uiui_real=np.log10(uiui_real+1e-16)
                pmesh=ax.pcolormesh(kxrange/scale,kyrange/scale,uiui_real.T,cmap='cividis',vmin=np.min(uiui_real[1:,1:]))
                cbar=fig.colorbar(pmesh,ax=ax)
                cbar.ax.tick_params(labelsize=fntsz)
                ax.set_xlabel("$\\alpha$",fontsize=fntsz+4)
                ax.set_ylabel("$\\beta$",fontsize=fntsz+4,labelpad=10,rotation=0)   
                cbar.set_label("$\\log E(\\alpha,\\beta)$",fontsize=fntsz+8,labelpad=30,rotation=90)
                ax.xaxis.set_tick_params(labelsize=fntsz)
                ax.yaxis.set_tick_params(labelsize=fntsz)
                plt.tight_layout()
                ax.set_aspect('equal','box')
                plt.savefig("frames/AlphaBeta_spectrum_"+str(i).zfill(4)+".png")    
                plt.close()


            #Sum over kz modes
            uiui=np.sum(uiui,axis=-1)
            uiui=np.average(uiui[-numsnaps:,:,:],axis=0)#average over time
            '''
            i_len=np.shape(uiui)[0]
            for i in range(N):
                for j in range(N):
                    np.savetxt('alphabeta/'+str(int(kx[i,0,0]) )+"_"+str( int(ky[0,j,0]) )+'.txt', uiui[:,i,j].real )
            '''
        
        alpha_extent=int(N/2)
        #alpha_extent=alpha_max*scale+1
        kxrange=np.arange(alpha_extent)
        kyrange=np.arange(alpha_extent)
        '''
        growthrate=np.zeros( (alpha_extent,alpha_extent) )
        for i in range(N):
            for j in range(N):
                if kx[i,0,0]>=0 and kx[i,0,0]<alpha_extent and ky[0,j,0]>=0 and ky[0,j,0]<alpha_extent:
                    filename='alphabeta/'+str(int(kx[i,0,0]) )+"_"+str( int(ky[0,j,0]) )+'.txt'
                    usq_kxky=np.loadtxt(filename)
                    I=int(kx[i,0,0])
                    J=int(ky[0,j,0])
                    # if usq_kxky[-1]>1e-10:
                    #     print('$k_x,k_y=%d,%d$'%(kx[i,0,0],ky[0,j,0]))
                    #     growthrate[I,J]=np.log(usq_kxky[i_stop]/usq_kxky[i_start])/2/duration
                    growthrate[I,J]=np.log(usq_kxky[i_stop]/usq_kxky[i_start])/2/duration
                # if (kx[i,0,0]==-int(N/2) and ky[0,j,0]>=0) or (ky[0,j,0]==-int(N/2) and kx[i,0,0]>=0) :
                    # filename='alphabeta/'+str(int(kx[i,0,0]) )+"_"+str( int(ky[0,j,0]) )+'.txt'
                    # usq_kxky=np.loadtxt(filename)
                    # i_start=int(len(t)/2)
                    # i_stop=-1
                    # duration=t[i_stop]-t[i_start]
                    # I=abs(int(kx[i,0,0]))
                    # J=abs(int(ky[0,j,0]))
                    # growthrate[I,J]=np.log(usq_kxky[i_stop]/usq_kxky[i_start])/2/duration
                    # 
        '''
        fig, ax = plt.subplots(1, 1,figsize=(12,9))
        print(kxrange)
        print(np.shape(kxrange))
        print(np.shape(uiui))
        print(np.shape(uiui[:alpha_extent,:alpha_extent]))
        uiui[0,0]=0
        uiui_real=np.log10(np.abs(uiui[:alpha_extent,:alpha_extent])+1e-16)
        pmesh=ax.pcolormesh(kxrange/scale,kyrange/scale,uiui_real.T,cmap='cividis',vmin=np.min(uiui_real[1:,1:]))
        cbar=fig.colorbar(pmesh,ax=ax)
        cbar.ax.tick_params(labelsize=fntsz)
        #plt.title('Dedalus')
        # ax.set_xlabel("$k_x\\; (\\alpha)$",fontsize=fntsz+4,labelpad=0)
        # ax.set_ylabel("$k_y\\; (\\beta)$",fontsize=fntsz+4,labelpad=30,rotation=0)   
        ax.set_xlabel("$\\alpha$",fontsize=fntsz+4)
        ax.set_ylabel("$\\beta$",fontsize=fntsz+4,labelpad=10,rotation=0)   
        cbar.set_label("$\\log E(\\alpha,\\beta)$",fontsize=fntsz+8,labelpad=30,rotation=90)
        #ax.tick_params(labelsize=fntsz)
        ax.xaxis.set_tick_params(labelsize=fntsz)
        ax.yaxis.set_tick_params(labelsize=fntsz)
        plt.tight_layout()
        ax.set_aspect('equal','box')
        plt.savefig("frames/AlphaBeta_spectrum.png")    



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

