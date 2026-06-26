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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors


def main(filename, start, count, output):
    """Save plot of specified tasks for given range of analysis writes."""
    print(filename)
    print(start)
    print(count)
    print(output)
    s=int(filename[-4:-3])
    def save_index(index):
        return index+(s-1)*count
    dpi = 100
    figsize = (8, 8)
    savename_func = lambda write: 'write_{:06}.png'.format(write)
    skip=1
    skip_frames=5
    twod_frames=0
    twod_frames_MHD=0
    MHD_ON=1
    colormap=plt.cm.RdBu_r
    with h5py.File(filename, mode='r') as file:
        mu_XY = file['tasks']['mu_XY']
        Az_XY = file['tasks']['Az_XY']
        Bx_XY = file['tasks']['Bx_XY']
        Ux_XY = file['tasks']['Ux_XY']
        print(np.shape(mu_XY))
        X = mu_XY.dims[2][0][:].ravel()[::-1]
        Y = mu_XY.dims[1][0][:].ravel()[::-1]
        #t = mu_XY.dims[0]['sim_time']
        print(np.shape(X))
        print(np.shape(Y))
        #print(np.shape(t))
        for index in range(count):
            fig, ax = plt.subplots()
            cntrf=ax.pcolormesh(X,Y, Az_XY[index,:,:,0],cmap=colormap) 
            cbar=fig.colorbar(cntrf)
            plt.xlabel('X')
            plt.ylabel('Y')
            savepath = "frames/Az_XY_"+str(index)
            fig.savefig(str(savepath), dpi=dpi)
            plt.close(fig)
        '''
        for index in range(count):
            fig, ax = plt.subplots()
            cntrf=ax.pcolormesh(X,Y, mu_XY[index,:,:,0],cmap=colormap) 
            cbar=fig.colorbar(cntrf)
            plt.xlabel('X')
            plt.ylabel('Y')
            savepath = "frames/mu_XY_"+str(index)
            fig.savefig(str(savepath), dpi=dpi)
            plt.close(fig)
        
        for index in range(count):
            fig, ax = plt.subplots()
            cntrf=ax.pcolormesh(X,Y, Bx_XY[index,:,:,0],cmap=colormap) 
            cbar=fig.colorbar(cntrf)
            plt.xlabel('X')
            plt.ylabel('Y')
            savepath = "frames/Bx_XY_"+str(index)
            fig.savefig(str(savepath), dpi=dpi)
            plt.close(fig)        
        for index in range(count):
            fig, ax = plt.subplots()
            cntrf=ax.pcolormesh(X,Y, Ux_XY[index,:,:,0],cmap=colormap) 
            cbar=fig.colorbar(cntrf)
            plt.xlabel('X')
            plt.ylabel('Y')
            savepath = "frames/Ux_XY_"+str(index)
            fig.savefig(str(savepath), dpi=dpi)
            plt.close(fig)
        '''
    '''
        #RSLICES
        HD_slices=[ur_thetaphi_r9,utheta_thetaphi_r9,uphi_thetaphi_r9]
        HD_names=['ur_thetaphi_r9','utheta_thetaphi_r9','uphi_thetaphi_r9']
        HD_names=['ur_thetaphi_r9']
        
        for i in range(len(HD_names)):
            for index in range(start, start+count)[::skip_frames]:
                fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
                maxval=np.max(np.abs(HD_slices[i][index,0,:,:]))
                cntrf=ax.pcolormesh(phi,theta, HD_slices[i][index,:,:,0].T,cmap=colormap,vmin=-maxval,vmax=maxval) 
                cbar=fig.colorbar(cntrf)
                savepath = "frames/RSlice_"+str(HD_names[i])+"_"+str(save_index(index))
                fig.savefig(str(savepath), dpi=dpi)
                plt.close(fig)
        #PHISLICES
        HD_slices=[mu_phi0,uphi_phi0,ur_phi0,utheta_phi0]
        HD_names=['mu_phi0','uphi_phi0','ur_phi0','utheta_phi0']
        HD_names=['mu_phi0','uphi_phi0']
        
        for i in range(len(HD_names)):
            for index in range(start, start+count)[::skip_frames]:
                fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
                maxval=np.max(np.abs(HD_slices[i][index,0,:,:]))
                cntrf=ax.pcolormesh(theta, r, HD_slices[i][index,0,:,:].T,cmap=colormap,vmin=-maxval,vmax=maxval) 
                cbar=fig.colorbar(cntrf)
                savepath = "frames/PhiSlice_"+str(HD_names[i])+"_"+str(save_index(index))
                fig.savefig(str(savepath), dpi=dpi)
                plt.close(fig)
        #Cartesian
        
        for i in range(len(HD_names)):
            for index in range(start, start+count)[::skip_frames]:
                fig, ax = plt.subplots()
                maxval=np.max(np.abs(HD_slices[i][index,0,:,:]))
                cntrf=ax.pcolormesh( r,theta_deg, HD_slices[i][index,0,:,:],cmap=colormap,vmin=-maxval,vmax=maxval) 
                cbar=fig.colorbar(cntrf)
                plt.xlabel('r')
                plt.ylabel('theta')
                plt.ylim(0,180)
                savepath = "frames/CartPhiSlice_"+str(HD_names[i])+"_"+str(save_index(index))
                fig.savefig(str(savepath), dpi=dpi)
                plt.close(fig)
        

        if MHD_ON:        
            #Non azimuthal fluctuations
            for index in range(start, start+count)[::skip_frames]:
                fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
                maxval=np.max(np.abs( Bphi_phi0[index,0,:,:]-Bphi_azi[index,0,:,:]))
                cntrf=ax.pcolormesh(theta,r, mu_phi0[index,0,:,:].T-mu_azi[index,0,:,:].T,cmap=colormap,vmin=-maxval,vmax=maxval) 
                cbar=fig.colorbar(cntrf)
                savepath = "frames/PhiSlice_mu_nonazi_phi0_"+str(save_index(index))
                fig.savefig(str(savepath), dpi=dpi)
                plt.close(fig)

            for index in range(start, start+count)[::skip_frames]:
                fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
                maxval=np.max(np.abs( Bphi_phi0[index,0,:,:]-Bphi_azi[index,0,:,:]))
                cntrf=ax.pcolormesh(theta,r, Bphi_phi0[index,0,:,:].T-Bphi_azi[index,0,:,:].T,cmap=colormap,vmin=-maxval,vmax=maxval) 
                cbar=fig.colorbar(cntrf)
                savepath = "frames/PhiSlice_Bphi_nonazi_phi0_"+str(save_index(index))
                fig.savefig(str(savepath), dpi=dpi)
                plt.close(fig)

            for index in range(start, start+count)[::skip_frames]:
                fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
                maxval=np.max(np.abs(Bphi_thetaphi_r9[index,:,:,0]-Bphi_thetaphi_r9_azi[index,:,:,0]))
                cntrf=ax.pcolormesh(phi,theta, Bphi_thetaphi_r9[index,:,:,0].T-Bphi_thetaphi_r9_azi[index,:,:,0].T,cmap=colormap,vmin=-maxval,vmax=maxval) 
                cbar=fig.colorbar(cntrf)
                savepath = "frames/RSlice_Bphi_thetaphi_r9_nonazi_"+str(save_index(index))
                fig.savefig(str(savepath), dpi=dpi)
                plt.close(fig)
        
            #MHD_slices=[Bphi_phi0,Br_phi0,Btheta_phi0,Ar_phi0,Atheta_phi0,FBradial_phi0,FBtheta_phi0,LFradial_phi0,LFtheta_phi0,PGradial_phi0,PGtheta_phi0]
            #MHD_names=['Bphi_phi0','Br_phi0','Btheta_phi0','Ar_phi0','Atheta_phi0','FBradial_phi0','FBtheta_phi0','LFradial_phi0','LFtheta_phi0','PGradial_phi0_full','PGtheta_phi0']
            MHD_slices=[Bphi_phi0,Br_phi0,Btheta_phi0,Bphi_azi,Br_azi]
            MHD_names=['Bphi_phi0','Br_phi0','Btheta_phi0','Bphi_azi','Br_azi']
            #Polar
            for i in range(len(MHD_names)):
                for index in range(start, start+count)[::skip_frames]:
                    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
                    maxval=np.max(np.abs(MHD_slices[i][index,0,:,:]))
                    cntrf=ax.pcolormesh(theta, r, MHD_slices[i][index,0,:,:].T,cmap=colormap,vmin=-maxval,vmax=maxval) 
                    cbar=fig.colorbar(cntrf)
                    savepath = "frames/PhiSlice_"+str(MHD_names[i])+"_"+str(save_index(index))
                    fig.savefig(str(savepath), dpi=dpi)
                    plt.close(fig)
            #Cartesian
            
            for i in range(len(MHD_names)):
                for index in range(start, start+count)[::skip_frames]:
                    fig, ax = plt.subplots()
                    maxval=np.max(np.abs(MHD_slices[i][index,0,:,:]))
                    cntrf=ax.pcolormesh( r,theta_deg, MHD_slices[i][index,0,:,:],cmap=colormap,vmin=-maxval,vmax=maxval) 
                    cbar=fig.colorbar(cntrf)
                    plt.xlabel('r')
                    plt.ylabel('theta')
                    plt.ylim(0,180)
                    savepath = "frames/CartPhiSlice_"+str(MHD_names[i])+"_"+str(save_index(index))
                    fig.savefig(str(savepath), dpi=dpi)
                    plt.close(fig)
            

            MHD_slices=[Bphi_thetaphi_r9,Btheta_thetaphi_r9,Br_thetaphi_r9]
            MHD_names=['Bphi_thetaphi','Btheta_thetaphi_r9','Br_thetaphi_r9']
            MHD_names=['Bphi_thetaphi_r9']
            
            for i in range(len(MHD_names)):
                for index in range(start, start+count)[::skip_frames]:
                    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
                    maxval=np.max(np.abs(MHD_slices[i][index,0,:,:]))
                    cntrf=ax.pcolormesh(phi,theta, MHD_slices[i][index,:,:,0].T,cmap=colormap,vmin=-maxval,vmax=maxval) 
                    cbar=fig.colorbar(cntrf)
                    savepath = "frames/RSlice_"+str(MHD_names[i])+"_"+str(save_index(index))
                    fig.savefig(str(savepath), dpi=dpi)
                    plt.close(fig)
                    print(save_index(index))

        if twod_frames:
            
            for index in range(start, start+count)[::skip_frames]:
                fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
                cntrf=ax.pcolormesh(theta, r, T_azi[index,0,:,:].T,cmap=colormap) 
                cbar=fig.colorbar(cntrf)
                savepath = "frames/T_m0_"+str(index)
                fig.savefig(str(savepath), dpi=dpi)
                plt.close(fig)
            
            for index in range(start, start+count)[::skip_frames]:
                fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
                cntrf=ax.pcolormesh(theta, r, ur_azi[index,0,:,:].T,cmap=colormap) 
                cbar=fig.colorbar(cntrf)
                savepath = "frames/ur_m0_"+str(index)
                fig.savefig(str(savepath), dpi=dpi)
                plt.close(fig)

            for index in range(start, start+count)[::skip_frames]:
                fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
                cntrf=ax.pcolormesh(theta, r, utheta_azi[index,0,:,:].T,cmap=colormap) 
                cbar=fig.colorbar(cntrf)
                savepath = "frames/utheta_m0_"+str(index)
                fig.savefig(str(savepath), dpi=dpi)
                plt.close(fig)
            
            for index in range(start, start+count)[::skip_frames]:
                fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
                cntrf=ax.pcolormesh(theta, r, uphi_azi[index,0,:,:].T,cmap=colormap) 
                cbar=fig.colorbar(cntrf)
                savepath = "frames/uphi_m0_"+str(index)
                fig.savefig(str(savepath), dpi=dpi)
                plt.close(fig)
            
            for index in range(start, start+count):
                fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
                cntrf=ax.pcolormesh(phi, r, omegaz_equatorial_plane[index,:,0,:].T,cmap=plt.cm.RdBu) 
                cbar=fig.colorbar(cntrf)
                savepath = "frames/omegaz_equatorial_plane_"+str(index)
                fig.savefig(str(savepath), dpi=dpi)
                plt.close(fig)
            
            for index in range(start, start+count)[::skip_frames]:
                fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
                cntrf=ax.pcolormesh(phi, r, ur_equatorial_plane[index,:,0,:].T,cmap=colormap) 
                cbar=fig.colorbar(cntrf)
                savepath = "frames/ur_equatorial_plane_"+str(index)
                fig.savefig(str(savepath), dpi=dpi)
                plt.close(fig)
            for index in range(start, start+count)[::skip_frames]:
                fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
                cntrf=ax.pcolormesh(phi, r, uphi_equatorial_plane[index,:,0,:].T,cmap=colormap) 
                cbar=fig.colorbar(cntrf)
                savepath = "frames/uphi_equatorial_plane_"+str(index)
                fig.savefig(str(savepath), dpi=dpi)
                plt.close(fig)

    '''

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

