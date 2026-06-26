"""
Convert spherical Dedalus h5 snapshot to vtk format

Usage:
    convert_to_vtk.py <files>...
"""
import h5py
import numpy as np
import pyvista as pv
from pyevtk.hl import gridToVTK



def main(filename, index=-1):
    with h5py.File(filename, mode='r') as file:

        # Load data
        #ux = file['tasks']['ux']
        #uy = file['tasks']['uy']
        #uz = file['tasks']['uz']
        print('loading')
        name='omegaz'
        dset = file['tasks']['omegaz']
        print(dset)
        x = dset.dims[1][0][:].ravel()
        y = dset.dims[2][0][:].ravel()
        z = dset.dims[3][0][:].ravel()
        # ux=np.asarray(ux[:,:,:,:])
        # uy=np.asarray(uy[:,:,:,:])
        # uz=np.asarray(uz[:,:,:,:])
        # omegaz=np.asarray(omegaz[:,:,:,:])
        # print(np.shape(ux))
        # I_save=np.shape(ux)[0]
        # print('I_save')
        # print(I_save)
        # print(np.shape(dset[:,:,:,:]))
        # num_times=np.shape(dset[:,:,:,:])
        num_times=21
        # I_save=0
        # dset_slice = np.asarray(dset[I_save,:,:,:])
        # pointData = {'omegaz': dset_slice[:,:,:]}
        # gridToVTK("snapshots_VTK_"+name+"_"+str(I_save).zfill(3), x, y, z, pointData=pointData)

        for I_save in range(num_times):
            print("I_save=%d"%(I_save))
            dset_slice = np.asarray(dset[I_save,:,:,:])
            pointData = {'omegaz': dset_slice[:,:,:]}
            gridToVTK("snapshots_VTK_"+name+"_"+str(I_save).zfill(3), x, y, z, pointData=pointData)

        #grid_pv = pv.RectilinearGrid(x, y, z)

        # 2. Combine the U, V, W components into a single (N_points, 3) NumPy array
        # Ensure the dimensions match PyVista's internal point ordering (typically C-order/row-major).
        # np.stack creates a (DimX, DimY, DimZ, 3) array
        #velocity_combined_3D = np.stack((ux_slice, uy_slice, uz_slice), axis=-1) 

        # .reshape(-1, 3, order='C') flattens to (N_points, 3) in C-order
        #velocity_flat_for_pyvista = velocity_combined_3D.reshape(-1, 3, order='C')

        # 3. Assign the combined vector data to the PyVista grid's point_data
        # The key 'u' will be the name of your vector array in the VTK file
        #grid_pv.point_data['u'] = velocity_flat_for_pyvista


        # 4. Save the PyVista grid to a VTK file
        #output_vtk_filename = "snapshots_VTK.vtr"
        #grid_pv.save(output_vtk_filename)
        #dset_loaded = pv.read(output_vtk_filename)
        #print("\n--- PyVista Read Result (from PyVista-exported file) ---")
        #print(dset_loaded) # This should now show 'Vectors: u' and correctly identify it.
        #print(f"Vector field saved to '{output_vtk_filename}' using PyVista.")

        
        # print(np.shape(omegaz[I_save,:,:,:]))
        # Save to VTK
        # pointData = {'u': (ux[I_save,:,:,:], uy[I_save,:,:,:], uz[I_save,:,:,:])}
        # pointData = {'ux': ux_slice[:,:,:]}
        # gridToVTK("snapshots_ux_VTK", x, y, z, pointData=pointData)
        # pointData = {'uy': uy_slice[:,:,:]}
        # gridToVTK("snapshots_uy_VTK", x, y, z, pointData=pointData)
        # pointData = {'uz': uz_slice[:,:,:]}
        # gridToVTK("snapshots_uz_VTK", x, y, z, pointData=pointData)
        # pointData = {'omegaz': omegaz_slice[:,:,:]}
        # gridToVTK("snapshots_omegaz_VTK", x, y, z, pointData=pointData)


if __name__=='__main__':
    from docopt import docopt
    from dedalus.tools import post

    args = docopt(__doc__)
    main(args['<files>'][0])
