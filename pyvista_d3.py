import pyvista as pv
import numpy as np
CMAP='RdBu_r'

file_name = 'snapshots_VTK_omegaz_007' # This seems correct based on your output
file_path = file_name+'.vtr' # This seems correct based on your output
dset_omegaz = pv.read(file_path)
# dset_omegaz.plot()
p = pv.Plotter()
p.camera_position=[(-113.28812764230723, -120.94139145487351, 165.8158067603819),
 (21.515146519065716, 25.36215299155889, 31.461084656053252),
 (0.3779168465854753, 0.41281766278855725, 0.828710102723916)]


p.add_mesh(dset_omegaz,
# clim=[-0.25,0.25], 
clim=[-0.5,0.5], 
show_edges=False, cmap=CMAP, show_scalar_bar=False) # Using viridis as an example colormap
p.show_axes()
bounds = dset_omegaz.bounds
print(bounds)
center = dset_omegaz.center
p.add_scalar_bar(title='',fmt='%.1f')

# Place the light source to one side, slightly above, looking at the center
# Adjust these multipliers based on the scale of your data and desired effect
light_position = (-bounds[1] * 4, -bounds[3] * 4, -bounds[5] * 1) # Example: twice max X, Y, Z extent
scene_light = pv.Light(position=light_position,
                       focal_point=center,
                       intensity=0.3, # Even higher intensity for a strong effect
                       light_type='scene light')
p.add_light(scene_light)


saved_camera_positions = [] # List to store multiple saved positions

def save_camera_callback():
    """Callback function to save the current camera position."""
    current_cpos = p.camera_position
    saved_camera_positions.append(current_cpos)
    print(f"Camera position saved! Total saved: {len(saved_camera_positions)}")
    print(f"Saved Position: {current_cpos}")

def print_all_saved_cameras_callback():
    """Callback to print all saved camera positions."""
    if saved_camera_positions:
        print("\n--- All Saved Camera Positions ---")
        for i, cpos in enumerate(saved_camera_positions):
            print(f"Position {i+1}: {cpos}")
        print("----------------------------------")
    else:
        print("No camera positions saved yet.")

# Add key events to the plotter
# When you press 's' in the plot window, it will call save_camera_callback
p.add_key_event('s', save_camera_callback)
# When you press 'p' in the plot window, it will print all saved positions
p.add_key_event('p', print_all_saved_cameras_callback)

p.show()
p.screenshot('screenshots/'+file_name+'.png')


