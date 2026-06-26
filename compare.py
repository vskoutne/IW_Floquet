import h5py
import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as colors
        
dpi = 500
figsize = (9, 8)
fntsz=24
lnwdth=3

filenames1=['A1p0_theta10_rk443_visc3_N256', 'A1p0_theta30_rk443_visc3_N256', 'A1p0_theta45_rk443_visc3_N256', 'A1p0_theta56_rk443_visc3_N256', 'A1p0_theta60_rk443_visc3_N256', 'A1p0_theta72_rk443_visc3_N256','A1p0_theta80_rk443_visc3_N256']#, 'A1p0_theta82_rk443_visc3_N256']
angles1=[9.46, 26.5, 45, 56, 63.4, 72, 80.5]#, 82]

GS_fraction1,Etot,Egs,time=[],[],[],[]

for filename in filenames1:
    with h5py.File(filename+'/series/series_s1.h5', mode='r') as file:
        print(file)
        print(file.keys())
        print(file['scales'].keys())
        print(file['tasks'].keys())

        Urmssq = file['tasks']['Urmssq']
        Ux_rmssq_gs = file['tasks']['Ux_rmssq_gs']
        Uy_rmssq_gs = file['tasks']['Uy_rmssq_gs']
        Uz_rmssq_gs = file['tasks']['Uz_rmssq_gs']
        I_m=-1
        print(filename)
        print(Urmssq[0,0,0,0])
        print((Ux_rmssq_gs[:,0,0,0]+Uy_rmssq_gs[:,0,0,0]+Uz_rmssq_gs[:,0,0,0]))
        print((Ux_rmssq_gs[I_m,0,0,0]+Uy_rmssq_gs[I_m,0,0,0]+Uz_rmssq_gs[I_m,0,0,0])/(Urmssq[0,0,0,0]))
        GS_fraction1=GS_fraction1+[(Ux_rmssq_gs[I_m,0,0,0]+Uy_rmssq_gs[I_m,0,0,0]+Uz_rmssq_gs[I_m,0,0,0])/(Urmssq[0,0,0,0])]
        Etot=Etot+[(Urmssq[:,0,0,0])]
        Egs=Egs+[(Ux_rmssq_gs[:,0,0,0]+Uy_rmssq_gs[:,0,0,0]+Uz_rmssq_gs[:,0,0,0])]
        t = Urmssq.dims[0]['sim_time']
        t=np.asarray(t[:])
        time=time+[t]

filenames2=['A0p6_theta45_rk443_visc3_N256']
angles2=[45]

GS_fraction2=[]

for filename in filenames2:
    with h5py.File(filename+'/series/series_s1.h5', mode='r') as file:
        print(file)
        print(file.keys())
        print(file['scales'].keys())
        print(file['tasks'].keys())

        Urmssq = file['tasks']['Urmssq']
        Ux_rmssq_gs = file['tasks']['Ux_rmssq_gs']
        Uy_rmssq_gs = file['tasks']['Uy_rmssq_gs']
        Uz_rmssq_gs = file['tasks']['Uz_rmssq_gs']
        I_m=-1
        GS_fraction2=GS_fraction2+[(Ux_rmssq_gs[I_m,0,0,0]+Uy_rmssq_gs[I_m,0,0,0]+Uz_rmssq_gs[I_m,0,0,0])/(Urmssq[0,0,0,0])]
        Etot=Etot+[(Urmssq[:,0,0,0])]
        Egs=Egs+[(Ux_rmssq_gs[:,0,0,0]+Uy_rmssq_gs[:,0,0,0]+Uz_rmssq_gs[:,0,0,0])]
        t = Urmssq.dims[0]['sim_time']
        t=np.asarray(t[:])
        time=time+[t]

#filenames3=['A0p3_theta10_rk443_visc3_N256','A0p3_theta45_rk443_visc3_N256','A0p3_theta80_rk443_visc3_N256']
#angles3=[9.46, 45, 80.5]

filenames3=['A0p3_theta10_rk443_visc3_N256', 'A0p3_theta30_rk443_visc3_N256', 'A0p3_theta45_rk443_visc3_N256', 'A0p3_theta56_rk443_visc3_N256', 'A0p3_theta60_rk443_visc3_N256', 'A0p3_theta72_rk443_visc3_N256','A0p3_theta80_rk443_visc3_N256']#, 'A0p3_theta82_rk443_visc3_N256']
angles3=[9.46, 26.5, 45, 56, 63.4, 72, 80.5]#, 82]

GS_fraction3=[]

for filename in filenames3:
    with h5py.File(filename+'/series/series_s1.h5', mode='r') as file:
        print(file)
        print(file.keys())
        print(file['scales'].keys())
        print(file['tasks'].keys())

        Urmssq = file['tasks']['Urmssq']
        Ux_rmssq_gs = file['tasks']['Ux_rmssq_gs']
        Uy_rmssq_gs = file['tasks']['Uy_rmssq_gs']
        Uz_rmssq_gs = file['tasks']['Uz_rmssq_gs']
        I_m=-1
        GS_fraction3=GS_fraction3+[(Ux_rmssq_gs[I_m,0,0,0]+Uy_rmssq_gs[I_m,0,0,0]+Uz_rmssq_gs[I_m,0,0,0])/(Urmssq[0,0,0,0])]

print(angles1)
print(GS_fraction1)
fig = plt.figure(figsize=figsize)
for i in range(len(angles1)):
    plt.plot(time[i],Etot[i],label='$\\theta=%d$'%(angles1[i]))
plt.legend(fontsize=fntsz)
#plt.yscale('log')
#plt.xlim(0,90)
plt.xlabel("Time, $\\omega t$",fontsize=fntsz)
plt.tick_params(labelsize=fntsz-4)
fig.savefig('compare/Energy_tot.png', dpi=dpi)
plt.close()

fig = plt.figure(figsize=figsize)
for i in range(len(angles1)):
    plt.plot(time[i],Egs[i],label='$\\theta=%d$'%(angles1[i]))
plt.legend(fontsize=fntsz)
plt.yscale('log')
plt.ylim(bottom=1e-6)
plt.xlabel("$\\theta$",fontsize=fntsz)
plt.tick_params(labelsize=fntsz-4)
fig.savefig('compare/Energy_gs.png', dpi=dpi)
plt.close()

print(angles1)
angles1=np.asarray(angles1)*np.pi/180
angles2=np.asarray(angles2)*np.pi/180
angles3=np.asarray(angles3)*np.pi/180
print(angles1)
print(np.cos(angles1))
print(GS_fraction1)
fig = plt.figure(figsize=figsize)
ax=plt.gca()
plt.scatter(np.cos(angles1),GS_fraction1,s=50,label='$A\'=1.0$',color='blue')
#plt.scatter(np.cos(angles2),GS_fraction2,label='$A\'=0.6$')
plt.scatter(np.cos(angles3),GS_fraction3,s=50,label='$A\'=0.3$',color='orange')
plt.legend(fontsize=fntsz)
#plt.yscale('log')
plt.ylim(bottom=0)
plt.xlabel("$\\omega/2\\Omega$",fontsize=fntsz)
ax.tick_params(axis='both', which='major',  width=1.5, length=10)
ax.tick_params(axis='both', which='minor',  width=0.75, length=5)
plt.tick_params(labelsize=fntsz-4)
fig.savefig('compare/GS_fraction.png', dpi=dpi)
plt.close()
