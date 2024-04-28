import numpy as np
from sympy import *
from sympy.vector import dot,cross,CoordSys3D
import matplotlib.pyplot as plt
import scipy.optimize as scpop
import scipy as sp
from scipy import linalg
import warnings as warnings

#omega=np.sin(45*np.pi/180);
#print(omega)
#k=1; kz=omega*k; kx=np.sqrt(k**2-kz**2);
#st = 1 # Wave steepness
#print('omega=',omega)

def PSI_run(omega,k,kz,kx,st,beta,gamma):
    kAx = Symbol('kAx'); kAy = Symbol('kAy'); kAz = Symbol('kAz');
    kBx = Symbol('kBx'); kBy = Symbol('kBy'); kBz = Symbol('kBz');
    e = CoordSys3D('e')
    Omega=-(kx/k)*e.i+0*e.j+(kz/k)*e.k;
    kA=kAx*e.i+kAy*e.j+kAz*e.k; kAm=sqrt(dot(kA,kA))
    kB=kBx*e.i+kBy*e.j+kBz*e.k; kBm=sqrt(dot(kB,kB))
    omegaA=-kAz*kz/k-dot(Omega,kA)/kAm
    omegaB=-kBz*kz/k+dot(Omega,kB)/kBm # plus or minus?
    omegaAt=omegaA+kAz*kz/k; omegaBt=omegaB+kBz*kz/k; 
    Atau = Symbol('Atau'); B = Symbol('B')
    Btau = Symbol('Btau'); A = Symbol('A') 
    kw=1*e.k
    uw=(1/2)*(1*e.i-I*e.j)*(-st*k/kx)
    uwc=(1/2)*(1*e.i+I*e.j)*(-st*k/kx)

    nv=np.linspace(1,10,10)
    kAxv=np.zeros(np.size(nv))
    for ind in range(np.size(nv)):
        n=nv[ind]
        #print(n)
        freqres=simplify((omegaA-omegaB).subs([(kBx,kAx),(kBy,kAy),(kBz,(kAz+1))]))
        eqn=lambdify(kAx,freqres.subs([(kAy,beta),(kAz,gamma+n)]))
        sol=scpop.root_scalar(eqn,bracket=[0,30])
        kAxv[ind]=sol.root
    
        kAyeval=beta  # beta=0

    # Eigenfunctions
    uAx=1
    uAy=(-I*omegaAt*kAy*kAz-dot(Omega,kA)*kAx)*uAx/(-I*omegaAt*kAx*kAz+dot(Omega,kA)*kAy)
    uAz=(-I*omegaAt*kAz*kAy+dot(Omega,kA)*kAx)*uAx/(-I*omegaAt*kAx*kAy-dot(Omega,kA)*kAz)
    uA=uAx*e.i+uAy*e.j+uAz*e.k
    pA=omegaAt*(kx*uAx-kz*uAz)/(kAx*kx-kAz*kz)
    uBx=1
    uBy=(-I*omegaBt*kBy*kBz-dot(Omega,kB)*kBx)*uBx/(-I*omegaBt*kBx*kBz+dot(Omega,kB)*kBy)
    uBz=(-I*omegaBt*kBz*kBy+dot(Omega,kB)*kBx)*uBx/(-I*omegaBt*kBx*kBy-dot(Omega,kB)*kBz)
    uB=uBx*e.i+uBy*e.j+uBz*e.k
    pB=omegaBt*(kx*uBx-kz*uBz)/(kBx*kx-kBz*kz)
    
    warnings.simplefilter('ignore')
    coeffAtau=dot((-dot(Omega,kA)*cross(kA,uA)-I*omegaAt*kAm**2*uA),e.i)
    coeffB=dot((-I*dot(Omega,kB-kw)*(dot(uwc,kB)*cross(kB-kw,uB)-dot(uB,kw)*cross(kB-kw,uwc))-(omegaBt-kz/k)*(dot(uwc,kB)*((kB-kw)*dot(-kw,uB)-dot(kB-kw,kB-kw)*uB)-dot(uB,kw)*((kB-kw)*dot(kB,uwc)-dot(kB-kw,kB-kw)*uwc))),e.i)
    coeffBtau=dot((-dot(Omega,kB)*cross(kB,uB)-I*omegaBt*kBm**2*uB),e.i)
    coeffA=dot((-I*dot(Omega,kA+kw)*(dot(uw,kA)*cross(kA+kw,uA)+dot(uA,kw)*cross(kA+kw,uw))-(omegaAt+kz/k)*(dot(uw,kA)*((kA+kw)*dot(kw,uA)-dot(kA+kw,kA+kw)*uA)+dot(uA,kw)*((kA+kw)*dot(kA,uw)-dot(kA+kw,kA+kw)*uw))),e.i)
    a=coeffB/(-I*coeffAtau)
    b=coeffA/(I*coeffBtau)
    s=sqrt(a*b)
    seval=lambdify([kAx,kAz],s.subs([(kBx,kAx),(kAy,kAyeval),(kBy,kAyeval),(kBz,kAz+1)]))
    growth=np.zeros(np.size(nv))
    freq=np.zeros(np.size(nv))
    for i in range(np.size(kAxv)):
       # if(i==0):
       #     growth[i]=0
       #     freq[i]=0
       # else:
            growth[i]=re(seval(kAxv[i],gamma+nv[i]))
            freq[i]=im(seval(kAxv[i],gamma+nv[i]))

    return kAxv,growth
      
# print(kAxv)
# print(growth)
# growth[np.isnan(growth)] = 0.
# growth[np.isinf(growth)] = 0.
# plt.plot(kAxv,growth,'kx')
# plt.show()

   # Solvability conditions (vectors i.e. 3 equations)
  #   gA = Atau*(-dot(Omega,kA)*cross(kA,uA)-I*omegaAt*kAm**2*uA)+B*(-I*dot(Omega,kB-kw)*(dot(uwc,kB)*cross(kB-kw,uB)-dot(uB,kw)*cross(kB-kw,uwc))-(omegaBt-kz/k)*(dot(uwc,kB)*((kB-kw)*dot(-kw,uB)-dot(kB-kw,kB-kw)*uB)-dot(uB,kw)*((kB-kw)*dot(kB,uwc)-dot(kB-kw,kB-kw)*uwc)))
  #   gB = Btau*(-dot(Omega,kB)*cross(kB,uB)-I*omegaBt*kBm**2*uB)+A*(-I*dot(Omega,kA+kw)*(dot(uw,kA)*cross(kA+kw,uA)+dot(uA,kw)*cross(kA+kw,uw))-(omegaAt+kz/k)*(dot(uw,kA)*((kA+kw)*dot(kw,uA)-dot(kA+kw,kA+kw)*uA)+dot(uA,kw)*((kA+kw)*dot(kA,uw)-dot(kA+kw,kA+kw)*uw)))
    
   #  gAx=dot(gA,e.i).expand()
  #   gBx=dot(gB,e.i).expand()
  #   a=(gAx.coeff(B))/(-I*gAx.coeff(Atau))
  #   b=(gBx.coeff(A))/(I*gBx.coeff(Btau))
  #   s=sqrt(a*b)
  #   eval=s.subs([(kAx,kAxeval),(kAy,kAyeval),(kAz,n),(kBx,kAxeval),(kBy,kAyeval),(kBz,n+1)])
  #   print(N(eval))
    
    # Alternative evaluation of the above
    # coeffAtau=dot((-dot(Omega,kA)*cross(kA,uA)-I*omegaAt*kAm**2*uA),e.i)
    # coeffB=dot((-I*dot(Omega,kB-kw)*(dot(uwc,kB)*cross(kB-kw,uB)-dot(uB,kw)*cross(kB-kw,uwc))-(omegaBt-kz/k)*(dot(uwc,kB)*((kB-kw)*dot(-kw,uB)-dot(kB-kw,kB-kw)*uB)-dot(uB,kw)*((kB-kw)*dot(kB,uwc)-dot(kB-kw,kB-kw)*uwc))),e.i)
    # coeffBtau=dot((-dot(Omega,kB)*cross(kB,uB)-I*omegaBt*kBm**2*uB),e.i)
    # coeffA=dot((-I*dot(Omega,kA+kw)*(dot(uw,kA)*cross(kA+kw,uA)+dot(uA,kw)*cross(kA+kw,uw))-(omegaAt+kz/k)*(dot(uw,kA)*((kA+kw)*dot(kw,uA)-dot(kA+kw,kA+kw)*uA)+dot(uA,kw)*((kA+kw)*dot(kA,uw)-dot(kA+kw,kA+kw)*uw))),e.i)
    # a=coeffB/(-I*coeffAtau)
    # b=coeffA/(I*coeffBtau)
    # s=sqrt(a*b)
    # eval=s.subs([(kAx,kAxeval),(kAy,kAyeval),(kAz,n),(kBx,kAxeval),(kBy,kAyeval),(kBz,n+1)])
    # print(N(eval)) # i component
    
    # coeffAtau=dot((-dot(Omega,kA)*cross(kA,uA)-I*omegaAt*kAm**2*uA),e.j)
    # coeffB=dot((-I*dot(Omega,kB-kw)*(dot(uwc,kB)*cross(kB-kw,uB)-dot(uB,kw)*cross(kB-kw,uwc))-(omegaBt-kz/k)*(dot(uwc,kB)*((kB-kw)*dot(-kw,uB)-dot(kB-kw,kB-kw)*uB)-dot(uB,kw)*((kB-kw)*dot(kB,uwc)-dot(kB-kw,kB-kw)*uwc))),e.j)
    # coeffBtau=dot((-dot(Omega,kB)*cross(kB,uB)-I*omegaBt*kBm**2*uB),e.j)
    # coeffA=dot((-I*dot(Omega,kA+kw)*(dot(uw,kA)*cross(kA+kw,uA)+dot(uA,kw)*cross(kA+kw,uw))-(omegaAt+kz/k)*(dot(uw,kA)*((kA+kw)*dot(kw,uA)-dot(kA+kw,kA+kw)*uA)+dot(uA,kw)*((kA+kw)*dot(kA,uw)-dot(kA+kw,kA+kw)*uw))),e.j)
    # a=coeffB/(-I*coeffAtau)
    # b=coeffA/(I*coeffBtau)
    # s=sqrt(a*b)
    # eval=s.subs([(kAx,kAxeval),(kAy,kAyeval),(kAz,n),(kBx,kAxeval),(kBy,kAyeval),(kBz,n+1)])
    # print(N(eval)) # j component
    
    # coeffAtau=dot((-dot(Omega,kA)*cross(kA,uA)-I*omegaAt*kAm**2*uA),e.k)
    # coeffB=dot((-I*dot(Omega,kB-kw)*(dot(uwc,kB)*cross(kB-kw,uB)-dot(uB,kw)*cross(kB-kw,uwc))-(omegaBt-kz/k)*(dot(uwc,kB)*((kB-kw)*dot(-kw,uB)-dot(kB-kw,kB-kw)*uB)-dot(uB,kw)*((kB-kw)*dot(kB,uwc)-dot(kB-kw,kB-kw)*uwc))),e.k)
    # coeffBtau=dot((-dot(Omega,kB)*cross(kB,uB)-I*omegaBt*kBm**2*uB),e.k)
    # coeffA=dot((-I*dot(Omega,kA+kw)*(dot(uw,kA)*cross(kA+kw,uA)+dot(uA,kw)*cross(kA+kw,uw))-(omegaAt+kz/k)*(dot(uw,kA)*((kA+kw)*dot(kw,uA)-dot(kA+kw,kA+kw)*uA)+dot(uA,kw)*((kA+kw)*dot(kA,uw)-dot(kA+kw,kA+kw)*uw))),e.k)
    # a=coeffB/(-I*coeffAtau)
    # b=coeffA/(I*coeffBtau)
    # s=sqrt(a*b)
    # eval=s.subs([(kAx,kAxeval),(kAy,kAyeval),(kAz,n),(kBx,kAxeval),(kBy,kAyeval),(kBz,n+1)])
    # print(N(eval)) # k component

# Aurelie's simplifications...
#uAx=(kAx*kAz-I*kAy*kAm)*(kAx*kAy+I*kAz*kAm)
#uAy=(kAy*kAz+I*kAx*kAm)*(kAx*kAy+I*kAz*kAm)
#uAz=(kAy*kAz-I*kAx*kAm)*(kAx*kAz-I*kAy*kAm)
#uA=uAx*e.i+uAy*e.j+uAz*e.k
#pA=(I*kAx*kAz+kAy*kAm)*(I*kAy*dot(kA,Omega)+kAm*dot(e.j,cross(Omega,kA)))/kAm
#uBx=(kBx*kBz-I*kBy*kBm)*(kBx*kBy+I*kBz*kBm)
#uBy=(kBy*kBz+I*kBx*kBm)*(kBx*kBy+I*kBz*kBm)
#uBz=(kBy*kBz-I*kBx*kBm)*(kBx*kBz-I*kBy*kBm)
#uB=uBx*e.i+uBy*e.j+uBz*e.k
#pB=(I*kBx*kBz+kBy*kBm)*(I*kBy*dot(kB,Omega)+kBm*dot(e.j,cross(Omega,kB)))/kBm
# Test eigenvector satisfies equations
#print(simplify(dot(-I*omegaAt*uA+cross(Omega,uA)+I*kA*pA,e.i)))

# plt.contourf(kAxv,kAyv,np.log10(np.transpose(growth)),rasterized=True)
# plt.pcolor(kAxv,kAyv,np.transpose(growth),cmap='jet',rasterized=True)
# plt.colorbar()
# plt.show()
#maxgrowth=np.amax(np.amax(growth))
#ind=np.unravel_index(np.argmax(growth), np.shape(growth))
#indx=ind[0];
#print('max growth: ',maxgrowth,'kAx: ',kAxv[indx],'kAy: ',nv[indx])
#maxgrowth=np.amax(growth[:])
#indx=np.argmax(growth[:]);
#print('max growth along x: ',maxgrowth,'kAx: ',kAxv[indx])
#growth[growth<0] = 0. # Remove negative/decaying modes

 #if(manyns):
 # Plot of growth rate on (\alpha,\beta) plane maximising over n
  #nv=-np.linspace(1,10,10)
  #kAxv=np.linspace(0,4,100)
 # kAyv=np.linspace(0,4,100)
  #warnings.simplefilter('ignore')
  #seval=lambdify([kAx,kAy,kAz],s.subs([(kBx,kAx+kx),(kBy,kAy),(kBz,kAz+kz)]))
  #growthn=np.zeros((np.size(kAxv),np.size(kAyv),np.size(nv)))
  #freqn=np.zeros((np.size(kAxv),np.size(kAyv),np.size(nv)))
  #maxgrowthv=np.zeros((np.size(kAxv),np.size(kAyv)))
  #ind=0   
  #for ind in range(np.size(nv)):
   # n=nv[ind]
   # for i in range(np.size(kAxv)):
   #   for j in range(np.size(kAyv)):
  #     growthn[i,j,ind]=re(seval(kAxv[i],kAyv[j],n))
   #    freqn[i,j,ind]=im(seval(kAxv[i],kAyv[j],n))
  #  ind=ind+1
  #growthn[np.isnan(growthn)] = 0.
  #growthn[np.isinf(growthn)] = 0.
 # growthn[growthn>100]=0. # fix...
 # maxgrowth=np.amax(np.amax(growthn))
 # ind=np.unravel_index(np.argmax(growthn), np.shape(growthn))
 # indx=ind[0]; indy=ind[1]; indn=ind[2]
 # print('max growth: ',maxgrowth,'kAx: ',kAxv[indx],'kAy: ',kAyv[indy],'n: ',nv[indn])
  #growthn[growthn<0] = 0. # Remove negative/decaying modes

# for i in range(np.size(kAxv)):
#    for j in range(np.size(kAyv)):
#      maxgrowthv[i,j]=np.squeeze(np.amax(growthn[i,j,:]))   
#plt.contourf(kAxv,kAyv,np.transpose(maxgrowthv),rasterized=True)
#plt.pcolor(kAxv,kAyv,np.transpose(maxgrowthv),cmap='jet',rasterized=True)
#plt.colorbar()
#plt.show()
