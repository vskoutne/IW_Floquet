from mpi4py import MPI
import numpy as np
import scipy.linalg as LA
import scipy.sparse as SM
import scipy.sparse.linalg as SLA
import matplotlib.pyplot as plt
import matplotlib
import time
from PSI import PSI_run

#world_comm = MPI.COMM_WORLD
#world_size = world_comm.Get_size()
#my_rank = world_comm.Get_rank()

Ek    = 0e-16
nv = 4
N_modes     = 32

beta=1
gamma = 0.5
#theta=30*np.pi/180
#print(theta)
#omega=np.sin(theta)
#print(omega)
omega=0.7
k=1
kz=omega*k
kx=np.sqrt(k**2-kz**2)
A = 0.1
#Aold=A*k**2/kx/kz
#To compare with old normalization, A_new= A_old*k**2/kx/kz and sigma_new*kz/k=sigma_old
s=1.0
Ek=Ek#*np.abs(kz)/np.sqrt(kx**2+kz**2)
flag=1

def getGR(A,Ek,N_res,N_modes,alphamax,betamax,gamma,kx,kz,s,nv,flag,beta):
    #alpharange, betarange = np.linspace(-alphamax,alphamax,N_res), [0]
    if flag==1:
     alpharange = np.linspace(0,alphamax,N_res)
    else:
     alpharange = [alphamax]
    betarange=[beta]
    k      = np.sqrt(kx**2+kz**2)
    om     = s*kz/k
    om_ratio=om/np.abs(om)
    abs_kz=np.abs(kz)
    target = A*om+1j*om

    Nbeta=np.shape(betarange)[0]
    Amat  = np.zeros((nv*N_modes,nv*N_modes),dtype=complex)
    Bmat  = np.zeros((nv*N_modes,nv*N_modes),dtype=complex)
    Bn    = np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,0.0]],dtype=complex)
    growthrate=np.zeros((N_res,Nbeta))
    for i in range(N_res):
    #for i in range(my_rank,N_res,world_size):
    	alpha=alpharange[i]
    	#print("alpha loop: i = "+str(i)+" out of: "+str(N_res))
    	for j in range(Nbeta):
    		#print("beta loop: j = "+str(j)+" out of: "+str(N_res))
    		beta=betarange[j]
    		for m in np.arange(N_modes):
    			n  = m-N_modes/2+1  #mode number starts at -N/2+1 and ends at +N/2
    			Ln = -alpha**2-beta**2-(gamma+n)**2
    			
    			if nv==4:
    				An = np.array([
    				[1j*kz/k*(gamma+n)+Ek*Ln, kz/k, 0.0, -1j*alpha],
    				[-kz/k, 1j*kz/k*(gamma+n)+Ek*Ln, -kx/k, -1j*beta],
    				[0.0, kx/k, 1j*kz/k*(gamma+n)+Ek*Ln, -1j*(gamma+n)],
    				[1j*beta*kz/k, -1j*kz/k*alpha-1j*kx/k*(gamma+n), 1j*beta*kx/k, Ln]
    				], dtype=complex)
                    
    				Anpo = np.array([
    				[1j*alpha*A*k/2.0/kx-beta*A*k/2.0/kx, 0.0, -1j*A*k/2.0/kx, 0.0],
    				[0.0, 1j*alpha*A*k/2/kx-beta*A*k/2.0/kx, A*k/2.0/kx, 0.0],
    				[0.0, 0.0, 1j*alpha*A*k/2.0/kx-beta*A*k/2.0/kx, 0.0],
    				[0.0, 0.0, -alpha*A*k/kx-1j*beta*A*k/kx, 0.0]
    				], dtype=complex)
    				
    				Anmo = np.array([
    				[1j*alpha*A*k/2.0/kx+beta*A*k/2.0/kx, 0.0,  1j*A*k/2.0/kx, 0.0],
    				[0.0, 1j*alpha*A*k/2/kx+beta*A*k/2.0/kx, A*k/2.0/kx, 0.0],
    				[0.0, 0.0, 1j*alpha*A*k/2.0/kx+beta*A*k/2.0/kx, 0.0],
    				[0.0, 0.0,  alpha*A*k/kx-1j*beta*A*k/kx, 0.0]
    				], dtype=complex)
    				''' New Valentin nondim
    				An = np.array([
    				[1j*om_ratio*(gamma+n)+Ek*Ln, kz/abs_kz, 0.0, -1j*alpha],
    				[-kz/abs_kz, 1j*om_ratio*(gamma+n)+Ek*Ln, -kx/abs_kz, -1j*beta],
    				[0.0, kx/abs_kz, 1j*om_ratio*(gamma+n)+Ek*Ln, -1j*(gamma+n)],
    				[1j*beta*kz/abs_kz, -1j*kz/abs_kz*alpha-1j*kx/abs_kz*(gamma+n), 1j*beta*kx/abs_kz, Ln]
    				], dtype=complex)
    				adv=1
    				Anpo = -np.array([
    				[1j*alpha*A/2.0-beta*s*A/2.0, 0.0, -1j*adv*A/2.0, 0.0],
    				[0.0, 1j*alpha*A/2-beta*s*A/2.0, s*adv*A/2.0, 0.0],
    				[0.0, 0.0, 1j*alpha*A/2.0-beta*s*A/2.0, 0.0],
    				[0.0, 0.0, -alpha*A-1j*beta*s*A, 0.0]
    				], dtype=complex)
    				
    				Anmo = -np.array([
    				[1j*alpha*A/2.0+beta*s*A/2.0, 0.0,  1j*adv*A/2.0, 0.0],
    				[0.0, 1j*alpha*A/2+beta*s*A/2.0, s*adv*A/2.0, 0.0],
    				[0.0, 0.0, 1j*alpha*A/2.0+beta*s*A/2.0, 0.0],
    				[0.0, 0.0,  alpha*A-1j*beta*s*A, 0.0]
    				], dtype=complex)
    				'''
    				Bmat[m*nv:(m+1)*nv,m*nv:(m+1)*nv]=Bn
    				
    			else:
    				An = np.array([
    				[1j*om_ratio*(gamma+n)+Ek*Ln+1j*alpha*1j*(beta*kz/abs_kz/Ln),kz/abs_kz+1j*alpha*(-1j*alpha*kz/abs_kz-1j*kx/abs_kz*(gamma+n))/Ln, 1j*alpha*1j*beta*kx/abs_kz/Ln],
    				[-kz/abs_kz+1j*beta*1j*beta*kz/abs_kz/Ln, 1j*om_ratio*(gamma+n)+Ek*Ln+1j*beta*(-1j*alpha*kz/abs_kz-1j*kx/abs_kz*(gamma+n))/Ln, -kx/abs_kz+1j*beta*1j*beta*kx/abs_kz/Ln],
    				[1j*(gamma+n)*1j*beta*kz/abs_kz/Ln, kx/abs_kz+1j*(gamma+n)*(-1j*alpha*kz/abs_kz-1j*kx/abs_kz*(gamma+n))/Ln, 1j*om_ratio*(gamma+n)+Ek*Ln +1j*(gamma+n)*1j*beta*kx/abs_kz/Ln]], dtype=complex)
    				'''
    				An = k/kz*np.array([[1j*kz/k*(gamma+n)+Ek*Ln+1j*alpha*1j*(beta*kz/k/Ln),kz/k+1j*alpha*(-1j*alpha*kz/k-1j*kx/k*(gamma+n))/Ln, 1j*alpha*1j*beta*kx/k/Ln],
    				[-kz/k+1j*beta*1j*beta*kz/k/Ln, 1j*kz/k*(gamma+n)+Ek*Ln+1j*beta*(-1j*alpha*kz/k-1j*kx/k*(gamma+n))/Ln, -kx/k+1j*beta*1j*beta*kx/k/Ln],
    				[1j*(gamma+n)*1j*beta*kz/k/Ln, kx/k+1j*(gamma+n)*(-1j*alpha*kz/k-1j*kx/k*(gamma+n))/Ln, 1j*kz/k*(gamma+n)+Ek*Ln +1j*(gamma+n)*1j*beta*kx/k/Ln]
    				], dtype=complex)
    				'''
    				Anpo = A*np.array([
    				[1j*alpha/2.0-beta*s/2.0, 0.0, -1j/2.0+1j*alpha*(-alpha/2.0-1j*beta*s/2.0)/Ln*2],
    				[0.0, 1j*alpha/2-beta*s/2.0, s/2.0+1j*beta*(-alpha/2.0-1j*beta*s/2.0)/Ln*2],
    				[0.0, 0.0, 1j*alpha/2.0-beta*s/2.0+1j*(gamma+n)*(-alpha/2.0-1j*beta*s/2.0)/Ln*2]], dtype=complex)
    				
    				Anmo = A*np.array([
    				[1j*alpha/2.0+beta*s/2.0, 0.0,  1j/2.0+1j*alpha*( alpha/2.0-1j*beta*s/2.0)/Ln*2],
    				[0.0, 1j*alpha/2+beta*s/2.0, s/2.0+1j*beta*( alpha/2.0-1j*beta*s/2.0)/Ln*2],
    				[0.0, 0.0, 1j*alpha/2.0+beta*s/2.0+1j*(gamma+n)*( alpha/2.0-1j*beta*s/2.0)/Ln*2]], dtype=complex)

    			Amat[m*nv:(m+1)*nv,m*nv:(m+1)*nv]=An

    			if m<N_modes-1:
    				Amat[m*nv:(m+1)*nv,(m+1)*nv:(m+2)*nv]=Anpo #need to shift the column index
    				#Amat[(m+1)*nv:(m+2)*nv,(m)*nv:(m+1)*nv]=Anpo #like in the original Matlab code with the bug

    			if m>0:
    				Amat[m*nv:(m+1)*nv,(m-1)*nv:m*nv]=Anmo #need to shift the column index
    				#Amat[(m-1)*nv:(m)*nv,(m)*nv:(m+1)*nv]=Anmo  #like in the original Matlab code with the bug

    		if nv==4:
    			[D,V]=LA.eig(Amat,Bmat)
    		else:
    			[D,V]=SLA.eigs(Amat,20,sigma=target)
    		D=np.real(D)
    		#print(D)	
    		D[np.isnan(D)]=0
    		D[np.isinf(D)]=0
    		D=np.sort(D)[::-1]
    		#print(D)
    		if D[0]>0:
    			growthrate[i,j]=D[0]#*kz/k
    temp=np.copy(growthrate)
    #world_comm.Allreduce( [growthrate, MPI.DOUBLE], [temp, MPI.DOUBLE], op = MPI.SUM )
    return temp

N_res=1000
alphamax=10
#alpharange = np.linspace(-alphamax,alphamax,N_res)
alpharange = np.linspace(0,alphamax,N_res)
growth=np.zeros(np.shape(alpharange))
growth= getGR(A,Ek,N_res,N_modes,alphamax,0,gamma,kx,kz,s,nv,1,beta)

alphaPSI,GRPSI=PSI_run(omega,k,kz,kx,A,beta,gamma)
print(alphaPSI)
print(GRPSI)

alpha_asymp = np.linspace(alphaPSI[2],1)
growth_asymp= getGR(A,Ek,1,N_modes,alphaPSI[2],0,gamma,kx,kz,s,nv,2,beta)
print('Error= ',np.abs(GRPSI[2]-growth_asymp)/GRPSI[2])

plt.plot(alpharange,growth,'k-')
plt.plot(alphaPSI,GRPSI,'bx')
plt.xlim([0,alphamax])
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$\sigma$')
plt.title(r'$\omega/2\Omega = 0.7, Ek=0, A=0.1, \gamma=0.5, \beta=1$')
plt.tight_layout()
plt.show()

