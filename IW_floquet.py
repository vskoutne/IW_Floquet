from mpi4py import MPI
import numpy as np
import scipy.linalg as LA
import scipy.sparse as SM
import scipy.sparse.linalg as SLA
import matplotlib.pyplot as plt
import matplotlib
import time

world_comm = MPI.COMM_WORLD
world_size = world_comm.Get_size()
my_rank = world_comm.Get_rank()




#if scanning A and omega
N_res_A_om  = 8
N_res_gamma = 1


alphamax,betamax = 2,2 # currently scans positive beta
nv          = 4      # 3 or 4
N_res_ab    = 8     # Resolution of alpha_beta plane
N_modes     = 16
Ek          = 0#1e-5
theta       = 10
norm        = 2      #1 is time normalization wrt 2\Omega, 2 is time normalization wrt \omega
A           = 0.2

savename = 'A%d_theta%d_Ek0e0_Nab%d_Nmodes%d_nv%d'%(int(100*A),theta,N_res_ab,N_modes,nv)
# savename = 'A25e-2_theta%d_Ek0e0_Nab%d_Nmodes%d_nv%d'%(theta,N_res_ab,N_modes,nv)
readtxt  = 0

#To compare with old normalization, A_new = A_old*k**2/kx/kz and sigma_new = sigma_old*k/kz

s     = 1.0
gamma = 0
kx    = np.sin(theta*np.pi/180.0)
kz    = np.cos(theta*np.pi/180.0)
k     = np.sqrt(kz**2+kx**2)
# norm  = 1
#A     = 0.2/(k**2/kx/kz)

def GR_at_alpha_beta_gamma(A,Ek,N_modes,alpha,beta,gamma,kx,kz,s,nv,norm):    
    k        = np.sqrt(kx**2+kz**2)
    om       = s*kz/k
    om_ratio = om/np.abs(om)
    abs_kz   = np.abs(kz)
    target   = A*om+1j*om
    Amat     = np.zeros((nv*N_modes,nv*N_modes),dtype=complex)
    Bmat     = np.zeros((nv*N_modes,nv*N_modes),dtype=complex)
    Bn       = np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],[0.0,0.0,0.0,0.0]],dtype=complex)
    for m in np.arange(N_modes):
        n  = m-N_modes/2+1  #mode number starts at -N/2+1 and ends at +N/2
        Ln = -alpha**2-beta**2-(gamma+n)**2        
        if nv==4:
            if norm==1:# time normalized wrt 2\Omega
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
            if norm==2:# time normalized wrt \omega
                An = np.array([
                [1j*om_ratio*(gamma+n)+Ek*Ln, kz/abs_kz, 0.0, -1j*alpha],
                [-kz/abs_kz, 1j*om_ratio*(gamma+n)+Ek*Ln, -kx/abs_kz, -1j*beta],
                [0.0, kx/abs_kz, 1j*om_ratio*(gamma+n)+Ek*Ln, -1j*(gamma+n)],
                [1j*beta*kz/abs_kz, -1j*kz/abs_kz*alpha-1j*kx/abs_kz*(gamma+n), 1j*beta*kx/abs_kz, Ln]
                ], dtype=complex)
                adv=1#toggle if want to turn off advective term
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
            
            Bmat[m*nv:(m+1)*nv,m*nv:(m+1)*nv]=Bn
                    
        else:
            if norm==1:
                An = np.array([[1j*kz/k*(gamma+n)+Ek*Ln+1j*alpha*1j*(beta*kz/k/Ln),kz/k+1j*alpha*(-1j*alpha*kz/k-1j*kx/k*(gamma+n))/Ln, 1j*alpha*1j*beta*kx/k/Ln],
                [-kz/k+1j*beta*1j*beta*kz/k/Ln, 1j*kz/k*(gamma+n)+Ek*Ln+1j*beta*(-1j*alpha*kz/k-1j*kx/k*(gamma+n))/Ln, -kx/k+1j*beta*1j*beta*kx/k/Ln],
                [1j*(gamma+n)*1j*beta*kz/k/Ln, kx/k+1j*(gamma+n)*(-1j*alpha*kz/k-1j*kx/k*(gamma+n))/Ln, 1j*kz/k*(gamma+n)+Ek*Ln +1j*(gamma+n)*1j*beta*kx/k/Ln]
                ], dtype=complex)
                A=A*k/kx
            if norm==2:
                An = np.array([
                [1j*om_ratio*(gamma+n)+Ek*Ln+1j*alpha*1j*(beta*kz/abs_kz/Ln),kz/abs_kz+1j*alpha*(-1j*alpha*kz/abs_kz-1j*kx/abs_kz*(gamma+n))/Ln, 1j*alpha*1j*beta*kx/abs_kz/Ln],
                [-kz/abs_kz+1j*beta*1j*beta*kz/abs_kz/Ln, 1j*om_ratio*(gamma+n)+Ek*Ln+1j*beta*(-1j*alpha*kz/abs_kz-1j*kx/abs_kz*(gamma+n))/Ln, -kx/abs_kz+1j*beta*1j*beta*kx/abs_kz/Ln],
                [1j*(gamma+n)*1j*beta*kz/abs_kz/Ln, kx/abs_kz+1j*(gamma+n)*(-1j*alpha*kz/abs_kz-1j*kx/abs_kz*(gamma+n))/Ln, 1j*om_ratio*(gamma+n)+Ek*Ln +1j*(gamma+n)*1j*beta*kx/abs_kz/Ln]], dtype=complex)
                
            Anpo = A*np.array([
            [1j*alpha/2.0-beta*s/2.0, 0.0, -1j/2.0+1j*alpha*(-alpha/2.0-1j*beta*s/2.0)/Ln*2],
            [0.0, 1j*alpha/2-beta*s/2.0, s/2.0+1j*beta*(-alpha/2.0-1j*beta*s/2.0)/Ln*2],
            [0.0, 0.0, 1j*alpha/2.0-beta*s/2.0+1j*(gamma+n)*(-alpha/2.0-1j*beta*s/2.0)/Ln*2]], dtype=complex)
            
            Anmo = A*np.array([
            [1j*alpha/2.0+beta*s/2.0, 0.0,  1j/2.0+1j*alpha*( alpha/2.0-1j*beta*s/2.0)/Ln*2],
            [0.0, 1j*alpha/2+beta*s/2.0, s/2.0+1j*beta*( alpha/2.0-1j*beta*s/2.0)/Ln*2],
            [0.0, 0.0, 1j*alpha/2.0+beta*s/2.0+1j*(gamma+n)*( alpha/2.0-1j*beta*s/2.0)/Ln*2]], dtype=complex)

        Amat[m*nv:(m+1)*nv,m*nv:(m+1)*nv] = An
        if m<N_modes-1:
            Amat[m*nv:(m+1)*nv,(m+1)*nv:(m+2)*nv] = Anpo #need to shift the column index
        if m>0:
            Amat[m*nv:(m+1)*nv,(m-1)*nv:m*nv] = Anmo #need to shift the column index
    if nv==4:
        [D,V] = LA.eig(Amat,Bmat)
    else:
        [D,V] = SLA.eigs(Amat,20,sigma=target)

    Dreal = np.real(D)    
    D[np.isnan(Dreal)] = 0
    D[np.isinf(Dreal)] = 0
    i_sigma_max = Dreal.argmax()
    sigma_max   = D[i_sigma_max]
    v_max       = V[:,i_sigma_max]
    
    return sigma_max,v_max

        
def get_alpha_beta_plane(A,Ek,N_res,N_modes,alphamax,betamax,gamma,kx,kz,s,nv,norm):
    #return a 2d array containing the growth rate for each alpha,beta

    #Scan only positive beta
    N_res_alpha,N_res_beta = N_res,int(N_res*betamax/alphamax/2)
    alpharange, betarange  = np.linspace(-alphamax,alphamax,N_res_alpha), np.linspace(0,betamax,N_res_beta)
    growthrate=np.zeros((N_res_alpha,N_res_beta))
    for i in range(my_rank,N_res_alpha,world_size):
        alpha=alpharange[i]
        for j in range(N_res_beta):
            beta=betarange[j]
            sigma, un = GR_at_alpha_beta_gamma(A,Ek,N_modes,alpha,beta,gamma,kx,kz,s,nv,norm)
            sigma = np.real(sigma)
            if sigma>0:
                growthrate[i,j]=sigma#*kz/k
    temp=np.copy(growthrate)
    world_comm.Allreduce( [growthrate, MPI.DOUBLE], [temp, MPI.DOUBLE], op = MPI.SUM )
    return temp

def get_alpha_beta_plane_singlecore(A,Ek,N_res,N_modes,alphamax,betamax,gamma,kx,kz,s,nv,norm):
    #return a 2d array containing the growth rate for each alpha,beta
    #Scan only positive beta
    N_res_alpha,N_res_beta = N_res,int(N_res*betamax/alphamax/2)
    alpharange, betarange  = np.linspace(-alphamax,alphamax,N_res_alpha), np.linspace(0,betamax,N_res_beta)
    growthrate=np.zeros((N_res_alpha,N_res_beta))
    for i in range(my_rank,N_res_alpha):
        alpha=alpharange[i]
        for j in range(N_res_beta):
            beta=betarange[j]
            sigma, un = GR_at_alpha_beta_gamma(A,Ek,N_modes,alpha,beta,gamma,kx,kz,s,nv,norm)
            sigma = np.real(sigma)
            if sigma>0:
                growthrate[i,j]=sigma#*kz/k
    return growthrate



def plotGR_alphabeta(A,Ek,N_res,N_modes,alphamax,betamax,gamma,kx,kz,s,nv,norm, savename, readtxt):
    if not readtxt:
        growthrate = get_alpha_beta_plane(A,Ek,N_res,N_modes,alphamax,betamax,gamma,kx,kz,s,nv,norm)
        if my_rank==0:
            np.savetxt(savename+"_ab.txt",growthrate)
    if my_rank==0:
        growthrate=np.loadtxt(savename+"_ab.txt")
        N_res_alpha,N_res_beta=N_res,int(N_res*betamax/alphamax/2)
        alpharange, betarange = np.linspace(-alphamax,alphamax,N_res_alpha), np.linspace(0,betamax,N_res_beta)
        k      = np.sqrt(kx**2+kz**2)
        om     = kz/k
        FNTSZ=24
        lw=3        
        m=np.unravel_index(growthrate.argmax(), growthrate.shape)#find the mode with the highest growth rate
        alpha_at_max,beta_at_max=alpharange[m[0]],betarange[m[1]]
        sigma,v=GR_at_alpha_beta_gamma(A,Ek,N_modes,alpha_at_max,beta_at_max,gamma,kx,kz,s,nv,norm)

        print("Highest GR: "+str(sigma))
        ############# alpha beta plane plot ###############
        fig,ax=plt.subplots(1,1,figsize=(11,8))
        if norm==1:
            sigma_norm=(A*k/np.abs(om))
        if norm==2:
            sigma_norm = A
        sigma_norm=1*kz/k  #if you don't want to normalize the growth rate
        pmesh=ax.pcolormesh(alpharange,betarange,growthrate.T/sigma_norm,cmap="cividis")#,shading='gouraud')
        #pmesh=ax.contourf(alpharange,betarange,growthrate.T/(A*k/np.abs(kx*om)),cmap="cividis")
        #ax.scatter([ alpha_at_max ],[ beta_at_max ],s=100,color='red',marker='*')
        #ax.scatter([ -alpha_at_max ],[ -beta_at_max ],s=100,color='red',marker='*')
        cbar=fig.colorbar(pmesh,ax=ax)

        cbar.ax.tick_params(labelsize=FNTSZ)
        cbar.set_label("$\\frac{\\sigma}{A'|\\omega|}$",fontsize=FNTSZ+8,rotation=0,labelpad=30)
        ax.set_xlabel("$\\alpha$",fontsize=FNTSZ+4)
        ax.set_ylabel("$\\beta$",fontsize=FNTSZ+4,rotation=0,labelpad=10)

        ax.tick_params(labelsize=FNTSZ)
        #ax.set_aspect("equal")
        #plt.title("s="+str(A)+", Ek="+str(Ek)+", $\\omega/2\\Omega=$%0.2f"%(om)+", $\\gamma=$"+str(gamma)+", N="+str(N_modes)+", nv="+str(nv),fontsize=FNTSZ+4)
        plt.tight_layout()
        plt.savefig(savename+".png")
        plt.show()
        
        ############ plot of eigenvectors for the fastest growing mode ###########
        u_n=v[::nv]
        v_n=v[1::nv]
        w_n=v[2::nv]
        alpha2,beta2=-alpha_at_max,beta_at_max
        sigma2,v=GR_at_alpha_beta_gamma(A,Ek,N_modes,alpha2,beta2,gamma,kx,kz,s,nv,norm)
        u2_n=v[::nv]#[::-1]
        v2_n=v[1::nv]#[::-1]
        w2_n=v[2::nv]#[::-1]
        
        fig,ax=plt.subplots(1,1,figsize=(11,8))
        ax.plot(np.linspace(-N_modes/2+1,N_modes/2,N_modes),np.real(u_n),linewidth=lw,label="Re($u_n$) for $\\alpha,\\beta$",color='r')
        ax.plot(np.linspace(-N_modes/2+1,N_modes/2,N_modes),np.real(u2_n),linewidth=lw,label="Re($u_n$) for $\\alpha_2,\\beta_2$",color='r',linestyle=":")
        ax.plot(np.linspace(-N_modes/2+1,N_modes/2,N_modes),np.real(v_n),linewidth=lw,label="Re($v_n$) for $\\alpha,\\beta$",color='g')
        ax.plot(np.linspace(-N_modes/2+1,N_modes/2,N_modes),np.real(v2_n),linewidth=lw,label="Re($v_n$) for $\\alpha_2,\\beta_2$",color='g',linestyle=":")
        ax.plot(np.linspace(-N_modes/2+1,N_modes/2,N_modes),np.real(w_n),linewidth=lw,label="Re($w_n$) for $\\alpha,\\beta$",color='blue')
        ax.plot(np.linspace(-N_modes/2+1,N_modes/2,N_modes),np.real(w2_n),linewidth=lw,label="Re($w_n$) for $\\alpha_2,\\beta_2$",color='blue',linestyle=":")
        ax.legend(fontsize=FNTSZ-8,loc=0)
        ax.axvline(x=0,color='gray',linestyle=":")
        ax.set_xlabel("$n$",fontsize=FNTSZ+4)
        ax.set_ylabel("Eigenvector Components"+", N="+str(N_modes)+", nv="+str(nv),fontsize=FNTSZ,rotation=90,labelpad=10)
        ax.tick_params(labelsize=FNTSZ)
        plt.title("$(\\alpha,\\beta,\\gamma)=$(%0.1f,%0.1f,%0.1f)"%(alpha_at_max,beta_at_max,gamma)+", $(\\alpha_2,\\beta_2,\\gamma)=$(%0.1f,%0.1f,%0.1f)"%(alpha2,beta2,gamma)+", A=%0.2f"%(A)+", $(k_x,k_z)=$(%0.1f,%0.1f)"%(kx,kz),fontsize=FNTSZ-4)
        plt.tight_layout()
        plt.savefig(savename+"_eigenvector_un_Re.png")

        fig,ax=plt.subplots(1,1,figsize=(11,8))
        ax.plot(np.linspace(-N_modes/2+1,N_modes/2,N_modes),np.imag(u_n),linewidth=lw,label="Im($u_n$) for $\\alpha,\\beta$",color='r')
        ax.plot(np.linspace(-N_modes/2+1,N_modes/2,N_modes),np.imag(u2_n),linewidth=lw,label="Im($u_n$) for $\\alpha_2,\\beta_2$",color='r',linestyle=":")
        ax.plot(np.linspace(-N_modes/2+1,N_modes/2,N_modes),np.imag(v_n),linewidth=lw,label="Im($v_n$) for $\\alpha,\\beta$",color='g')
        ax.plot(np.linspace(-N_modes/2+1,N_modes/2,N_modes),np.imag(v2_n),linewidth=lw,label="Im($v_n$) for $\\alpha_2,\\beta_2$",color='g',linestyle=":")
        ax.plot(np.linspace(-N_modes/2+1,N_modes/2,N_modes),np.imag(w_n),linewidth=lw,label="Im($w_n$) for $\\alpha,\\beta$", color='blue')
        ax.plot(np.linspace(-N_modes/2+1,N_modes/2,N_modes),np.imag(w2_n),linewidth=lw,label="Im($w_n$) for $\\alpha_2,\\beta_2$", color='blue',linestyle=":")
        ax.text(-N_modes/2+1,0.4,"$\\sigma(%0.2f,%0.2f)=$"%(alpha_at_max,beta_at_max)+str(sigma))
        ax.text(-N_modes/2+1,0.35,"$\\sigma(%0.2f,%0.2f)=$"%(alpha2,beta2)+str(sigma2))
        ax.legend(fontsize=FNTSZ-8,loc=0)
        ax.axvline(x=0,color='gray',linestyle=":")
        ax.set_xlabel("$n$",fontsize=FNTSZ+4)
        ax.set_ylabel("Eigenvector Components"+", N="+str(N_modes)+", nv="+str(nv),fontsize=FNTSZ,rotation=90,labelpad=10)
        ax.tick_params(labelsize=FNTSZ)
        plt.title("$(\\alpha,\\beta,\\gamma)=$(%0.1f,%0.1f,%0.1f)"%(alpha_at_max,beta_at_max,gamma)+", $(\\alpha_2,\\beta_2,\\gamma)=$(%0.1f,%0.1f,%0.1f)"%(alpha2,beta2,gamma)+", A=%0.2f"%(A)+", $(k_x,k_z)=$(%0.1f,%0.1f)"%(kx,kz),fontsize=FNTSZ-4)
        #plt.tight_layout()
        plt.savefig(savename+"_eigenvector_un_Im.png")


        plt.show()


def scan_n_save_A_vs_omega(Ek,s,N_res_A_om,N_res_ab,N_modes,alphamax,betamax,N_res_gamma,nv,norm,savename):
    growthrate = np.zeros((N_res_A_om,N_res_A_om))
    gamma      = np.linspace(0,1-1.0/N_res_gamma,N_res_gamma)
    theta      = np.linspace(3.0,87.0,N_res_A_om)*np.pi/180
    kx         = np.sin(theta)
    kz         = np.cos(theta)
    k          = np.sqrt(kx**2+kz**2)
    om         = kz/k
    A          = np.logspace(-1,0,N_res_A_om)

    for i in range(my_rank,N_res_A_om,world_size):
        #print("s loop: i = "+str(i)+" out of: "+str(N_res_A_om))
        for j in range(N_res_A_om):
            sigma=0
            for k in range(N_res_gamma):
                growthrate_temp=get_alpha_beta_plane_singlecore(A[i],Ek,N_res_ab,N_modes,alphamax,betamax,gamma[k],kx[j],kz[j],s,nv,norm)
                #growthrate_temp=getGR(s[i],Ek,N_res_ab,N_modes,alphamax,betamax,gamma[k],kx,kz[j],nv)
                sigma=max(sigma, growthrate_temp.max()) #find the mode with the highest growth rate
            growthrate[i,j]=sigma
    temp=np.copy(growthrate)
    world_comm.Allreduce( [growthrate, MPI.DOUBLE], [temp, MPI.DOUBLE], op = MPI.SUM )
    if my_rank==0:
        f = open(savename+".txt",'w')
        for i in range(N_res_A_om):
            f.write(" ".join(map(str, temp[i,:]))+"\n")
        f.close()
    
def plot_scan_A_vs_om(Ek,N_res_A_om,nv,savename):
    cmap=matplotlib.cm.plasma
    FNTSZ=24
    lw=2
    skip=1
    theta      = np.linspace(1.0/N_res_A_om,90-1.0/N_res_A_om,N_res_A_om)*np.pi/180
    kx         = np.sin(theta)
    kz         = np.cos(theta)
    k          = np.sqrt(kx**2+kz**2)
    om         = kz/k
    A          = np.logspace(-1,0,N_res_A_om)
    growthrate=np.loadtxt(savename+".txt")

    fig,ax=plt.subplots(1,1,figsize=(10,8))
    
    norm = matplotlib.colors.Normalize(vmin=np.min(np.log10(A)),vmax=np.max(np.log10(A)))
    s_m  = matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm)
    s_m.set_array([])

    for i in range(N_res_A_om)[::skip]:
        ax.plot(om,growthrate[i,:]/A[i]/om,linewidth=lw,color=s_m.to_rgba(np.log10(A[i]) ))#label="$s$=%.2f"%(s[i]))
    cbar=plt.colorbar(s_m,ax=plt.gca())
    cbar.set_label("$\\log(A')$",rotation=90,fontsize=FNTSZ)
    cbar.ax.tick_params(labelsize=FNTSZ-4)
    ax.set_xlabel("$\\frac{\\omega}{2\\Omega}$",fontsize=FNTSZ+4)
    ax.set_ylabel("$\\frac{\\sigma}{A'|\\omega|}$",fontsize=FNTSZ+4,rotation=0,labelpad=20)
    #ax.set_xscale('log')
    #ax.set_yscale('log')

    #SNOOPY comparison
    om_marker=[1/np.sqrt(6**2+1**2),1/np.sqrt(4**2+1**2),1/np.sqrt(3**2+1**2),1/np.sqrt(2**2+1**2),1/np.sqrt(1**2+1**2),2/np.sqrt(1**2+2**2),3/np.sqrt(1**2+3**2)]
    sigma_marker=[1.63,1.18,0.95,0.72,0.59,0.79,1.1]
    ax.scatter(om_marker,sigma_marker,marker='x',color='black',s=100,zorder=100)
    
    ax.tick_params(labelsize=FNTSZ-4)
    plt.tight_layout()
    plt.savefig(savename+"_om_vs_gr.png")
    
    fig,ax=plt.subplots(1,1,figsize=(10,8))
    for i in range(N_res_A_om)[::skip]:
        ax.plot(A,growthrate[:,i]/A/om[i],label="$\\omega$=%.2f"%(om[i]))
    ax.legend()
    ax.set_xlabel("$A'$",fontsize=FNTSZ+4)
    ax.set_ylabel("$\\frac{\\sigma}{A'|\\omega|}$",fontsize=FNTSZ+4,rotation=0,labelpad=10)
    
    #ax.set_ylim(top=om[-1])
    #ax.set_xlim(right=s[-1])
    ax.tick_params(labelsize=FNTSZ)
    plt.title("Ek="+str(Ek)+", N="+str(N_modes)+", nv="+str(nv),fontsize=FNTSZ+4)
    plt.tight_layout()
    plt.savefig(savename+"_s_vs_gr.png")

    fig,ax=plt.subplots(1,1,figsize=(10,8))
    #pmesh=ax.pcolormesh(s,om,np.log10(growthrate.T),cmap="cividis",shading='gouraud')
    pmesh=ax.contourf(A,om,np.log10(growthrate.T),cmap="cividis",levels=20)
    cbar=fig.colorbar(pmesh,ax=ax)

    cbar.ax.tick_params(labelsize=FNTSZ)
    cbar.set_label("$\\sigma$",fontsize=FNTSZ+4,rotation=0)
    ax.set_xlabel("$A'$",fontsize=FNTSZ+6)
    ax.set_ylabel("$\\frac{\\omega}{2\\Omega}$",fontsize=FNTSZ+6,rotation=0,labelpad=10)
    ax.set_xscale('log')
    #ax.set_yscale('log')
    #ax.set_ylim(top=om[-1])
    #ax.set_xlim(right=A[-1])
    ax.tick_params(labelsize=FNTSZ)
    #ax.set_aspect("equal")
    plt.title("Ek="+str(Ek)+", N="+str(N_modes)+", nv="+str(nv),fontsize=FNTSZ+4)
    plt.tight_layout()
    plt.savefig(savename+".png")
    plt.show()

def writeEigenmode(A,Ek,N_res,N_modes,gamma,kx,kz,nv,alpha,beta):
    #write the real and imag components of the eigenvector of the fastest growing mode for a given alpha, beta
    #then plot the eigenmode and the divergence versus n
    sigma,v=getMode(A,Ek,N_res,N_modes,alpha,beta,gamma,kx,kz,nv)
    start=0#int(N_modes/2)-1
    print("Modes")
    print(np.linspace(-N_modes/2+1,N_modes/2,N_modes))
    n_index=np.linspace(-N_modes/2+1,N_modes/2,N_modes)
    nummodes=N_modes
    showmaxn=N_modes/2
    u_n_re=np.real(v[::nv])[start:start+nummodes]
    v_n_re=np.real(v[1::nv])[start:start+nummodes]
    w_n_re=np.real(v[2::nv])[start:start+nummodes]
    
    u_n_im=np.imag(v[::nv])[start:start+nummodes]
    v_n_im=np.imag(v[1::nv])[start:start+nummodes]
    w_n_im=np.imag(v[2::nv])[start:start+nummodes]
    #for i in range(nummodes):
    #    print(v[::nv]+i*v[2::nv])
    eigenvector=np.asarray([u_n_re,u_n_im,v_n_re,v_n_im,w_n_re,w_n_im])
    np.savetxt("temp.txt",eigenvector,fmt='%.6e',delimiter=',')
    
    u_n=v[::nv]
    v_n=v[1::nv]
    w_n=v[2::nv]
    div_vel=np.abs(alpha*u_n+beta*v_n+n_index*w_n)
    print("Growth rate:"+str(sigma))
    #print(u_n)
    lw=2
    FNTSZ=16
    fig,ax=plt.subplots(1,1,figsize=(10,8))
    ax.plot(n_index,u_n_re,linewidth=lw,color='red',label="$u_n$ real")
    ax.plot(n_index,u_n_im,linewidth=lw,linestyle=":",color='red',label="$u_n$ imag")
    ax.plot(n_index,v_n_re,linewidth=lw,color='blue',label="$v_n$ real")
    ax.plot(n_index,v_n_im,linewidth=lw,linestyle=":",color='blue',label="$v_n$ imag")
    ax.plot(n_index,w_n_re,linewidth=lw,color='green',label="$w_n$ real")
    ax.plot(n_index,w_n_im,linewidth=lw,linestyle=":",color='green',label="$w_n$ imag")
    ax.set_xlim(-showmaxn,showmaxn)
    ax.legend(fontsize=FNTSZ+4)
    ax.axvline(x=0,color='gray',linestyle=":")
    ax.set_xlabel("$n$",fontsize=FNTSZ+4)
    ax.set_ylabel("Abs(eigenvector)",fontsize=FNTSZ+4,rotation=90,labelpad=10)
    ax.tick_params(labelsize=FNTSZ)
    plt.title("s="+str(A)+", Ek="+str(Ek)+", $\\alpha=$"+str(alpha)+", $\\beta=$"+str(beta)+", $\\gamma=$"+str(gamma),fontsize=FNTSZ+4)
    plt.tight_layout()
    #plt.show()
    
    fig,ax=plt.subplots(1,1,figsize=(10,8))
    ax.plot(n_index,div_vel,linewidth=lw)
    ax.set_xlim(-showmaxn,showmaxn)
    #ax.axvline(x=0,color='gray',linestyle=":")
    ax.set_xlabel("$n$",fontsize=FNTSZ+4)
    ax.set_ylabel("$|\\vec{k} \\cdot \\vec{u}(\\vec{k})| $",fontsize=FNTSZ+4,rotation=90,labelpad=10)
    ax.tick_params(labelsize=FNTSZ)
    plt.title("s="+str(A)+", Ek="+str(Ek)+", $\\alpha=$"+str(alpha)+", $\\beta=$"+str(beta)+", $\\gamma=$"+str(gamma),fontsize=FNTSZ+4)
    plt.tight_layout()
    plt.show()


time_start = time.time()


plotGR_alphabeta(A,Ek,N_res_ab,N_modes,alphamax,betamax,gamma,kx,kz,s,nv,norm, savename, readtxt)


#Uncomment if the scan of A vs \omega is needed
#scan_n_save_A_vs_omega(Ek,s,N_res_A_om,N_res_ab,N_modes,alphamax,betamax,N_res_gamma,nv,norm,savename)
#if my_rank==0:
#    plot_scan_A_vs_om(Ek,N_res_A_om,nv,savename)
#writeEigenmode(A,Ek,N_res_ab,N_modes,gamma,kx,kz,nv,alpha=0.0,beta=4.0)
if my_rank==0:
    time_end = time.time()
    elapsedTime = int(time_end - time_start)
    print(f'The elapsed time is {elapsedTime} seconds')
