#HIP-POP: HI Probe POPulator. Trieste 2020 
#Marta Spinelli (INAF OATs) - Acknowledgments: Tiago Castro (INAF OATs), Emiliano Munari (INAF OATs)

import numpy as np
import healpy as hp
from scipy.interpolate import interp1d
import astropy.cosmology as CS

F_21=1420.405 #MHz

#Cosmology needs to be coherent with the Pinocchio simulation
C = 2.99e8 # SPEED OF LIGHT IN M/S
h=0.73
Omegab=0.045
Omegam=0.25
COSMO = CS.FlatLambdaCDM(H0=h*100, Om0=Omegam,Ob0=Omegab)  # Using H0 = 100 km/s/Mpc

def COSMO_h():
    return COSMO.h

def COSMO_Ob0():
    return COSMO.Ob0

def COSMO_Om0():
    return COSMO.Om0

def rad2deg(rad):
    return rad/np.pi*180.

def deg2rad(deg):
    return deg*np.pi/180.

def f2z(f):
    #freq in MHz
    return F_21/f-1.

def z2rho(z):
    '''
    Get spherical coordinate rho from redshift assuming a cosmology
    Output in Mpc/h
    '''
    return COSMO.comoving_distance(z).value*h

def rho_c(z):
    M0=1.99*1e30 #Kg
    rho_c=COSMO.critical_density(z).value #g/cm^3
    pc=3.08567758130573*1e16 #m
    rho_c=rho_c/(1e3*M0)*(pc*1e2)**3/(1e10*(h)**2)*(1e6)**3
    # 10^10 M0/(Mpc)^3 h^2
    return rho_c

def R_pix(nside,z):
    #in Mpc
    return hp.nside2pixarea(nside)**0.5*COSMO.comoving_transverse_distance(z).value

def R_vir(M,z):
    #in Mpc
    return (3*M/(200*rho_c(z)*4*np.pi))**(1./3.)/h 

def Mh2MHI(M,a1,a2,a,b,Mb,Mmin):
    '''
    from Mh and pars compute MHI
    NB. Mh and MHI in M0h-1/10^10
    '''
    return (a1*(M)**b*np.exp(-(M/Mb)**a)+a2)*M*np.exp(-(Mmin/M)**0.5)

def get_MHI_par(z):
    '''
    from files with fitted values compute popt for MHI
    '''
    npar=6
    filein='input/MHI_Mh_range01.npz'
    z_fit=np.load(filein)['z']
    mean_par=np.load(filein)['mean_par']
    print(mean_par.shape, z_fit.shape,z_fit)
    popt=np.zeros(npar)
    for i in range(npar):
        f=interp1d(z_fit,mean_par[:,i])
        popt[i]=f(z)    
    return popt

def get_MHI_sigma(z,Mh):
    '''
    from files with fitted values compute sigma
    NB. Mh in M0h-1/10^10
    '''
    npar_sig=8
    filein='input/sigma_coefs_range01.npz'
    coefs=np.load(filein)['coefs']
    save_coef=np.zeros(npar_sig)
    for j in range(npar_sig):
        p=np.poly1d(coefs[j])
        save_coef[j]=p(z)
    
    sigma=np.poly1d(save_coef)
    err=sigma(np.log10(Mh)+10)
    err[np.where(err<=0)]=1e-10
    return err

def cart2spher(xyz):
    '''
    Transform cartesian coordinates in spherical
    NOTE: output is in degree
    '''
    x,y,z = xyz
    rho = np.sqrt(x**2+y**2+z**2)
    theta = -1*np.arccos(z/rho)/np.pi*180+90.
    phi = np.arctan2(y,x)/np.pi*180
    if phi<0. : phi+=360.
    
    return np.array([rho, theta, phi])

def spher2cart(rho,theta,phi):
    '''
    Transform spherical coordinates in cartesian
    NOTE: input in degree
    '''
    theta=-theta*np.pi/180+np.pi/2
    x=rho*np.sin(theta)*np.cos(phi/180*np.pi)
    y=rho*np.sin(theta)*np.sin(phi/180*np.pi)
    z=rho*np.cos(theta)
    return np.array([x,y,z])

def C_matrix(D):
    '''
    Input: the direction D of Pinocchio plc (specified in the parameter file, if not default is 1,1,1)
    Output: change of base matrix C to go to the box coordinate to the plc
    '''
    D = D/np.linalg.norm(D)
    ivers=np.array([0.,0.,1.])
    p = np.cross(D,ivers)
    if np.linalg.norm(p)==0:
        ivers=np.array([1.,0.,0.])
        p = np.cross(D,ivers)
    p /= np.linalg.norm(p)
    q = np.cross(D,p)

    C=np.array([p,q,D])
    return C

def plc2box(pos,C):
    '''
    Change coordinates from plc back to box
    '''
    invC=np.linalg.inv(C)
    
    return np.dot(invC,pos)

def box2plc(pos,C):
    '''
    Change coordinates from the box to the plc
    '''
    return np.dot(C,pos)

def Omega_HI(MHItot,z1,z2):
    vol=4./3.*np.pi*(COSMO.comoving_distance(z2).value**3-COSMO.comoving_distance(z1).value**3)
    #print('vol=', vol)
    z=z1+(z2-z1)/2
    Omega_HI=MHItot/vol/h**2/rho_c(z) 
    return Omega_HI

def OmegaHI_z(z):
    print("I am reading Omega HI data from file")
    HI_file='input/rho_HI_z_split_bis_MII_Ms6.npz'
    OHI=np.load(HI_file)['rho']
    OmegaHIz=interp1d(np.arange(6),OHI)
    rhoc0=1.5e-7*(1e6)**3/(h**2*1e10)
    return OmegaHIz(z)*rhoc0/rho_c(z) 

def Tbz(OmegaHI,z):
    #compute Tb
    Omegab_z=COSMO.Ob(z)
    Omega_H=Omegab_z*(0.76) #fraction of baryons that are H
    x_HI=OmegaHI/Omega_H
    Tbz=23.88*x_HI*(Omegab*h**2/0.02)*np.sqrt((0.15/(Omegam*h**2))*((1.+z)/10)) #mK
    return Tbz
    
def MHI2d(MHI,nside,z,z1,z2):
    he=np.abs(COSMO.comoving_distance(z1).value-COSMO.comoving_distance(z2).value) #Mpc
    A=hp.nside2pixarea(nside)*COSMO.comoving_transverse_distance(z).value**2
    vol_pixel=he*A
    vol=4./3.*np.pi*(COSMO.comoving_distance(z2).value**3-COSMO.comoving_distance(z1).value**3)
    print(A,he,vol_pixel,vol,np.sum(MHI)/vol,MHI[MHI!=0]/vol_pixel)
    return (MHI/vol_pixel)/(np.mean(MHI)/vol_pixel)
