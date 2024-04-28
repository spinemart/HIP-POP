#HIP-POP: HI Probe POPulator. Trieste 2020 
#Marta Spinelli (INAF OATs) - Acknowledgments: Tiago Castro (INAF OATs), Emiliano Munari (INAF OATs)

import numpy as np
import healpy as hp
import os,sys
import scipy
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib import rc
from pylab import cm
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
import HP_func as HPf
import SyntheticHalo as synt #courtesy of Emiliano Munari (INAF OATs)
import astropy.cosmology as CS

if __name__=='__main__':
    from optparse import OptionParser
    o = OptionParser()
    o.set_usage('%prog [option]')
    o.set_description(__doc__)
    o.add_option('--indir',dest='indir',default='../outdir/TEST/',help='input directory')
    o.add_option('--inname',dest='inname',default='TEST',help='input file name')
    o.add_option('--outdir',dest='outdir',default='../outdir/',help='output directory')
    o.add_option('--outname',dest='outname',default='TEST',help='output file name')
    o.add_option('--fmin',dest='fmin',default=950.,help='Min freq in MHz')
    o.add_option('--fmax',dest='fmax',default=1400.,help='Max freq in MHz')
    o.add_option('--Nf',dest='Nf',default=512,help='Number of freq bins')
    o.add_option('--seed',dest='seed',default=1234,help='seed')
    o.add_option('--nside',dest='nside',default=256,help='nside of healpy map')
    o.add_option('--verbose', default=False, help='Print information if True')

    '''
    NOTE: Check PLCAxis entry in Pinocchio par file (un-normalized direction of the cone axis) Default is: [1,1,1]
    '''
    #PLCAxis
    D=[1.,1.,1.]
    print("Assuming default PLCAxis in Pinocchio: Default is: [1,1,1]")
    #computing change of base matrix
    C = HPf.C_matrix(D)
    h = HPf.COSMO_h()
    Omegab = HPf.COSMO_Ob0()
    Omegam = HPf.COSMO_Om0()
    COSMO = CS.FlatLambdaCDM(H0=h*100, Om0=Omegam,Ob0=Omegab)  # Using H0 = 100 km/s/Mpc
    
    opts, args = o.parse_args(sys.argv[1:])
    print(opts,args)

    verbose=bool(opts.verbose)
    if verbose: print('Verbose!')

    indir=str(opts.indir)
    inname=str(opts.inname)
    outdir=str(opts.outdir)
    outname=str(opts.outname)
    fmin=float(opts.fmin)
    fmax=float(opts.fmax)
    Nf=int(opts.Nf)
    seed=int(opts.seed)
    nside=int(opts.nside)

    f_vec=np.linspace(fmin,fmax,num=Nf, endpoint=True)
    Df=f_vec[1]-f_vec[0]
    f_edges=(f_vec-Df/2.)
    f_edges=np.append(f_edges,f_vec[-1]+Df/2.)
    z_vec=HPf.f2z(f_vec)
    z_edges=HPf.f2z(f_edges)


    for ff in range(Nf): 
        z=z_vec[ff]
        zmax=z_edges[ff]
        zmin=z_edges[ff+1]
        fc=f_vec[ff]

        if verbose: print('freq index:  ',ff,'freq: ', fc, 'z: ',z, 'zmin-zmax: ',zmin,zmax)
    
        filename=indir+'plc_f'+str(ff)+'_'+inname+'.npz'
        if verbose: print('filename= ', filename)

        theta=np.load(filename)['theta']
        phi=np.load(filename)['phi']
        obsz=np.load(filename)['obsz']
        Mass=np.load(filename)['Mass']/1e10 #M0 h-1 10^10

        N=len(Mass)
        R_vir=HPf.R_vir(Mass,z) #in Mpc

        
        if verbose:
            print('Num of halos= ', N)
            print('Max Halo Mass= ', np.log10(np.max(Mass))+10)
            print('max R_vir= ', np.max(R_vir))

        
        #Code to transform from Mh to MHI
        #fix the seed
        np.random.seed(seed)
        
        #read the paramters fitted on GAEA
        popt=HPf.get_MHI_par(z)
        err=HPf.get_MHI_sigma(z,Mass)

        #compute MHI
        MHI=HPf.Mh2MHI(Mass,*popt)
       
        #start healpy pixelisation
        npix=hp.nside2npix(nside)
        R_pix=HPf.R_pix(nside,z)
        
        #check for pixel dimensions
        if verbose: print('NFW to better pixel location for how many halos?', len(np.where(2*R_vir>R_pix)[0]))

        #######
        smindex=np.where(2*R_vir>R_pix)[0]
        sm_R_vir=R_vir[smindex]
        sm_Mass=Mass[smindex] #value of the mass to smooth
        sm_MHI=MHI[smindex]
        sm_obsz=obsz[smindex]

        Nbh = len(sm_Mass)
        
        N_sub_halos=100
        for mm in range(Nbh):

            if verbose: print('halo loop', mm, 'of ', Nbh)
            #compute rho from redshift
            rho0 = HPf.z2rho(sm_obsz[mm]) #assuming cosmology
            theta0=theta[smindex[mm]] #in deg
            phi0=phi[smindex[mm]] #in deg

            ##from theta, phi in spherical coord in the rotated box to cartesian
            pos_r=HPf.spher2cart(rho0,theta0,phi0)
            pos_r=np.transpose(pos_r)
            #rotate back to box
            pos_b=HPf.plc2box(pos_r,C)


            M0=sm_Mass[mm]
            MHI0=sm_MHI[mm]
            #
            CT = synt.ClusterTools(sm_obsz[mm],M200=M0*1e10/h,cosmo=COSMO)
            sph = synt.Sphere(CT)
            pos = sph.computePositions(externalRadius=3.,datasize=N_sub_halos)[:3] #NOTE: output is in Mpc


            #remove old entries
            np.delete(obsz,smindex[mm])
            np.delete(theta,smindex[mm])
            np.delete(phi,smindex[mm])
            np.delete(Mass,smindex[mm])
            np.delete(MHI,smindex[mm])

            
            for ns in range(N_sub_halos):
                #compute the final position in the box in Mpc/h
                pos_new=[pos[0][ns]*h+pos_b[0],pos[1][ns]*h+pos_b[1],pos[2][ns]*h+pos_b[2]] 
                r,t,p = HPf.cart2spher(HPf.box2plc(pos_new,C))

                #add new ones
                obsz=np.append(obsz, obsz[smindex[mm]])
                theta=np.append(theta,t)
                phi=np.append(phi,p)
                Mass=np.append(Mass,M0/N_sub_halos)
                MHI=np.append(MHI,MHI0/N_sub_halos)
           
        #construct map
        if verbose: print("Constructing maps")
        mapMh=np.zeros(npix,dtype='float64')
        mapHI=np.zeros(npix,dtype='float64')
        mapTb=np.zeros(npix,dtype='float64')
        mapNh=np.zeros(npix)


        ipix = hp.ang2pix(nside,HPf.deg2rad(theta+90.), HPf.deg2rad(phi))

        

        u_pix, u_ind,i_inv, counts =np.unique(ipix, return_index=True, return_inverse=True, return_counts=True)
        if verbose: print("Number of unique pixels: ", len(u_pix), "while num of halos: ", N)
    
        #filling maps
        mapMh[u_pix]=np.bincount(np.searchsorted(u_pix,ipix),Mass)
        mapHI[u_pix]=np.bincount(np.searchsorted(u_pix,ipix),MHI)
       
        if verbose: print('max Mass:', np.log10(np.max(mapMh))+10)
        mapNh[u_pix] =  counts
        
        if verbose: 
            print('sanity check:', np.sum(mapNh))
            print('sum HI should be equal:', np.sum(MHI),np.sum(mapHI))

        #compute OmegaHI
        OmegaHI=HPf.Omega_HI(np.sum(MHI),zmin,zmax)
        OmegaHIz=HPf.OmegaHI_z(z)
        print('OmegaHI=', OmegaHI,OmegaHIz)

        #compute the final temperature brightness map
        mapTb = HPf.MHI2d(mapHI,nside,z,zmin,zmax)*HPf.Tbz(OmegaHIz,z)
        if verbose: 
            print('mean temp: ', np.mean(mapTb))
            print('max temp: ', np.max(mapTb))
        
        map_file=outdir+'maps_'+str(nside)+'_'+str(ff)+'_of'+str(Nf)+'_'+outname+'.npz'
        np.savez(map_file, mapMh=mapMh,mapNh=mapNh,mapHI=mapHI,mapTb=mapTb)   
        
    print('all maps saved')

