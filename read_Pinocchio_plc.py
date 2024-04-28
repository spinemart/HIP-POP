#HIP-POP: HI Probe POPulator. Trieste 2020 
#Marta Spinelli (INAF OATs) - Acknowledgments: Tiago Castro (INAF OATs), Emiliano Munari (INAF OATs)

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from optparse import OptionParser


class plc:
    '''
    Class to read Pinocchio past light-cone. From Pinocchio code repository.
    Group catalog on the Past Light Cone for a minimal mass of 10 particles
    1) group ID
    2) true redshift
    3-5) comoving position (Mpc/h)
    6-8) velocity (km/s)
    9) group mass (Msun/h)
    10) theta (degree)
    11) phi (degree)
    12) peculiar velocity along the line-of-sight (km/s)
    13) observed redshift
    '''
    
    def __init__(self,filename):
        if (not os.path.exists(filename)):
            print( "file not found:", filename)
            sys.exit()

        f = np.loadtxt(filename)
        print('data shape: ', f.shape)
        Ngroups =f.shape[0]
        print('Ngroups: ', Ngroups)

        #intitalization
        self.name = np.empty(Ngroups,dtype=np.uint64)
        self.redshift = np.empty(Ngroups,dtype=np.float64)
        self.pos = np.empty((Ngroups,3),dtype=np.float64)
        self.vel = np.empty((Ngroups,3),dtype=np.float64)
        self.Mass = np.empty(Ngroups,dtype=np.float64)
        self.theta = np.empty(Ngroups,dtype=np.float64)
        self.phi = np.empty(Ngroups,dtype=np.float64)
        self.vlos = np.empty(Ngroups,dtype=np.float64)
        self.obsz = np.empty(Ngroups,dtype=np.float64)

        #read the corresponding columns in the Pinocchio file
        self.name = f[:,0]
        self.redshift = f[:,1]
        self.pos = f[:,2:5]
        self.vel = f[:,5:8]
        self.Mass= f[:,8]
        self.theta = f[:,9]
        self.phi = f[:,10]
        self.vlos = f[:,11]
        self.obsz = f[:,12]
        
        del f



F_21=1420.405 #MHz
def f2z(f):
    '''
    freq in MHz
    '''
    return F_21/f-1.


if __name__=='__main__':
    
    o = OptionParser()
    o.set_usage('%prog [option]')
    o.set_description(__doc__)
    o.add_option('--plc_dir',dest='plc_dir',default='pinocchio',help='input directory')
    o.add_option('--pname',dest='pname',default='pinocchio.TEST.plc.out.',help='Pinocchio filename')
    o.add_option('--outdir',dest='outdir',default='../output/',help='output directory')
    o.add_option('--outname',dest='outname',default='TEST',help='Name the npz files')
    o.add_option('--Ntask',dest='Ntask',default=32,help='Number of task for Pinocchio input')
    o.add_option('--fmin',dest='fmin',default=950.,help='Min freq in MHz')
    o.add_option('--fmax',dest='fmax',default=1400,help='Max freq in MHz')
    o.add_option('--Nf',dest='Nf',default=512,help='Number of freq bins')
    o.add_option('--verbose', default=False, help='Print information if True')

    
    opts, args = o.parse_args(sys.argv[1:])
    print(opts,args)

    verbose=bool(opts.verbose)
    if verbose: print('Verbose!')

    plc_dir=str(opts.plc_dir)
    pname=str(opts.pname)
    outdir=str(opts.outdir)
    outname=str(opts.outname)
    Ntask=int(opts.Ntask)
    fmin=float(opts.fmin)
    fmax=float(opts.fmax)
    Nf=int(opts.Nf)

    #create frequency vector
    f_vec=np.linspace(fmin,fmax,num=Nf, endpoint=True)
    
    #corresponding frequency resolution
    Df=f_vec[1]-f_vec[0]
    #edges of the freq range
    f_edges=(f_vec-Df/2.)
    f_edges=np.append(f_edges,f_vec[-1]+Df/2.)
    #compute the corresponding redshift range
    z_vec=f2z(f_vec)
    z_edges=f2z(f_edges)

    #create frequencies file
    print('Frequency info are saved in a .txt')
    np.savetxt(outname+"_frequencies.txt", f_vec, delimiter=',')

    
    if verbose: print(f_vec, len(z_vec), z_edges, len(f_edges))

    #initialize array for reading data
    redshift=np.empty((Nf,),dtype=object) #true redshift
    name=np.empty((Nf,),dtype=object) #object ID 
    pos=np.empty((Nf,),dtype=object)  #comoving position (Mpc/h)
    vel=np.empty((Nf,),dtype=object)  #velocity (km/s)
    Mass=np.empty((Nf,),dtype=object) #group mass (Msun/h)
    theta=np.empty((Nf,),dtype=object) #theta (degree)
    phi=np.empty((Nf,),dtype=object) #phi (degree)
    vlos=np.empty((Nf,),dtype=object) #peculiar velocity along the line-of-sight (km/s)
    obsz=np.empty((Nf,),dtype=object) #observed redshift


    for nt in range(Ntask): 
        filename=plc_dir+pname+str(nt)
        if verbose: print( 'filename: ',filename)
        myplc = plc(filename)
        if verbose: print('sanity check! ', np.max(myplc.obsz),np.min(myplc.obsz), np.max(myplc.Mass))

        for f in range(Nf): 
        
            zmax=z_edges[f]
            zmin=z_edges[f+1]

            if verbose: print('zmin, zmax: ', zmin, zmax)
            cut=np.where((myplc.obsz>=zmin) & (myplc.obsz<zmax))
            print(len(cut[0]))
            if nt==0:

                if verbose: print('cutting from ',len(myplc.obsz),'to ', len(myplc.obsz[cut]))
                name[f] = myplc.name[cut].tolist()
                redshift[f] = myplc.redshift[cut].tolist()
                pos[f] = myplc.pos[cut].tolist()
                vel[f] = myplc.vel[cut].tolist()
                Mass[f] = myplc.Mass[cut].tolist()
                theta[f] = myplc.theta[cut].tolist()
                phi[f] = myplc.phi[cut].tolist()
                vlos[f] = myplc.vlos[cut].tolist()
                obsz[f] = myplc.obsz[cut].tolist()
            
            else:
            
                if len(cut[0])!=0:
                
                    if verbose: print('cutting from ',len(myplc.obsz),'to ', len(myplc.obsz[cut]))
                    name[f]+= myplc.name[cut].tolist()
                    redshift[f]+= myplc.redshift[cut].tolist()
                    pos[f]+= myplc.pos[cut].tolist()
                    vel[f]+= myplc.vel[cut].tolist()
                    Mass[f]+= myplc.Mass[cut].tolist()
                    theta[f]+= myplc.theta[cut].tolist()
                    phi[f]+= myplc.phi[cut].tolist()
                    vlos[f]+= myplc.vlos[cut].tolist()
                    obsz[f]+= myplc.obsz[cut].tolist()

    
            if verbose: print('cutting from ',len(myplc.obsz),'to ', len(myplc.obsz[cut]))




    if verbose: print("saving an npz file for each frequency")

    for f in range(Nf):  
        filename=outdir+outname+'/plc_f'+str(f)+'_'+outname+'.npz'
        np.savez(filename,name=name[f], redshift=redshift[f], pos=pos[f], vel=vel[f], Mass= Mass[f], theta = theta[f], phi = phi[f], vlos= vlos[f], obsz=obsz[f])

    print("npz files named ", outname, "saved in ", outdir)
    print("The End")

