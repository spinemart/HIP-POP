#Courtesy of Emiliano Munari (INAF OATs)
import numpy as np
import matplotlib.pyplot as plt
import astropy.constants as cst
from scipy.integrate import odeint,trapz,quad
import scipy.interpolate as intp
import astropy.cosmology as csm


''' All units are "natural" (not in units of h) and are in Mpc for
distances and Msun for masses '''


class ClusterTools:
    def __init__(self,_redshift,cosmo=csm.Planck15,M200=-1,R200=-1,conc=-1,rs=-1,Delta=200.):
        self.cosmo = cosmo
        self.G = cst.G.to('km2 Mpc / (Msun s2)').value
        self.Delta = Delta
        self.redshift = _redshift

        if ((M200 >= 0.) & (R200 >= 0.)):
            raise Exception ('Either M200 or R200 must be provided, not both')
        if ((M200 < 0.) & (R200 < 0.)):
            raise Exception ('Either M200 or R200 must be provided')
        if ((conc >= 0.) & (rs >= 0.)):
            raise Exception ('Either concentration or scale radius must be provided, not both')
        
        if (M200 >= 0.):
            self.M200 = M200
            self.R200 = (2.*self.G*self.M200/(self.Delta*self.cosmo.H(self.redshift).value**2))**(1./3.)
        else:
            self.R200 = R200
            self.M200 = (self.R200**3*self.Delta*self.cosmo.H(self.redshift).value**2)/(2.*self.G)

        if (rs >= 0.):
            # Given the check made before, conc cannot be >= 0
            self.rs = rs
            self.conc = self.R200/self.rs
        else:
            if (conc >= 0):
                self.conc = conc
            else:
                self.conc = self.get_conc(self.M200)
            self.rs = self.R200/self.conc
        
        self.rho0 = self.M200/(4.*np.pi*self.rs**3*(np.log((self.rs+self.R200)/self.rs) - self.R200/(self.rs + self.R200)))
      
    def get_conc(self,mvir):
        # concentration parameter for LCDM 
        # (Maccio', Dutton & van den Bosch 08, relaxed halos, WMAP5, Delta=200)
        slope = -0.098
        norm = 6.76
        return norm*(mvir*self.cosmo.h/1.e12)**slope


class Jeans:
    def __init__(self,_CT,betaType='ID',xbeta=-1,ybeta=-1):
        self.CT = _CT
        if (betaType == 'ID'):
            rr = np.array([ 0.15112373,  0.29596034,  0.44821877,  0.60052861,  0.75289845,
                         0.89769221,  1.05014775,  1.20246616,  1.35105654,  1.49581602,
                         1.6519139 ,  1.80413804,  1.95259129,  2.10063317,  2.25642252,
                         2.40025642,  2.55558298,  2.69950258,  2.85163245])
            vv = np.array([ 0.05225225,  0.1027027 ,  0.11351351,  0.13513514,  0.16936937,
                         0.21081081,  0.26306306,  0.28648649,  0.32612613,  0.36036036,
                         0.37837838,  0.38198198,  0.39279279,  0.31711712,  0.27027027,
                         0.10990991, -0.03423423, -0.17657658, -0.19279279])
            self.setCustomBeta(rr*self.CT.R200,vv)
        if (betaType == 'ML'):
            self.Beta = self.BetaML
        if (betaType == 'T'):
            self.Beta = self.BetaT
        if (betaType == 'Custom'):
            if ((np.isscalar(xbeta) == True) | (np.isscalar(ybeta) == True)):
                raise Exception('You must provide a meaningful beta profile')
            self.setCustomBeta(xbeta,ybeta)
            
    def setCustomBeta(self,xbeta,ybeta):
        self.xbeta = np.concatenate([[0],xbeta,np.linspace(xbeta.max(),60,num=100)[1:]])
        self.ybeta = np.concatenate([[ybeta[0]],ybeta,np.ones(99)*ybeta[-1]])
        self.betaintp = intp.splrep(self.xbeta,self.ybeta,k=1)            
        self.Beta = self.BetaCustom
    def BetaT(self,x):
        return  0.5*x/(x+self.CT.rs)
    def BetaML(self,x):
        return  0.5*x/(x+0.18*self.CT.R200)
    def BetaCustom(self,x):
        return  intp.splev(x,self.betaintp)
            
    def densProfile(self,rr):
        return self.CT.rho0*self.nu(rr)
    def nu(self,rr):
        # not normalized
        return 1./( (rr/self.CT.rs) * (1. + rr/self.CT.rs)**2 )
    def nuRz(self,zz,RR):
        # not normalized
        return self.nu(self.get_r3D(RR,zz))
    def get_r3D(self,rproj,rlos):
        return np.sqrt(rproj**2 + rlos**2)
    def get_dlnudlr(self,x):
        return -( self.CT.rs/x + 2./(1 + x/self.CT.rs) )*x/self.CT.rs
    def get_MNFW(self,x):
        return 4.*np.pi*self.CT.rho0*self.CT.rs**3*( np.log((self.CT.rs + x)/self.CT.rs) - x/(self.CT.rs + x)  )
    def get_massprofile(self,x):
        return self.get_MNFW(x)
        


    

    def ode(self,yy,tt):
        ''' ODE to solve Jeans equation for sigma_r^2
        Input: 
        yy = sigma_r^2
        tt = radius
        Output:
        derivative of sigma_r^2 wrt radius, computed at radius tt
        '''
        A = self.get_dlnudlr(tt) + 2.*self.Beta(tt)
        B = self.CT.G*self.get_MNFW(tt)/tt
        dydt = -(A*yy + B)/tt
        return dydt

    def compute_sr_from_r(self,_rad):
        # We add a very distant point which represents the
        # "infinite". For ODE to converge, we must start from the
        # infinite and integrate toward rad=0
        rad = np.sort(np.concatenate([_rad,[60]]))[::-1]
        # This is the value at infinite. Integration is not sensitive
        # to the exact value of it
        sr2_0 = self.CT.G*self.get_massprofile(rad[0])/rad[0]/np.sqrt(2.) 
        sr2 = odeint(self.ode, sr2_0, rad).ravel()
        return rad[::-1][:-1], np.sqrt(sr2[::-1][:-1])




def velocityFromSphToCart(vr,vth,vphi,xx,yy,zz):
    RR2 = xx**2 + yy**2
    RR = np.sqrt(RR2)
    rad = np.sqrt(RR2 + zz**2)
    vx = (RR*vphi + xx*rad*vr/yy - xx*zz*zz*vr/(yy*rad) + xx*zz*RR*vth/(yy*rad))/(yy + xx*xx/yy)
    vz = (zz*(zz*vr-RR*vth)/rad)/zz
    vy = (rad*vr - zz*(zz*vr-RR*vth)/rad - xx*vx)/yy
    return vx,vy,vz


class Sphere:
    def __init__(self,_CT,betaType='ID',xbeta=-1.,ybeta=-1.):
        self.CT = _CT
        self.JJ = Jeans(self.CT,betaType,xbeta=xbeta,ybeta=ybeta)
        
    def densProb(self,x):
        ''' Probability density 
        Input:
        x: radius in Mpc
        Output:
        probability density

        Warning: this is NOT normalized
        '''
        # Density is in 3D, while the profile (like NFW) is 1-D, so we
        # have to multiply for 4 pi r**2
        return 4.*np.pi*self.JJ.densProfile(x)*x**2

    def normDist(self,x):
        "Must be normalized in the [a,b] range"
        return self.densProb(x)/self.norm

    
    def getRndRad(self,internalRadius,externalRadius,datasize,height_threshold=1.1):
        ## Before calling this method, the normalization of the prob
        ## (self.norm) must be set!
        hmax = height_threshold*self.normDist(np.linspace(internalRadius,externalRadius,num=1000)).max()
        NN = 2*datasize
        data = np.array([])
        exitstatus = 1
        while(True):
            yy = internalRadius+np.random.rand(NN)*(externalRadius-internalRadius)
            uu = np.random.rand(NN)
            fofy = self.normDist(yy)
            if ((fofy > hmax).any()):
                exitstatus = 0
                break
            vals = yy[(uu < fofy/hmax)]
            data = np.append(data,vals)
            if (data.size >= datasize):
                data = data[:datasize]
                break
        if (exitstatus == 0):
            print('Problem with the peak height in getRndRad. Rerunning with higher threshold')
            data = self.getRndRad(internalRadius,externalRadius,datasize,height_threshold=height_threshold*1.1)            
        return data


    def computePositions(self,internalRadius=1.e-3,externalRadius=-1,datasize=3000):
        datasize = np.int(datasize)
        if (externalRadius < 0.):
            externalRadius = 2.*self.CT.R200
        if (internalRadius >= externalRadius):
            print(externalRadius,self.CT.R200)
            raise Exception('internalRadius must be < externalRadius')
            
        # Normalize the probability
        self.norm = quad(self.densProb,internalRadius,externalRadius)[0]
        data = self.getRndRad(internalRadius,externalRadius,datasize)

        
        # Later, when the Jeans equation will be solved, results will
        # be returned ordered in radius. To be coherent with that,
        # here we sort by radius
        rad = np.sort(data)
        uu = 2.*np.random.rand(datasize)-1.
        thetarnd = np.random.rand(datasize)*2.*np.pi
        xx = rad*np.sqrt(1.-uu**2)*np.cos(thetarnd)
        yy = rad*np.sqrt(1.-uu**2)*np.sin(thetarnd)
        zz = rad*uu

        theta = np.arctan2(yy,xx)
        phi = np.arccos(zz/rad)

        return xx,yy,zz,rad,theta,phi

    def computeVelocities(self,xx,yy,zz):
        rad = np.sqrt(xx**2 + yy**2 + zz**2)

        sr = self.JJ.compute_sr_from_r(rad)[1]
        st = sr*np.sqrt(1.-self.JJ.Beta(rad))
        vr = np.random.normal(0,sr,sr.size)
        vth = np.random.normal(0,st,st.size)
        vphi = np.random.normal(0,st,st.size)

        vx,vy,vz = velocityFromSphToCart(vr,vth,vphi,xx,yy,zz)
        
        return sr,st,vr,vth,vphi,vx,vy,vz

    def computeSlosProfile(self,xx,yy,zz,vx,vy,vz,bins=10):
        allR = np.concatenate([np.sqrt(xx**2 + yy**2),np.sqrt(xx**2 + zz**2),np.sqrt(zz**2 + yy**2)])
        allV = np.concatenate([vz,vy,vx])
        Rbins = np.sort(allR)[np.ceil(np.linspace(0,allR.size-1,num=np.int(allR.size/10000))).astype(np.int)]        
        RR = np.zeros(Rbins.size-1)
        slos = np.zeros(Rbins.size-1)
        mask = np.zeros(RR.size,dtype=np.bool)
        for rr in xrange(RR.size):
            ww = (allR > Rbins[rr]) & (allR <= Rbins[rr + 1])
            if (ww.sum() > 0):
                RR[rr] = allR[ww].mean()
                slos[rr] = allV[ww].std()
                mask[rr] = True
        return RR[mask],slos[mask]
