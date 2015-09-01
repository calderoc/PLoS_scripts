
# Written by Chris Calderon 2015 (Chris.Calderon@UrsaAnalytics.com)
# set of scripts illustrating computations used in paper:
# Calderon & Bloom, PLOS ONE (2015)
#
# Copyright 2015 Ursa Analytics, Inc.

   # Licensed under the Apache License, Version 2.0 (the "License");
   # you may not use this file except in compliance with the License.
   # You may obtain a copy of the License at

   #     http://www.apache.org/licenses/LICENSE-2.0

__license__ = "Apache License, Version 2.0"
__author__  = "Chris Calderon, Ursa Analytics, Inc. <Chris.Calderon@UrsaAnalytics.com>"
__status__  = "Development"

import numpy as np
import scipy.linalg
import numpy.linalg as la
import scipy.sparse.linalg as sla
import scipy.sparse as sparse


class CostFuncMA1Diff(object):
    def __init__(self,tsData,dt): 
        """
        class for setting up costfunc of MA1 of differenced measurements in diffusion plus noise model
        dx_t= v dt+ sqrt(sqrtD*2)dBt
        y_ti = x_ti + \epsilon_i*sqrtR
        pars:=(sqrtD,sqrtR,v)
        dt:= scalar giving time sampling (can modify to vector dt if need be...see comments for code changes required)
        """
        

        self.__T=max(tsData.shape)-1 #compute length of differenced time series
        tsData=np.reshape(tsData,(self.__T+1,1),order='F')#reshape data to column vector (scalar time series assumed)
        self.__dy=np.diff(tsData,axis=0) #compute differenced time series (do not manipulate this, modify local copies via -pars[2]*dt)
        
        ii=np.arange(self.__T) #construct and store coo sparse indices for MA1
        self.__rowi=np.concatenate((ii,ii[:-1],ii[:-1]+1))
        self.__coli=np.concatenate((ii,ii[:-1]+1,ii[:-1]))
        self.__dt=dt #store internally to simplify interface to 3rd party opt functions

    def evalCostFuncVel(self,pars):
        """
        interface method to par opt routine
        MA1 of differenced measurements in diffusion plus noise model
        dx_t= v dt+ sqrt(sqrtD*2)dBt
        y_ti = x_ti + \epsilon_i*sqrtR
        pars:=(sqrtD,sqrtR,v)
        dt:= scalar giving time sampling (can modify to vector dt if need be...see comments for code changes required)
        """
        
        #compute MA1 covariance matrix
        S = self.constructMA1(pars)

        #compute eigenvalues and log(determinant) of MA1 covariance for loglikelihood computation
        # vals=self.sparseEigs(S) #comment out in production (keep in code since it can handle nonuniform dt case)
        ############################################
        # for uniform sampling cases, can just use closed form expression for evalues (see Supp Mat. Sec. 2.2.1. Calderon et al. JPCB 2013)
        valsE  = np.array([1.]*self.__T)*2.*pars[0]**2*self.__dt
        tmp    = np.arange(self.__T)+1
        valsE2 = (4.*pars[1]**2)*  ((np.sin(tmp*np.pi/2./(self.__T+1)))**2) #pure dirichlet boundary conditions
        vals   = valsE + valsE2 #
        #below sets up eigenvectors 
        # T=self.__T;V=np.array([[np.sqrt(2/(T+1.))*np.sin(i*j*np.pi/(T+1.)) for j in np.arange(T)+1] for i in np.arange(T)+1])
        # #norm of V*diag(vals)*V.T - full(S) should be zero (e.g., eigen decomposition)
        # S2 = np.dot(V.T,np.diag(vals))
        # S2 = np.dot(S2,V)
        ############################################
        loglikelihood=sum(np.log(vals))/2.
        
        #compute quadratic form contribution to log likelihood
        dy=self.__dy-pars[2]*self.__dt #
        #execute solve required to compute quadratic form
        tmp = self.sparseSolveWrapper(S,dy)
        quadForm = np.dot(dy.T,tmp)

        loglikelihood+=quadForm/2. #TODO:  replace this line by adding in quadratic form.
        #note negative of (unormalized) loglikelihood  computed above 


        return loglikelihood

    def evalCostFunc(self,pars):
        """
        interface method to par opt routine
        MA1 of differenced measurements in diffusion plus noise model
        dx_t= 0 dt+ sqrt(sqrtD*2)dBt
        y_ti = x_ti + \epsilon_i*sqrtR
        pars:=(sqrtD,sqrtR)
        dt:= scalar giving time sampling (can modify to vector dt if need be...see comments for code changes required)
        """
        
        #compute MA1 covariance matrix
        S = self.constructMA1(pars)

        #compute eigenvalues and log(determinant) of MA1 covariance for loglikelihood computation
        # vals=self.sparseEigs(S) #comment out in production (keep in code since it can handle nonuniform dt case)
        ############################################
        # for uniform sampling cases, can just use closed form expression for evalues (see Supp Mat. Sec. 2.2.1. Calderon et al. JPCB 2013)
        valsE  = np.array([1.]*self.__T)*2.*pars[0]**2*self.__dt
        tmp    = np.arange(self.__T)+1
        valsE2 = (4.*pars[1]**2)*  ((np.sin(tmp*np.pi/2./(self.__T+1)))**2) #pure dirichlet boundary conditions
        vals   = valsE + valsE2 #
        #below sets up eigenvectors 
        # T=self.__T;V=np.array([[np.sqrt(2/(T+1.))*np.sin(i*j*np.pi/(T+1.)) for j in np.arange(T)+1] for i in np.arange(T)+1])
        # #norm of V*diag(vals)*V.T - full(S) should be zero (e.g., eigen decomposition)
        # S2 = np.dot(V.T,np.diag(vals))
        # S2 = np.dot(S2,V)
        ############################################
        loglikelihood=sum(np.log(vals))/2.
        
        #compute quadratic form contribution to log likelihood
        # dy=self.__dy-pars[2]*self.__dt #
        dy=self.__dy-0.*self.__dt #
        #compute solve required to compute quadratic form
        tmp = self.sparseSolveWrapper(S,dy)
        quadForm = np.dot(dy.T,tmp)

        loglikelihood+=quadForm/2. #TODO:  replace this line by adding in quadratic form.
        #note negative of (unormalized) loglikelihood  computed above 


        return loglikelihood
        
    def sparseEigs(self,S):
        """
        compute eigenspectrum in parts for sparse SPD S of size nxn.  sparse symmetric eigen problem should be doable in one quick shot, but not currently possible in scipy.sparse.linalg    
        use krylov based eigensolver here to get full spectrum in two phases (built-in scipy funcs won't return full eigenspectrum)

        this routine is only needed for nonuniform time spacing case
        """ 
        k1 = int(np.ceil(self.__T/2.))
        vals1 = sla.eigsh(S, k=k1,return_eigenvectors=False,which='LM') 
        k2 = int(np.floor(self.__T/2.))
        vals2 = sla.eigsh(S, k=k2,return_eigenvectors=False,which='SM')
        vals=np.concatenate((vals1,vals2))

        return vals
    
    def constructMA1(self,pars):
        """
        precompute the coo sparse matrix indices of a tri-banded MA1  matrix (stored in rowi, coli) and return sparse coo mat
        pars:=(sqrtD,sqrtR,v)
        dt:= scalar giving time sampling (can modify to vector dt if need be)
        """
        #form sparse MA1 matrix
        R=pars[1]**2
        mainDiag=(2*R+self.__dt*2*pars[0]**2)*(np.array([1.]*self.__T)) #expression "np.array([1.]*N" like matlab ones(N,1) (with row/column left open)
        band=-R*(np.array([1.]*(self.__T-1)))
    
        svals=np.concatenate((mainDiag,band,band))
        svals=np.array([float(i) for i in svals]) #crude approach to computing a array with shape (T,) vs (T,1).  difference required for sparse
        S=sparse.coo_matrix((svals,(self.__rowi,self.__coli)),shape=(self.__T,self.__T))
        return S
    
    def sparseSolveWrapper(self,S,RHS):
    ###############Solver 1 [Mystery Method]##################### 
        # start_time = timeit.default_timer()
        # tmp = sla.spsolve(S.tocsc(),RHS) #this doesn't give details on how solve is done.  should exploit SPD nature of S for solve...unsure if scipy does this by default.  
        # elapsed = timeit.default_timer() - start_time
        # print 'tmystery', elapsed
        # print tmp
    ###############
        
    
    ############### Solver 2 [Explicit Call to SupaLU] ##################### 
        # start_time = timeit.default_timer()
        supalu = sla.splu(S.tocsc())
        tmp = supalu.solve(RHS.reshape(-1))
        # elapsed = timeit.default_timer() - start_time
        # print 'tsuperLU',elapsed 
        # print tmp
    ###############
         
        return tmp


def runOPT(sig1=.4,sig2=.4,sqrtR=35/100.,dt=10/1000.,N=10,T=50):
    #>> findMA1.runOPT(sig1=.4,sig2=.4,sqrtR=35/100.,dt=10/1000.,N=10,T=50)
    ts=np.arange(N)

    dW,W=simSDE.simDB(T=T,N=N,dt=dt,sig=sig1*np.sqrt(2))
    dW2,W2=simSDE.simDB(T=T,N=N,dt=dt,sig=sig2*np.sqrt(2))
    

    W2+=np.reshape(W[:,-1],(N,1)) #create a smooth transition by adding terminal value of 
    print W.shape
    W=np.hstack((W,W2))
    print W.shape

    Y=W+np.random.randn(W.shape[0],W.shape[1])*sqrtR
    fracsplit=.5 #adjust this parameter to reflect mix of sig1 and sig2 in sampled data
    sigEff = np.sqrt(fracsplit*2*sig1**2+fracsplit*2*sig2**2) 
    sigEff = sigEff/2. #make sigEff^2= D_Eff 
    Xtrue= np.array([sigEff,sqrtR,0])
    
    #iterate over paths and carry out optimization
    resH=[]
    for i,yi in enumerate(Y):
        print 'Iteration:',i
        costInstance = CostFuncMA1Diff(yi,dt)
        res = spo.minimize(costInstance.evalCostFunc, Xtrue/2., method='nelder-mead',options={'xtol': 1e-5, 'disp': False})
        print res.x
        resH.append(res.x)
    # res = spo.minimize(test2.evalCostFunc, Xtrue, method='nelder-mead',options={'xtol': 1e-4, 'disp': True})
    resH=np.asarray(resH)

    print '******* Result Summary ***********************'
    print ''
    print 'True (or Effective) Par:', Xtrue
    print ''
    print 'parameter means,medians, max, min of NxPar history vec:'
    print np.mean(np.abs(resH),axis=0) #takes abs value since optimization was unconstrained (cost function squares sig and sqrtR, so no diff;  physically both pars must be >0)
    print np.median(np.abs(resH),axis=0)
    print np.max(np.abs(resH),axis=0)
    print np.min(np.abs(resH),axis=0)
    print 'parameter STD of NxPar history vec:'
    print np.std(np.abs(resH),axis=0)

    print 'Ddt/2R:', sig1**2*dt/(2.*sqrtR**2)
    fname = '~/optRes_dt_'+str(int(dt*1000)) + '_sig1_' + str(sig1) +  '_sig2_' + str(sig1) +'.h5'
    print "saving  results to file: ", fname
    wouth5(fname,resH)

def simDB(N=1000,T=20,dt=2./100.,sig=1): #write out an ASCII file to "floc" given a 2D numpy array "DATAMAT"
    """
      simulate batch of increments and paths of pure BM.  
    """
    sdt=np.sqrt(dt)
    dW=np.random.randn(N,T)*sdt*sig;
    W=np.cumsum(dW,axis=1)
    return dW,W

def SDEnoiseFromDiscreteModel2D(F,Sigma,dt):
    # """
    # solves 2D equation A*Sigma + Sigma*A' = F*Q*F'-Q  [mixed discrete and continuous parameters in def.]
    # for Q [instantaneous covariance of continuous time SDE].  
    # F and Sigma are the 2x2 discrete versions corresponding to continuous time SDE model
    # Note:  all square matrices assumed full rank in current sample version.
    #
    # corresponding continuous SDE:
    # dX_t=A*X_tdt+sqrtm(Q)dB_t
    # where sqrtm() corresponds to the matrix square root (e.g., use "scipy.linalg.sqrtm")
    # discrete version
    # X_{n+1}= expm(A*del)*X_n + sqrtm(Sigma)*randn(dimX,1); [mixed discrete and continuous parameters in def.]

    # Inputs:  "F"      2x2 np.array corresponding to discrete linear drift term
 #             "Sigma"  np.array corresponding to discrete SLDS Gaussian Covariance
 #             "dt"        time between observations (float)
    # """
    #establish "RHS"
    A = scipy.linalg.logm(F)/dt
    RHSmat = np.dot(A,Sigma) + np.dot(Sigma,A.T)
    RHS =np.array([ RHSmat[0,0],RHSmat[1,1],RHSmat[1,0]]) #keep element ordering consistent throughout
    RHS = RHS.reshape((3,1),order='F')
    

    Fk = F
    F=F.reshape((4,1),order='F') #overwrite discrete F with continuous A elements
    
    #form coefficient matrix solving for components of Q 
    C=np.zeros((3,3)) 
    C[0,:]=[F[0]**2-1.,  F[2]**2,  2*F[0]*F[2]]
    C[1,:]=[F[1]**2,  F[3]**2-1.,  2*F[1]*F[3]]  
    C[2,:]=[F[0]*F[1],  F[2]*F[3],  F[1]*F[2]+F[0]*F[3]-1.]  
    
    
    tmp = scipy.linalg.solve(C,RHS)
    Q= np.array([[tmp[0][0], tmp[2][0]],[tmp[2][0], tmp[1][0]]]) #put back into standard matrix form ([0] ending artifact of ndarrays)
    #Fk*Sigma+Sigma*Fk'-(Fk*Q*Fk'-Q) #if setup correctly, this quantity should be zero.
    QsqrtM=scipy.linalg.sqrtm(Q)
    return Q,QsqrtM





