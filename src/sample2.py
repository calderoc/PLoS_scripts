import slds_util
import numpy as np
import scipy.linalg
import scipy.optimize as spo

if __name__ == "__main__":
    #setup example problem
    print "Illustrating How to Setup MA1 opt"
    
    dt = 40./1000.#time between observations in seconds  (adjusting for uneven sampling straightforward)
    sig = .25 #square root of diffusion coefficient (microns^2/s)
    sqrtR = 40/1000. #square root of effective measurement noise cov. (in nm)
    T = 200 #length of each time series
    N = 100 # number of trajectories to analyze
    #simulate a group of trajectories [sqrt(2) factor due legacy definitions used in stat. phys.]
    dW,W=slds_util.simDB(T=T,N=N,dt=dt,sig=sig*np.sqrt(2))

    #create observations
    Y=W+np.random.randn(W.shape[0],W.shape[1])*sqrtR

    #store the true parameters of the DGP for ref
    partrue= np.array([sig,sqrtR])

    resH=[] #store output in this list
    #find MLE pars of simple diffusion plus noise model on pathwise basis
    for i,yi in enumerate(Y):
        print 'Iteration:',i
        optobjecti = slds_util.CostFuncMA1Diff(yi,dt)
        res = spo.minimize(optobjecti.evalCostFunc, partrue/2., method='nelder-mead',options={'xtol': 1e-5, 'disp': False})
        # print res.x
        resH.append(np.abs(res.x)) #for simple diffusion plus noise model formulation +/- pars have identical likelihood
    resH=np.asarray(resH)
    print ''
    print '******* Result Summary ***********************'
    print ''
    print 'True (or Effective) Par:', partrue
    print ''
    print 'parameter means,medians, max, min of NxPar history vec:'
    print np.mean(np.abs(resH),axis=0) #takes abs value since optimization was unconstrained (cost function squares sig and sqrtR, so no diff;  physically both pars must be >0)
    print np.median(np.abs(resH),axis=0)
    print np.max(np.abs(resH),axis=0)
    print np.min(np.abs(resH),axis=0)
    print 'parameter STD of NxPar history vec:'
    print np.std(np.abs(resH),axis=0)