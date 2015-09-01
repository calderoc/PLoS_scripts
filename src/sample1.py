import slds_util
import numpy as np
import scipy.linalg

if __name__ == "__main__":
    #setup example problem
    print "Solving Benchmark Problem"

    dt = 40./1000.

    B = np.array([[-4., 2],[-4 ,-.5]])
    F = scipy.linalg.expm(B*dt)
    sqrtQref = np.array([[.2,.1],[.1 ,.4]]) #covariance corresponding to parameters above

    #Sigma below is the discrete sampling version of the continuous time SDE with specific parameters defined above (A=0)
    Sigma = np.array([[1.879470064891504e-03,2.295287869259251e-03],[2.295287869259247e-03,6.297905889674040e-03]])
    # print F,dt,Qref
    Q,QsqrtM = slds_util.SDEnoiseFromDiscreteModel2D(F,Sigma,dt)
    print "Norm below should be close to machine double precision zero:"
    print np.linalg.norm(sqrtQref - QsqrtM)