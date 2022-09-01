import numpy as np
import matplotlib.pyplot as plt

#nos of inputs, nos of steps is Δt, T=time, r= Int. rate, sigma= volatility,S_0=initital value of   
def GeneratePathsGBM(NoOfPaths,NoOfSteps,T,r,sigma,S_0):
    
    Z = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps]) # standard rand no. [matrix with nos of paths & nos of steps]
    X = np.zeros([NoOfPaths,NoOfSteps+1]) # process of ND
    S = np.zeros([NoOfPaths,NoOfSteps+1]) # exponent of X
    time = np.zeros([NoOfSteps+1])
    
    X[:,0] = np.log(S_0) # first coloum of matrix  X0= log return of initial stock
    
    dt = T / float(NoOfSteps) #dt=	Δt/steps to simulate our paths
    for i in range(0,NoOfSteps):
        #making sure the samples is in N.D it helps us to achieve better convergence
        if NoOfPaths > 1:
            Z[:,i] = (Z[:,i] - np.mean(Z[:,i])) / np.std(Z[:,i])
            
     #for next coloum of matrix we will do 
     #we simulate under log space we simulate the process X which is log of process S
        X[:,i+1] = X[:,i] + (r - 0.5 * sigma **2 ) * dt + sigma *        np.power(dt, 0.5)*Z[:,i]
        time[i+1] = time[i] +dt
   
    #Compute exponent of AM
    S = np.exp(X)
    paths = {"time":time,"S":S}
    return paths


def MainCode():
    NoOfPaths = 8
    NoOfSteps = 1000
    S_0 = 1
    r = 0.05   #risk neutal rate
    mu = 0.15  #on historical data or option future behaviour
    sigma = 0.1
    T = 1
    
    #Money savings account
    M = lambda t: np.exp(r * t)
    
    #Monte Carlo Paths
    pathsQ = GeneratePathsGBM(NoOfPaths,NoOfSteps,T,r,sigma,S_0)
    S_Q = pathsQ["S"]
    pathsP = GeneratePathsGBM(NoOfPaths,NoOfSteps,T,mu,sigma,S_0)
    S_P = pathsP["S"]
    time = pathsQ["time"]
                 
    #Discounted stock Paths
    S_Qdisc = np.zeros([NoOfPaths,NoOfSteps+1])           
    S_Pdisc = np.zeros([NoOfPaths,NoOfSteps+1]) 

    #Compute quantitites i.e ratio between stock at any time/money saving accounts
    # as we know S/M is martingale So martingle means value is constant over time
    i = 0
    for i, ti in enumerate(time):
        S_Qdisc[:,i] = S_Q[:,i]/M(ti)        
        S_Pdisc[:,i] = S_P[:,i]/M(ti)   #In real world measure the martingle means value is constant over time, will not constant
                 
    #S(T)/M(T) with stock growing with rate r
    plt.figure(1)
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("S(t)")
    eSM_Q = lambda t: S_0 * np.exp(r * t)/ M(t)
    plt.plot(time,eSM_Q(time), 'r--')
    plt.plot(time, np.transpose(S_Qdisc),'blue')   
    plt.legend(['E^Q[S(t)/M(t)]','paths S(t)/M(t)'])            
    
    # S(T)/M(T) with Stock growing with rate mu
    plt.figure(2)
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("S(t)")
    eSM_P = lambda t: S_0 * np.exp(mu *t) / M(t)
    plt.plot(time,eSM_P(time),'r--')
    plt.plot(time, np.transpose(S_Pdisc),'blue')   
    plt.legend(['E^P[S(t)/M(t)]','paths S(t)/M(t)'])
    
    
MainCode()