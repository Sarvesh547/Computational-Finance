
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st


#We generate Brownain motion paths
def GeneratePathsCorrelatedBM(NoOfPaths,NoOfSteps,T,rho):
    Z1 = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps]) #randon samples ND
    Z2 = np.random.normal(0.0,1.0,[NoOfPaths,NoOfSteps]) #randon samples ND
    W1 = np.zeros([NoOfPaths, NoOfSteps+1]) #empty martix for storage
    W2 = np.zeros([NoOfPaths, NoOfSteps+1]) #empty martix for storage
    
    
    dt = T / float(NoOfSteps)
    time = np.zeros([NoOfSteps+1])
    for i in range(0,NoOfSteps):    #every time step we stad our sample to ND
        #making sure sample is ND
        if NoOfPaths > 1:
            Z1[:,i] = (Z1[:,i] - np.mean(Z1[:,i])) / np.std(Z1[:,i])
            Z2[:,i] = (Z2[:,i] - np.mean(Z2[:,i])) / np.std(Z2[:,i])
            
        #Corelate noise 
        #for Z1 we have independent sample and for 
        #Z2 rho * Z1[:,i] + np.sqrt(1.0 - rho**2) * 2nd vector
        Z2[:,i]= rho * Z1[:,i] + np.sqrt(1.0 - rho**2) * Z2[:,i]
        
        #Generate brownain motion with iterative process
        W1[:,i+1] = W1[:,i] + np.power(dt, 0.5)*Z1[:,i]
        W2[:,i+1] = W2[:,i] + np.power(dt, 0.5)*Z2[:,i]
        
        time[i+1] = time[i] + dt
        
    
    paths = {"time":time,"W1":W1,"W2":W2} #we use dictionary to store result 
    return paths

def mainCalculation():
    NoOfPaths = 1    #for per process W1 & W2 we have 1 path 
    NoOfSteps = 500
    T = 1.0  #1year
    
# We have 3 case 
    # negative correlation
    rho =-0.9
    Paths = GeneratePathsCorrelatedBM(NoOfPaths,NoOfSteps,T,rho)
    timeGrid = Paths["time"]
    W1 = Paths["W1"]
    W2 = Paths["W2"]
    
    plt.figure(1)
    plt.plot(timeGrid, np.transpose(W1))   
    plt.plot(timeGrid, np.transpose(W2))   
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("W(t)")
    
    
    # positive correlation
    rho =0.9
    Paths = GeneratePathsCorrelatedBM(NoOfPaths,NoOfSteps,T,rho)
    timeGrid = Paths["time"]
    W1 = Paths["W1"]
    W2 = Paths["W2"]
    
    plt.figure(2)
    plt.plot(timeGrid, np.transpose(W1))   
    plt.plot(timeGrid, np.transpose(W2))   
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("W(t)")
    
    # Zero correlation
    rho =0.0
    Paths = GeneratePathsCorrelatedBM(NoOfPaths,NoOfSteps,T,rho)
    timeGrid = Paths["time"]
    W1 = Paths["W1"]
    W2 = Paths["W2"]
    
    plt.figure(3)
    plt.plot(timeGrid, np.transpose(W1)) 
    plt.plot(timeGrid, np.transpose(W2))   
    plt.grid()
    plt.xlabel("time")
    plt.ylabel("W(t)")
    
mainCalculation()
