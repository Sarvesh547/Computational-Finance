import numpy as np
import scipy.stats as st

V_market = 135.1
k=  1500;      
tau = 1;      
r = 0.021;     
S_0 = 1500;    
sigmaInit = 0.18;  
CP = "c"   


#Main core is computation of IV
def ImpliedVolatility(CP,S_0,k,sigma,tau,r):
    error = 1e10; #initial error it sould be verg large, if it to small calibration could stop 
    #handy lamvda expression
    optPrice = lambda sigma: BS_Call_Option_Price(CP,S_0,k,sigma,tau,r) 
    vega= lambda sigma: dV_dsigma(S_0,k,sigma,tau,r)
    
    
   
    #follow the iteration
    n = 1.0
    while error>10e-10:    # we iterate as value is bigger than 10^(-10)
        f         = V_market - optPrice(sigma);
        f_prim    = -vega(sigma)
        sigma_new = sigma - f / f_prim
        
        error=abs(sigma_new-sigma) 
        sigma=sigma_new
        
        print('iteration {0} with error = {1}'.format(n,error))
        
        n = n+1
    return sigma


def dV_dsigma(S_0,k,sigma,tau,r):
    #parameter and value of vega
    d2 = (np.log(S_0 / float(k)) + (r - 0.5 * np.power(sigma,2.0)) * tau) / float(sigma * np.sqrt(tau))
    value = k * np.exp(-r * tau)* st.norm.pdf(d2) * np.sqrt(tau)
    return value

def BS_Call_Option_Price (CP,S_0,k,sigma,tau,r):
    #Black-Scholes Call option price
    d1  =(np.log(S_0 / float(k)) + (r + 0.5 * np.power(sigma,2.0)) * tau) / float(sigma * np.sqrt(tau))
    d2  =d1 - sigma* np.sqrt(tau)
    if str(CP).lower()=="c" or str(CP).lower=="1":
        value = st.norm.cdf(d1) * S_0 -st.norm.cdf(d2) * k * np.exp(-r * tau)
    elif str(CP).lower()=="p" or str(CP).lower=="-1":
        value = st.norm.cdf(-d2) * k * np.exp(-r * tau) - st.norm.cdf(-d1) * S_0
    return value   #output is option value

sigma_imp = ImpliedVolatility(CP,S_0,k,sigmaInit,tau,r)
message  = '''Implied volatility for CallPrice= {}, strike K={}, 
      maturity T= {}, interest rate r= {} and initial stock S_0={} 
      equals to sigma_imp = {:.7f}'''.format(V_market,k,tau,r,S_0,sigma_imp)

print(message)

val = BS_Call_Option_Price(CP,S_0,k,sigma_imp,tau,r)
print('Option Price for implied volatility of {0} is equal to {1}'.format(sigma_imp, val))