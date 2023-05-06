from functools import cache
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
import numpy as np
import scipy.stats as si
import math
import random
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.sparse

# Read Data File
data = pd.read_csv('dataset/DataSet-l1.csv')

# Extract option parameters from data
# print (data)
S01 = data['Spot_Price']  # spot price at t=0
K0 = data['Strike_Price'] # strike price
r0 = data['rate']/100 # risk-free interest rate
T0 = data['time']# time to expiration in years
sigma0 = data['volatility'] # volatility of the underlying asset
LTP0 = data["LTP"]
Symbol0 = data["Symbol"]
Date0 = data["date"]
expiry0 = data["expiry"]
open0 = data["open"]
close0 = data["Close"]


length = len(S01)

csv = {
    "Symbol":[],
    "date":[],
    "expiry":[],
    "open":[],
    "close":[],
    "Strike_Price":[],
    "Spot_Price":[],
    "Risk_Free_Intrest":[],
    "Time":[],
    "Volatility":[],
    "LTP":[],
    "BS_Call":[],
    "MC_Call":[],
    "FD_Call":[]

}

for p in range(0,length) :
    if S01[p] != '-' and K0[p] != '-' and r0[p] != '-' and T0[p] != '-' and sigma0[p] != '-':
        S = float(S01[p]);
        S1 = S;
        K = float(K0[p]);
        r = float(r0[p]);
        T = float(T0[p]);
        # print(T)
        sigma = float(sigma0[p]);
        steps = 265;
        N = 10000;
        LTP = LTP0[p];
        Symbol = Symbol0[p]
        Date = Date0[p]
        expiry = expiry0[p]
        open = open0[p]
        close = close0[p]

    

        d1 = ((np.log(S/K)) + ((r+(sigma**2)/2)*T)) / (sigma*(T**(0.5)))
        d2 = ((np.log(S/K)) + ((r-(sigma**2)/2)*T)) / (sigma*(T**(0.5)))

        C = float(norm.cdf(d1))*S - float(norm.cdf(d2))*K*((2.303)**(-r*T))
        # print(C)

        def geo_paths(S, T, r, q, sigma, steps, N):
            dt = T/steps
            #S_{T} = ln(S_{0})+\int_{0}^T(\mu-\frac{\sigma^2}{2})dt+\int_{0}^T \sigma dW(t)
            ST = np.log(S) +  np.cumsum(((r - q - sigma**2/2)*dt +\
                                    sigma*np.sqrt(dt) * \
                                    np.random.normal(size=(steps,N))),axis=0)
            return np.exp(ST)

        def black_scholes_call(S,K,T,r,q,sigma):
            d1 = (np.log(S/K) + (r - q + sigma**2/2)*T) / sigma*np.sqrt(T)
            d2 = d1 - sigma* np.sqrt(T)
            
            call = S * np.exp(-q*T)* norm.cdf(d1) - K * np.exp(-r*T)*norm.cdf(d2)
            return call
        

        paths= geo_paths(S,T,r, 0,sigma,steps,N)

        payoffs = np.maximum(paths[-1]-K, 0)
        option_price = np.mean(payoffs)*np.exp(-r*T)
        MC_Call = option_price

        def bottom_boundary_condition(K,T,S_min, r, t):
            return np.zeros(t.shape)
        def top_boundary_condition(K,T,S_max, r, t):
            return S_max-np.exp(-r*(T-t))*K
        def final_boundary_condition(K,T,S_min, r, t):
            return np.maximum(S-K,0)
        
        def compute_abc( K, T, sigma, r, S, dt, dS ):
            a = -sigma**2 * S**2/(2* dS**2 ) + r*S/(2*dS)
            b = r + sigma**2 * S**2/(dS**2)
            c = -sigma**2 * S**2/(2* dS**2 ) - r*S/(2*dS)
            return a,b,c

        def compute_lambda( a,b,c ):
            return scipy.sparse.diags( [a[1:],b,c[:-1]],offsets=[-1,0,1])

        def compute_W(a,b,c, V0, VM): 
            M = len(b)+1
            W = np.zeros(M-1)
            W[0] = a[0]*V0 
            W[-1] = c[-1]*VM 
            return W
        
        # # Choose the shape of the grid
        # M = 20
        # N = 20
        # dt = T/N
        # S_min=0
        # S_max=K*np.exp(8*sigma*np.sqrt(T))
        # dS = (S_max-S_min)/M
        # S = np.linspace(S_min,S_max,M+1)
        # t = np.linspace(0,T,N+1)
        # V = np.zeros((N+1,M+1)) #...
        
        # # Set the boundary conditions
        # V[:,-1] = top_boundary_condition(K,T,S_max,r,t)
        # V[:,0] = bottom_boundary_condition(K,T,S_max,r,t)
        # V[-1,:] = final_boundary_condition(K,T,S,r,t) #...
        
        # # Apply the recurrence relation
        # a,b,c = compute_abc(K,T,sigma,r,S[1:-1],dt,dS)
        # Lambda =compute_lambda( a,b,c) 
        # identity = scipy.sparse.identity(M-1)
    
        # for i in range(N,0,-1):
        #     W = compute_W(a,b,c,V[i,0],V[i,M])
        #     # Use `dot` to multiply a vector by a sparse matrix
        #     V[i-1,1:M] = (identity-Lambda*dt).dot( V[i,1:M] ) - W*dt
        S0 = S
        N = 100     # number of grid points
        dt = T/N    # time step
        dx = sigma*np.sqrt(dt)  # space step

        # Define the grid
        x = np.arange(N+1)*dx - N*dx/2
        S = S0*np.exp(x)

        # Define the payoff at maturity
        payoff = np.maximum(S-K, 0)

        # Initialize the option value at maturity
        V = payoff

        # Apply the finite difference method backward in time
        for i in range(N-1, -1, -1):
            V[1:-1] = 0.5*(V[:-2] + V[2:])/(1+r*dt)   # Crank-Nicolson scheme
            V[0] = 0    # boundary condition at S=0
            V[-1] = S[-1]-K*np.exp(-r*(T-i*dt))    # boundary condition at S=inf
            
        # Interpolate the option value at S=S0
        from scipy.interpolate import interp1d
        f = interp1d(S, V, kind='linear', fill_value='extrapolate')
        call_price = f(S0)

        # print("The price of the European call option is:", call_price)
    

        # print(V)
        print(p)

        csv["Strike_Price"].append(S1);
        csv["Spot_Price"].append(K);
        csv["Time"].append(T);
        csv["Volatility"].append(sigma);
        csv["Risk_Free_Intrest"].append(r);
        csv["LTP"].append(LTP);
        csv["BS_Call"].append(C);
        csv["MC_Call"].append(MC_Call);
        csv["FD_Call"].append(call_price);
        csv["close"].append(close);
        csv["Symbol"].append(Symbol);
        csv["date"].append(Date);
        csv["expiry"].append(expiry);
        csv["open"].append(open);

        # print(V[19][18]);
    
results = pd.DataFrame(csv)
results.to_csv('dataset/DataSet-l2.csv', index=False)

