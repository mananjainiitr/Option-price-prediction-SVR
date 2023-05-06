from cmath import nan
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

data = pd.read_csv('dataset/DataSet-l3.csv')

K0 = data['Spot_Price']  # spot price at t=0
S0 = data['Strike_Price'] # strike price
r0 = data['Risk_Free_Intrest']/100 # risk-free interest rate
T0 = data['Time']# time to expiration in years
sigma0 = data['Volatility'] # volatility of the underlying asset
LTP0 = data["LTP"]
Symbol0 = data["Symbol"]
Date0 = data["date"]
expiry0 = data["expiry"]
open0 = data["open"]
close0 = data["close"]
BS_Call0 = data["BS_Call"]
MS_Call0 = data["MC_Call"]
FD_Call0 = data["FD_Call"]
cT10 = data["C(T+1)"]
bSCT0 = data["C_BS(T+1)"]
mCCT0 = data["C_MC(T+1)"]
fDCT0 = data["C_FD(T+1)"]


length = len(S0)

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
    "FD_Call":[],
    "C(T+1)":[],
    "C_BS(T+1)":[],
    "C_MC(T+1)":[],
    "C_FD(T+1)":[]
}
csv2 = {
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
    "FD_Call":[],
    "C(T+1)":[],
    "C_BS(T+1)":[],
    "C_MC(T+1)":[],
    "C_FD(T+1)":[]
}
csv3 = {
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
    "FD_Call":[],
    "C(T+1)":[],
    "C_BS(T+1)":[],
    "C_MC(T+1)":[],
    "C_FD(T+1)":[]
}
csv4 = {
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
    "FD_Call":[],
    "C(T+1)":[],
    "C_BS(T+1)":[],
    "C_MC(T+1)":[],
    "C_FD(T+1)":[]
}
csv5 = {
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
    "FD_Call":[],
    "C(T+1)":[],
    "C_BS(T+1)":[],
    "C_MC(T+1)":[],
    "C_FD(T+1)":[]
}
csv6 = {
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
    "FD_Call":[],
    "C(T+1)":[],
    "C_BS(T+1)":[],
    "C_MC(T+1)":[],
    "C_FD(T+1)":[]
}


# print(CT10)
for p in range(0,length) :
    if S0[p] != '-' and K0[p] != '-' and r0[p] != '-' and T0[p] != '-' and sigma0[p] != '-' and cT10[p] != -1 and bSCT0[p] != -1 and mCCT0[p] != -1 and fDCT0[p] != -1:
        S = float(S0[p]);
        K = float(K0[p]);
        r = float(r0[p]);
        T = float(T0[p]);
        # print(T)
        sigma = float(sigma0[p]);
        steps = 265;
        N = 10000;
        lTP = LTP0[p];
        Symbol = Symbol0[p]
        Date = Date0[p]
        expiry = expiry0[p]
        open = open0[p]
        close = close0[p]
        BS_Call = BS_Call0[p]
        MS_Call = MS_Call0[p]
        FD_Call = FD_Call0[p]
        cT1 = cT10[p]
        bSCT1 = bSCT0[p]
        mCCT1 = mCCT0[p]
        fDCT1 = fDCT0[p]

        if T*365 <= 31 and S/K < 0.97:
            csv["Strike_Price"].append(S);
            csv["Spot_Price"].append(K);
            csv["Time"].append(T);
            csv["Volatility"].append(sigma);
            csv["Risk_Free_Intrest"].append(r);
            csv["LTP"].append(lTP);
            csv["BS_Call"].append(BS_Call);
            csv["MC_Call"].append(MS_Call);
            csv["FD_Call"].append(FD_Call);
            csv["close"].append(close);
            csv["Symbol"].append(Symbol);
            csv["date"].append(Date);
            csv["expiry"].append(expiry);
            csv["open"].append(open);
            csv["C(T+1)"].append(cT1);
            csv["C_BS(T+1)"].append(bSCT1);
            csv["C_MC(T+1)"].append(mCCT1);
            csv["C_FD(T+1)"].append(fDCT1);
        
        elif T*365 <= 31 and S/K > 0.97 and S/K <1.05:

            csv2["Strike_Price"].append(S);
            csv2["Spot_Price"].append(K);
            csv2["Time"].append(T);
            csv2["Volatility"].append(sigma);
            csv2["Risk_Free_Intrest"].append(r);
            csv2["LTP"].append(lTP);
            csv2["BS_Call"].append(BS_Call);
            csv2["MC_Call"].append(MS_Call);
            csv2["FD_Call"].append(FD_Call);
            csv2["close"].append(close);
            csv2["Symbol"].append(Symbol);
            csv2["date"].append(Date);
            csv2["expiry"].append(expiry);
            csv2["open"].append(open);
            csv2["C(T+1)"].append(cT1);
            csv2["C_BS(T+1)"].append(bSCT1);
            csv2["C_MC(T+1)"].append(mCCT1);
            csv2["C_FD(T+1)"].append(fDCT1);
        
        elif T*365 <= 31 and S/K >1.05:

            csv3["Strike_Price"].append(S);
            csv3["Spot_Price"].append(K);
            csv3["Time"].append(T);
            csv3["Volatility"].append(sigma);
            csv3["Risk_Free_Intrest"].append(r);
            csv3["LTP"].append(lTP);
            csv3["BS_Call"].append(BS_Call);
            csv3["MC_Call"].append(MS_Call);
            csv3["FD_Call"].append(FD_Call);
            csv3["close"].append(close);
            csv3["Symbol"].append(Symbol);
            csv3["date"].append(Date);
            csv3["expiry"].append(expiry);
            csv3["open"].append(open);
            csv3["C(T+1)"].append(cT1);
            csv3["C_BS(T+1)"].append(bSCT1);
            csv3["C_MC(T+1)"].append(mCCT1);
            csv3["C_FD(T+1)"].append(fDCT1);

        elif T*365 > 31 and S/K < 0.97:

            csv4["Strike_Price"].append(S);
            csv4["Spot_Price"].append(K);
            csv4["Time"].append(T);
            csv4["Volatility"].append(sigma);
            csv4["Risk_Free_Intrest"].append(r);
            csv4["LTP"].append(lTP);
            csv4["BS_Call"].append(BS_Call);
            csv4["MC_Call"].append(MS_Call);
            csv4["FD_Call"].append(FD_Call);
            csv4["close"].append(close);
            csv4["Symbol"].append(Symbol);
            csv4["date"].append(Date);
            csv4["expiry"].append(expiry);
            csv4["open"].append(open);
            csv4["C(T+1)"].append(cT1);
            csv4["C_BS(T+1)"].append(bSCT1);
            csv4["C_MC(T+1)"].append(mCCT1);
            csv4["C_FD(T+1)"].append(fDCT1);
        
        elif T*365 > 31 and S/K > 0.97 and S/K <1.05:

            csv5["Strike_Price"].append(S);
            csv5["Spot_Price"].append(K);
            csv5["Time"].append(T);
            csv5["Volatility"].append(sigma);
            csv5["Risk_Free_Intrest"].append(r);
            csv5["LTP"].append(lTP);
            csv5["BS_Call"].append(BS_Call);
            csv5["MC_Call"].append(MS_Call);
            csv5["FD_Call"].append(FD_Call);
            csv5["close"].append(close);
            csv5["Symbol"].append(Symbol);
            csv5["date"].append(Date);
            csv5["expiry"].append(expiry);
            csv5["open"].append(open);
            csv5["C(T+1)"].append(cT1);
            csv5["C_BS(T+1)"].append(bSCT1);
            csv5["C_MC(T+1)"].append(mCCT1);
            csv5["C_FD(T+1)"].append(fDCT1);
        
        elif T*365 > 31 and S/K >1.05:

            csv6["Strike_Price"].append(S);
            csv6["Spot_Price"].append(K);
            csv6["Time"].append(T);
            csv6["Volatility"].append(sigma);
            csv6["Risk_Free_Intrest"].append(r);
            csv6["LTP"].append(lTP);
            csv6["BS_Call"].append(BS_Call);
            csv6["MC_Call"].append(MS_Call);
            csv6["FD_Call"].append(FD_Call);
            csv6["close"].append(close);
            csv6["Symbol"].append(Symbol);
            csv6["date"].append(Date);
            csv6["expiry"].append(expiry);
            csv6["open"].append(open);
            csv6["C(T+1)"].append(cT1);
            csv6["C_BS(T+1)"].append(bSCT1);
            csv6["C_MC(T+1)"].append(mCCT1);
            csv6["C_FD(T+1)"].append(fDCT1);


print("done")
results = pd.DataFrame(csv)
results.to_csv('D1.csv', index=False)

results = pd.DataFrame(csv2)
results.to_csv('D2.csv', index=False)

results = pd.DataFrame(csv3)
results.to_csv('D3.csv', index=False)

results = pd.DataFrame(csv4)
results.to_csv('D4.csv', index=False)

results = pd.DataFrame(csv5)
results.to_csv('D5.csv', index=False)

results = pd.DataFrame(csv6)
results.to_csv('D6.csv', index=False)
        