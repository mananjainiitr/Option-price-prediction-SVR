# This Code is intended to merge 10 data sheet of year 2013 & 2014 from month March to December.
# The name of the Sheets must be of form Sheet{Number} where number belong 1-10

import pandas as pd

data = [] 
Symbol = [] 
date = [] 
expiry = [] 
strike = [] 
open = [] 
Ltp =  [] 
spot = [] 
closed = []

for i in range(0,10):
    j = 1 + i
    data.append(pd.read_csv(f"dataset/Sheet{j}.csv"))

for i in range(0,10):

    Symbol.append(data[i]["Symbol  "])
    date.append(data[i]["Date  "])
    expiry.append(data[i]["Expiry  "])
    strike.append(data[i]["Strike Price  "])
    open.append(data[i]["Open  "])
    Ltp.append(data[i]["LTP  "])
    spot.append(data[i]["Underlying Value  "])
    closed.append(data[i]["Close  "])

csv = {
    "Symbol":[],
    "date":[],
    "expiry":[],
    "open":[],
    "Strike_Price":[],
    "Spot_Price":[],
    "LTP":[],
    "Close":[],
    "rate":[]
}

length = len(strike) 
print (length)

for i in range(0,length):
    l = len(strike[i])
    for j in range(0,l):
        csv["Symbol"].append(Symbol[i][j]);
        csv["date"].append(date[i][j]);
        csv["expiry"].append(expiry[i][j]);
        csv["open"].append(open[i][j]);
        csv["Strike_Price"].append(strike[i][j]);
        csv["Spot_Price"].append(spot[i][j]);
        csv["LTP"].append(Ltp[i][j])
        csv["Close"].append(closed[i][j]);
        r = 8.5
        if i == 0 and j <= (l/2) :
            r = 9.4
        elif i == 0 and j > (l/2) :
            r = 8.5
        elif i == 1 and j <= (l/2) :
            r = 8.2
        elif i == 1 and j > (l/2) :
            r = 8.2
        elif i == 2 and j <= (l/2) :
            r = 8.7
        elif i == 2 and j > (l/2) :
            r = 10.5
        elif i == 3 and j <= (l/2) :
            r = 10.5
        elif i == 3 and j > (l/2) :
            r = 9.5
        elif i == 4 and j <= (l/2) :
            r = 9
        elif i == 4 and j > (l/2) :
            r = 8.6
        elif i == 5 and j <= (l/2) :
            r = 9.7
        elif i == 5 and j > (l/2) :
            r = 9.2
        elif i == 6 and j <= (l/2) :
            r = 8.8
        elif i == 6 and j > (l/2) :
            r = 8.6
        elif i == 7 and j <= (l/2) :
            r = 8.7
        elif i == 7 and j > (l/2) :
            r = 8.7
        elif i == 8 and j <= (l/2) :
            r = 8.7
        elif i == 8 and j > (l/2) :
            r = 8.6
        elif i == 9 and j <= (l/2) :
            r = 8.5
        elif i == 9 and j > (l/2) :
            r = 8.5
        
        csv["rate"].append(r);
        

results = pd.DataFrame(csv)
results.to_csv('dataset/DataSet-L0.csv', index=False)
