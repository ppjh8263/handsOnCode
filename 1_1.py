import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.linear_model

oecdBLI = pd.read_csv('rawData/BLI_10092020072350637.csv')
gdp = pd.read_csv('rawData/WEO_Data.csv',thousands=',')

oecdBLI=oecdBLI[oecdBLI["INEQUALITY"]=="TOT"].pivot(index="Country", columns="Indicator", values="Value")
gdp.rename(columns={"2015": "GDP"}, inplace=True)
gdp.set_index("Country", inplace=True)

bliGdp = pd.merge(oecdBLI, gdp, on="Country")
bliGdp.sort_values(by="GDP", inplace=True)

bliGdp=bliGdp[["GDP", 'Life satisfaction']]

X=np.c_[bliGdp["GDP"]]
Y=np.c_[bliGdp["Life satisfaction"]]

bliGdp.plot(kind='scatter', x="GDP", y='Life satisfaction')
plt.show()

model = sklearn.linear_model.LinearRegression()

model.fit(X, Y)

X_new = [[22587]]  # 키프로스 1인당 GDP
print(model.predict(X_new))
