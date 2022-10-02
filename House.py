import numpy as np
import pandas as pd
import matplotlib.pyplot as plt






path="HousesPrice.txt"

data=pd.read_csv(path,header=0,names=["Size","Rooms","price"])
data=(data-data.mean())/data.std()
data.insert(0,"X0",1)
def costj(X,y,Thta):
    h=X@Thta.T
    return np.sum(np.power(h-y,2))/(2*len(X))

def GradientDescent(inputs,outputs,Thta,iters,alpha):
    temp=np.zeros(Thta.shape)
    cost=np.zeros(iters)
    args=Thta.shape[0]
    for i in range(iters):
        error=inputs@Thta.T-outputs
        for j in range(args):
            z=error*inputs[:,j]
            temp[j]=Thta[j]-((alpha/len(inputs))*np.sum(z))
        cost[i]=costj(inputs,outputs,Thta)
        Thta=temp
    return cost,Thta


cols=data.shape[1]
inputs=data.iloc[:,:cols-1].values
outputs=data.iloc[:,cols-1].values
thta=np.zeros(inputs.shape[1])

alpha=0.1
iters=1000

cost,finalargs=GradientDescent(inputs,outputs,thta,iters,alpha)

print("cost every iter\n",cost)
print("final args\n",finalargs)

x=np.linspace(data["Size"].min(),data["Size"].max())
fx=finalargs[1]*x+finalargs[2]

print(data)


#graph 

fig,ax=plt.subplots()
ax.plot(x,fx,color="r")
ax.scatter(data["Size"],data["price"],label="training data",color="b")
plt.show()



