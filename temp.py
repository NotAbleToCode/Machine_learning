import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

dataset=[0.697, 0.460, 1, 
    0.774, 0.376, 1, 
    0.634, 0.264, 1, 
    0.608, 0.318, 1, 
    0.556, 0.215, 1, 
    0.403, 0.237, 1, 
    0.481, 0.149, 1, 
    0.437, 0.211, 1, 
    0.666, 0.091, 0, 
    0.243, 0.267, 0, 
    0.245, 0.057, 0, 
    0.343, 0.099, 0, 
    0.639, 0.161, 0, 
    0.657, 0.198, 0, 
    0.360, 0.370, 0, 
    0.593, 0.042, 0, 
    0.719, 0.103, 0, ]
dataX=np.array(dataset).reshape((17,3))
dataY=np.array([i for i in dataX[:,2:3]]).reshape(17)
dataX=dataX[:,:2]
print(dataX[:8])
mean1,mean0 = np.mean(dataX[:8],axis=0),np.mean(dataX[8:],axis=0)
print('mean0, mean1:')
print(mean0,mean1)
mean1.reshape(1,2)
mean0.reshape(1,2)
# cov1=np.cov(np.array([dataX[:8,:1].reshape(8),dataX[:8,1:].reshape(8)]))
# cov0=np.cov(np.array([dataX[8:,:1].reshape(9),dataX[8:,1:].reshape(9)]))
# print(cov0+cov1)
# w = (np.mat(cov0+cov1)).I*(mean0-mean1).reshape(2,1)

def sigma(x,mean):
    total = np.zeros((2,2))
    for i in range(x.shape[0]):
        temp = (x[i]-mean).reshape(2,1)
        total += temp*temp.T
    return total

def getSw(dataX,mean0,mean1):
    Sw = sigma(dataX[:8],mean1)+sigma(dataX[8:],mean0)
    print('Sw')
    print(Sw)
    return np.mat(Sw)

def getw(mean0,mean1,Sw):
    return Sw.I*(mean0-mean1).reshape(2,1)

def getplotxy(w_):
    x = np.linspace(0,0.8,30)
    w_=w_.tolist()
    y = -w_[0][0]*x/w_[1][0]
    return x,y

if __name__=='__main__':
    Sw = getSw(dataX,mean0,mean1)
    w_ = getw(mean0,mean1,Sw)
    print('w:')
    print(w_)

    ##plot
    # plt.scatter(dataX[:8,:1].reshape(8),dataX[:8,1:].reshape(8),marker='+',color='black',label='好瓜')
    # plt.scatter(dataX[8:,:1].reshape(9),dataX[8:,1:].reshape(9),marker='x',color='red',label='坏瓜')
    # x,y = getplotxy(w_)
    # plt.plot(x,y,color='orange',label='投影线')
    # plt.legend(loc=2)
    # plt.xlabel('密度')
    # plt.ylabel('含糖率')
    # plt.show()
