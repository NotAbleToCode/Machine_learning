import numpy as np
datas = [[0.697, 0.460, 1, 1],
[0.774, 0.376, 1, 1],
[0.634, 0.264, 1, 1],
[0.608, 0.318, 1, 1],
[0.556, 0.215, 1, 1],
[0.403, 0.237, 1, 1],
[0.481, 0.149, 1, 1],
[0.437, 0.211, 1, 1],
[0.666, 0.091, 1, 0],
[0.243, 0.267, 1, 0],
[0.245, 0.057, 1, 0],
[0.343, 0.099, 1, 0],
[0.639, 0.161, 1, 0],
[0.657, 0.198, 1, 0],
[0.360, 0.370, 1, 0],
[0.593, 0.042, 1, 0],
[0.719, 0.103, 1, 0]]
datas = np.array(datas)

lr = 0.01

def loss(data, beta):
    temp1 = -data[:,3:4]*(data[:,0:3]@beta)
    temp2 = np.log(1+np.e**(data[:,0:3]@beta))
    return np.sum(temp1+temp2)

def gradient_decrease(data, beta, lr):
    while 1:
        temp = loss(data, beta)
        grad_1 = (loss(data, beta+[[0.001],[0],[0]])-temp)/0.001
        grad_2 = (loss(data, beta+[[0],[0.001],[0]])-temp)/0.001
        grad_3 = (loss(data, beta+[[0],[0],[0.001]])-temp)/0.001
        beta = beta - np.array([[grad_1],[grad_2],[grad_3]])*lr
        if max([grad_1,grad_2,grad_3])<0.00001:
            break
    return beta

beta = np.random.random([3,1])
print(beta)
beta = gradient_decrease(datas, beta, lr)
print(beta)

