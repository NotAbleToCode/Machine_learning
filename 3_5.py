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

pos = [] # 正类
neg = [] # 负类

for i in datas:
    if i[3] == 0:
        neg.append(i)
    else:
        pos.append(i)

pos = np.array(pos)[:,0:3]
neg = np.array(neg)[:,0:3]

# print(pos)
# print(neg)
# 样本的维度为2
dimen = 2
sigma_1 = np.zeros([dimen,dimen])
sigma_2 = np.zeros([dimen,dimen])
sum_1 = np.zeros([1,dimen])
sum_2 = np.zeros([1,dimen])
# 先求均值
for i in range(pos.shape[0]):
    sum_1 += pos[i:i+1,:dimen]
for i in range(neg.shape[0]):
    sum_2 += neg[i:i+1,:dimen]
u_1 = sum_1/pos.shape[0]
u_2 = sum_2/neg.shape[0]

# 先求类内散度矩阵
for i in range(pos.shape[0]):
    sigma_1 += (pos[i:i+1,:dimen]-u_1).T @ (pos[i:i+1,:dimen]-u_1)
    

for i in range(neg.shape[0]):
    sigma_2 += (neg[i:i+1,:dimen]-u_2).T @ (neg[i:i+1,:dimen]-u_2)
    

# print(sigma_1)
# print(sigma_2)

S_w = sigma_1+sigma_2
# print(S_w)
S_w_inv = np.linalg.inv(S_w)

w = S_w_inv @ (u_1-u_2).T

print(w)

