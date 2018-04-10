#!/usr/bin/python

import numpy as np

import matplotlib
#matplotlib.use('Agg',warn=False)
import matplotlib.pyplot as plt
import scipy.io

def angle(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'    """
    c = np.dot(v1, v2)
    s = np.linalg.norm(np.cross(v1, v2))
    return np.arctan2(s, c)

def angle2(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2' with sign   """
    return np.arccos(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))) * np.sign(np.cross(v1,v2))


v1 = np.array([ 1, 1])
v2 = np.array([ 1, -1])


fig = plt.figure()

mat = scipy.io.loadmat('racetrack.mat')
t_l = mat['t_l']
t_r = mat['t_r']
t_m = (t_l+t_r)/2

n = 1800
print(t_l.shape)
p = np.array([10*np.random.random()-8, 10*np.random.random()+250])

idx = 0
d_mp = 1e10
while np.linalg.norm(p-t_m[idx+1,:]) <= d_mp:
    mp = p - t_m[idx+1,:]
    d_mp = np.linalg.norm(mp)
    idx += 1

lr = t_r[idx,:] - t_l[idx,:]
lr_n = np.array([-lr[1],lr[0]])
d_lr = np.dot(mp.T,lr) / np.dot(lr,lr) * lr # rename this
d = np.linalg.norm(d_lr)
d_sign = np.sign(np.cross(mp,lr_n))

print(d*d_sign)
print( d <= 5/2. )

plt.plot(t_l[:n,0],t_l[:n,1], 'b-')
plt.plot(t_r[:n,0],t_r[:n,1], 'b-')
plt.plot(t_m[:n,0],t_m[:n,1], 'y-')

plt.plot(p[0], p[1], 'rx')
plt.plot((t_m[idx,0],p[0]),(t_m[idx,1],p[1]), 'r-')
plt.plot((t_m[idx,0],t_m[idx,0]+d_lr[0]),(t_m[idx,1],t_m[idx,1]+d_lr[1]), 'g-')
plt.plot((t_m[idx,0],t_m[idx,0]+lr_n[0]),(t_m[idx,1],t_m[idx,1]+lr_n[1]), 'c-')



# horizon old
idx = 200
l = 10 # numbe of points
k = 10 # distance in steps
x_red = t_m[idx:idx+(l+1)*k:k,:]
a_red = []
for i in range(l):
	a_red.append(angle2(x_red[i],x_red[i+1]))
a_red = np.array(a_red)
a_red *= 100
print(x_red)
print(a_red)


# horizon new
idx = 200
n = 10 # number of points
h = 10 # distance between points in m

x_last = t_m[idx]
x_red = [x_last]
i = j = 0
while True:
    while True:
        i += 1
        x_tmp = t_m[idx+i]
        if np.linalg.norm(x_tmp-x_last) > h:
            x_red.append(x_tmp)
            x_last = x_tmp
            break
    j += 1
    if j == n:
        break
x_red = np.array(x_red)
a_red = []
for i in range(n):
	a_red.append(angle2(x_red[i],x_red[i+1]))
a_red = np.array(a_red) # angel between horizon points
a_red *= 100/3.
print(x_red)
print(a_red)



plt.plot(x_red[:,0], x_red[:,1], 'k.')

ax = plt.axes()
ax.set_aspect('equal', 'datalim')
#ax.set_xlim(-20,10); ax.set_ylim(230,260)
ax.set_xlim(-40,30); ax.set_ylim(190,290)
plt.show()
