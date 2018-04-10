#!/usr/bin/python

from numpy import *
import numpy as np
import scipy.io
import sys
import matplotlib
import matplotlib.pyplot as plt

from genotype import *
from phenotype import *
from neat import *
from utils import *

matplotlib.use('Agg',warn=False)
np.set_printoptions(precision=6, linewidth=200)


# Dynamics model of the car, code was ported form matlab...
def singletrack(t,X,U):
    
    ###########################################################################
    ## function [X_dot] = singletrack(t,X)
    #
    # vector field for the single-track model
    #
    # inputs: (t (time),) x (x position), y (y position), v (velocity), beta
    # (side slip angle), psi (yaw angle), omega (yaw rate), x_dot (longitudinal
    # velocity), y_dot (lateral velocity), psi_dot (yaw rate (redundant)),
    # varphi_dot (wheel rotary frequency)
    #
    # outputs: x_dot (longitudinal velocity), y_dot (lateral velocity), v_dot
    # (acceleration), beta_dot (side slip rate), psi_dot (yaw rate), omega_dot
    # (yaw angular acceleration), x_dot_dot (longitudinal acceleration),
    # y_dot_dot (lateral acceleration), psi_dot_dot (yaw angular acceleration
    # (redundant)), varphi_dot_dot (wheel rotary acceleration)
    #
    # files requested: controller.m
    #
    # The model is adopted from the scripts of M. Gerdts and D. Schramm,
    # respectively.
    #
    # This file is for use within the "Project Competition" of the "Concepts of
    # Automatic Control" course at the University of Stuttgart, held by F.
    # Allgoewer.
    #
    # written by J. M. Montenbruck, Dec. 2013
    # mailto:jan-maximilian.montenbruck@ist.uni-stuttgart.de

    ############################# INITIALIZATION ##############################

    ## vehicle parameters
    m = 1239 # vehicle mass
    g = 9.81 # gravitation
    l_f = 1.19016 # distance of the front wheel to the center of mass
    l_r = 1.37484 # distance of the rear wheel to the center of mass
    #l = l_f+l_r # vehicle length (obsolete)
    R = 0.302 # wheel radius
    I_z = 1752 # vehicle moment of inertia (yaw axis)
    I_R = 1.5 # wheel moment of inertia
    i_g = [3.91,2.002,1.33,1,0.805] # transmissions of the 1st ... 5th gear
    i_0 = 3.91 # motor transmission
    B_f = 10.96 # stiffnes factor (Pacejka) (front wheel)
    C_f = 1.3 # shape factor (Pacejka) (front wheel)
    D_f = 4560.4 # peak value (Pacejka) (front wheel)
    E_f = -0.5 # curvature factor (Pacejka) (front wheel)
    B_r = 12.67 #stiffnes factor (Pacejka) (rear wheel)
    C_r = 1.3 #shape factor (Pacejka) (rear wheel)
    D_r = 3947.81 #peak value (Pacejka) (rear wheel)
    E_r = -0.5 # curvature factor (Pacejka) (rear wheel)
    f_r_0 = 0.009 # coefficient (friction)
    f_r_1 = 0.002 # coefficient (friction)
    f_r_4 = 0.0003 # coefficient (friction)

    ## control inputs
    #U = controller(X) #control input vector
    delta = U[0] # steering angle
    G = int(U[1]) # gear 1 ... 5
    F_b = U[2] #braking force
    zeta = U[3] # braking force distribution
    phi = U[4] # gas pedal position
    # input constraints
    if delta>0.53: # upper bound for steering angle exceeded?
        delta = 0.53 # upper bound for steering angle
    if delta<-0.53: # lower bound for steering angle exceeded?
        delta = -0.53 # lower bound for steering angle
    if F_b<0: # lower bound for braking force exceeded?
        F_b = 0 # lower bound for braking force
    if F_b>15000: # upper bound for braking force exceeded?
        F_b = 15000 # upper bound for braking force
    if zeta<0: # lower bound for braking force distribution exceeded?
        zeta = 0 # lower bound for braking force distribution
    if zeta>1: # upper bound for braking force distribution exceeded?
        zeta = 1 # upper bound for braking force distribution
    if phi<0: # lower bound for gas pedal position exceeded?
        phi = 0 # lower bound for gas pedal position
    if phi>1: # upper bound for gas pedal position exceeded?
        phi = 1 # upper bound for gas pedal position

    ## state vector
    #x = X[0] # x position (obsolete)
    #y = X[1] # y position (obsolete)
    v = X[2] # velocity
    beta = X[3] # side slip angle
    psi = X[4] # yaw angle
    omega = X[5] # yaw rate
    #x_dot = X[6] # longitudinal velocity (obsolete)
    #y_dot = X[7] # lateral velocity (obsolete)
    psi_dot = X[8] # yaw rate (redundant)
    varphi_dot = X[9] # wheel rotary frequency

    ################################# DYNAMICS ################################

    ## slip
    #slip angles and steering
    a_f = delta-arctan((l_f*psi_dot-v*sin(beta))/(v*cos(beta))) # front slip angle
    a_r = arctan((l_r*psi_dot+v*sin(beta))/(v*cos(beta))) #rear slip angle
    if a_f > a_r: #understeering?
        steering = 'understeering'
    if a_f < a_r: #oversteering?
        steering = 'oversteering'
    if a_f == a_r: #neutral steering?
        steering = 'neutral'
    if isnan(a_f): # front slip angle well-defined?
        a_f = 0 # recover front slip angle
    if isnan(a_r): # rear slip angle well-defined
        a_r = 0 # recover rear slip angle
    #wheel slip
    if v <= R*varphi_dot: # traction slip? (else: braking slip)
        S = 1-(v/(R*varphi_dot)) #traction wheel slip
    else:
        S = 1-((R*varphi_dot)/v) # braking slip
    if isnan(S): # wheel slip well-defined?
        S = 0 # recover wheel slip
    S = 0 # neglect wheel slip

    #print(array([v,beta,psi,omega,phi]))

    ## traction, friction, braking
    n = v*i_g[G-1]*i_0*(1/(1-S))/R # motor rotary frequency
    if isnan(n): # rotary frequency well defined?
        n = 0 #recover rotary frequency
    if n > (4800*pi)/30: # maximal rotary frequency exceeded?
        n = (4800*pi)/30 # recover maximal rotary frequency
    T_M = 200*phi*(15-14*phi)-200*phi*(15-14*phi)*(((n*(30/pi))**(5*phi))/(4800**(5*phi))) # motor torque
    if isnan(T_M): # hack
        T_M = 0.
    #print('T_M %s n %s phi %s' % (T_M,n,phi))
    M_wheel = i_g[G-1]*i_0*T_M # wheel torque
    F_w_r = (m*l_f*g)/(l_f+l_r) # weight rear
    F_w_f = (m*l_r*g)/(l_f+l_r) # weight front
    f_r = f_r_0+f_r_1*(abs(v)*3.6)/100+f_r_4*((abs(v)*3.6)/100)**4 # approximate friction
    F_b_r = zeta*F_b # braking force rear
    F_b_f = F_b*(1-zeta) # braking force front
    F_f_r = f_r*F_w_r # friction rear
    F_f_f = f_r*F_w_f # friction front
    F_x_r = (M_wheel/R)-sign(v*cos(beta))*F_b_r-sign(v*cos(beta))*F_f_r # longitudinal force rear wheel
    F_x_f = -sign(v*cos(beta))*F_b_f-sign(v*cos(beta))*F_f_f # longitudinal force front wheel
    F_y_r = D_r*sin(C_r*arctan(B_r*a_r-E_r*(B_r*a_r-arctan(B_r*a_r)))) # rear lateral force
    F_y_f = D_f*sin(C_f*arctan(B_f*a_f-E_f*(B_f*a_f-arctan(B_f*a_f)))) # front lateral force

    ################################## OUTPUT #################################

    ## vector field (right-hand side of differential equation)
    x_dot = v*cos(psi-beta) # longitudinal velocity
    y_dot = v*sin(psi-beta) # lateral velocity
    v_dot = (F_x_r*cos(beta)+F_x_f*cos(delta+beta)-F_y_r*sin(beta)-F_y_f*sin(delta+beta))/m # acceleration
    beta_dot = omega-(F_x_r*sin(beta)+F_x_f*sin(delta+beta)+F_y_r*cos(beta)+F_y_f*cos(delta+beta))/(m*v) # side slip rate
    psi_dot = omega # yaw rate
    omega_dot = (F_y_f*l_f*cos(delta)-F_y_r*l_r+F_x_f*l_f*sin(delta))/I_z # yaw angular acceleration
    x_dot_dot = (F_x_r*cos(psi)+F_x_f*cos(delta+psi)-F_y_f*sin(delta+psi)-F_y_r*sin(psi))/m # longitudinal acceleration
    y_dot_dot = (F_x_r*sin(psi)+F_x_f*sin(delta+psi)+F_y_f*cos(delta+psi)+F_y_r*cos(psi))/m # lateral acceleration
    psi_dot_dot = (F_y_f*l_f*cos(delta)-F_y_r*l_r+F_x_f*l_f*sin(delta))/I_z # yaw angular acceleration
    varphi_dot_dot = (F_x_r*R)/I_R # wheel rotary acceleration
    if isnan(beta_dot): # side slip angle well defined?
        beta_dot = 0 # recover side slip angle
    ## write outputs
    X_dot = array([x_dot,y_dot,v_dot,beta_dot,psi_dot,omega_dot,x_dot_dot,y_dot_dot,psi_dot_dot,varphi_dot_dot]) # left-hand side
    return X_dot


class RacetrackTaskNew():
    def __init__(self):
        mat = scipy.io.loadmat('racetrack.mat')
        self.t_l = mat['t_l']
        self.t_r = mat['t_r']
        self.t_m = (self.t_l+self.t_r)/2
        self.t_m2 = vstack([self.t_m,self.t_m])

        self.X_0 = np.array([-2.5,0,0,0,pi/2,0,0,0,0,0])
        self.X_0 = np.array([-2.5,0,0,0,pi/8*5,0,0,0,0,0])

        self.horizon = 4 #4
        self.horizon_space = 10

        self.n = len(self.X_0)
        self.n_inputs = self.n + 1 + self.horizon
        self.n_inputs = 1 + 1 + self.horizon
        #self.n_inputs = 3 + 1 + self.horizon
        self.n_outputs = 4

        self.dt = 0.01

        self.max_steps = 1000/self.dt-1 # 16 min

        print('max_idx ' + str(shape(self.t_m)[0]-1))


    def _setp(self, network, X_ext):

        X = X_ext[:self.n]

        x = X[0] # x position
        y = X[1] # y position
        v = X[2] # velocity (strictly positive)
        beta = X[3] # side slip angle
        psi = X[4] # yaw angle
        omega = X[5] # yaw rate
        x_dot = X[6] # longitudinal velocity
        y_dot = X[7] # lateral velocity
        psi_dot = X[8] # yaw rate (redundant)
        varphi_dot = X[9] # wheel rotary frequency (strictly positive)

        net_input = np.hstack([v, X_ext[10:]])

        net_output = network.feed(net_input, RunType.ACTIVE)
        # TODO: gets pickled for multiprocessing?!

        if hasattr(self,'net_statistic'):
            self.net_statistic.update(net_input, net_output)

        #print('net_input  ' + str(net_input))
        #print('net_output ' + str(net_output))

        delta = net_output[0]*0.53 # steering angle
        #G = int((net_output[1])*4.+1) #gear 1 ... 5
        #Fb = (net_output[2]+1)*15000/2. # braking force
        zeta = (net_output[3]+1)/2. #braking force distribution
        #phi = (net_output[4]+1)/2. # gas pedal position

        Fb = min(net_output[2],0)*15000 # braking force
        phi = max(0,net_output[2]) # gas pedal position

        # optimal gear
        R = 0.302 # wheel radius
        S = 0 # neglect wheel slip
        i_g = [3.91,2.002,1.33,1,0.805] # transmissions of the 1st ... 5th gear
        i_0 = 3.91 # motor transmission
        G = 1
        M_wheel_max = 0.
        for k in range(len(i_g)):
            n = v*i_g[k]*i_0*(1/(1-S))/R # motor rotary frequency
            T_M = 200*phi*(15-14*phi)-200*phi*(15-14*phi)*(((n*(30/pi))**(5*phi))/(4800**(5*phi))) # motor torque
            if isnan(T_M): # hack
                T_M = 0.
            M_wheel = i_g[k]*i_0*T_M # wheel torque
            if M_wheel > M_wheel_max:
                M_wheel_max = M_wheel
                G = k+1

        U = array([delta,G,Fb,zeta,phi])

        #U = array([0.,1.0,0.,0.5,0.2])

        #print("X " + str(X))
        #print('U ' + str(U))

        dX = singletrack(0.,X,U)

        X = X + dX * self.dt
        return (U, X)


    def _loop(self, network, max_steps):

        nperr = np.geterr()
        np.seterr(all='ignore')

        X = self.X_0

        t_l = self.t_l
        t_r = self.t_r
        t_m = self.t_m
        t_m2 = self.t_m2

        max_idx = shape(self.t_m)[0]-1

        idx = 0

        steps = 0
        states = []
        actions = []
        idx_all = []
        d_all = []

        idx_v = np.zeros(max_idx+1)
        idx_d = np.zeros(max_idx+1)

        while True:
            steps += 1

            # find index with minimal distance to side
            p = X[0:2]
            mp = p - t_m[idx,:]
            d_mp = np.linalg.norm(mp)

            while True:
                if idx == max_idx:
                    break
                mp_new = p - t_m[idx+1,:]
                d_mp_new = np.linalg.norm(mp_new)
                #print(' %s %s ' %(d_mp, d_mp_new))
                if d_mp_new > d_mp:
                    break
                else:
                    mp = mp_new
                    d_mp = d_mp_new
                    idx += 1
            #print('p %s idx %d d_mp %f' %(str(p),idx,d_mp))
            #idx = 10
            #mp = p-t_m[idx+1,:]

            # minimal distance to side
            lr = t_r[idx,:] - t_l[idx,:]
            lr_n = np.array([-lr[1],lr[0]])
            d_lr = np.dot(mp.T,lr) / np.dot(lr,lr) * lr # TODO: rename this
            d = np.linalg.norm(d_lr)
            d_sign = np.sign(np.cross(mp,lr_n))

            # check if we are on track
            #print('lr %s d_lr %s' % (lr,np.linalg.norm(d_lr)))
            if d > 5./2:
                #print('CRASH')
                break

            if steps > 200 and X[2] < 0.01:
                #print('TO SLOW')
                break


            # horizon
            #l = self.horizon # numbe of points
            #k = self.horizon_space # distance between points in steps
            #x_red = self.t_m2[idx:idx+(l+1)*k:k,:]
            #a_red = []
            #for i in range(l):
            #	a_red.append(angle2(x_red[i],x_red[i+1]))
            #a_red = np.array(a_red) # angel between horizon points
            ##print(a_red)
            #
            #X_ext = np.hstack([X, d*d_sign, a_red*0.5])

            # horizon
            n = self.horizon # number of points
            h = self.horizon_space # distance between points in m

            x_last = t_m2[idx]
            x_red = [x_last]
            i = j = 0
            while True:
                while True:
                    i += 1
                    x_tmp = t_m2[idx+i]
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

            X_ext = np.hstack([X, d*d_sign, a_red/3.])


            #print("a_red" + str(a_red*0.5))
            #print("X_ext" + str(X_ext))

            U, X = self._setp(network, X_ext)

            states.append(X)
            actions.append(U)

            idx_all.append(idx)
            d_all.append(d)
            idx_v[idx] = X[2]
            idx_d[idx] = max(idx_d[idx], d)

            if idx == max_idx:
                print('SOLVED')

            if idx == max_idx or steps == max_steps:
                break

        #print('steps ' + str(steps))

        np.seterr(**nperr)

        return steps, np.array(states), np.array(actions), np.array(idx_all), np.array(d_all), idx_v, idx_d


    def evaluate(self, network):

        #self.net_statistic = NetStatistic(self.n_inputs, self.n_outputs)

        network.data = steps, states, actions, idx_all, d_all, idx_v, idx_d = self._loop(network, self.max_steps)

        max_idx = shape(self.t_m)[0]-1
        idx = idx_all[-1]

        solved = int(idx == max_idx)

        #avg_v = sum(clip(states[:,2],0.0,inf))/steps
        #avg_d = sum(d_all**2)/steps
        #score_idx = 1.0 # idx           # [0,1]
        #score_v = 1.0 - 1./(1.+avg_v)   # [0,1]
        #score_d = 1./(1.+avg_d)         # [0,1]
        #score = score_d * idx # RacetrackTask_3

        if solved:
            w = np.ones(idx)
        else:
            w = 1-sigmoid(np.arange(idx)-(idx-150), 20)
        score_v = sum(w*idx_v[:idx]) / (1+sum(w))
        score_d = 1. / (1.+0.1*sum(idx_d[:idx]**2))
        score = (0.4*score_v + 0.8*score_d) * idx

        print('steps %5.d  idx %5.d  scroe_v %.5f  scroe_d %.5f  score %.5f' % (steps, idx, score_v, score_d, score))

        #if solved:
        #print(self.net_statistic)

        return [score, solved]


    def visualize(self, network, filename):
        #print('VISUALIZE SIMULATION')

        self.dump(network, filename)

        self._plot_racetrack(network.data, filename, network.genotype)


    def dump(self, network, filename):
        if not hasattr(network,'data'):
            self.evaluate(network)

        dumpf(filename+'.json', network.data)


    def _plot_racetrack(self, data, filename, genotype=None):

        #steps, states, actions, idx_all, d_all, idx_v, idx_d = self._loop(network, self.max_steps)
        steps, states, actions, idx_all, d_all, idx_v, idx_d = data

        idx = idx_all[-1]
        X = np.array(states)
        U = np.array(actions)

        fig = plt.figure(figsize=(8,16))
        options = {
            "markersize": 0.1,
            "linewidth": 0.1,
            "antialiased": False
        }
        plt.plot(self.t_l[:idx+1,0],self.t_l[:idx+1,1], 'b-o', **options)
        plt.plot(self.t_r[:idx+1,0],self.t_r[:idx+1,1], 'b-o', **options)
        plt.plot(self.t_m[:idx+1,0],self.t_m[:idx+1,1], 'y-', **options)
        plt.plot(X[:,0],X[:,1], 'r-', **options)

        ax = plt.axes()
        ax.set_aspect('equal', 'datalim')
        #ax.set_xlim([-7.5,2.5])
        #ax.set_xlim([-100,100])
        ax.set_ylim([-50,450])
        if genotype != None:
            g = genotype
            fig.text(0.02,0.02,'genome_id %04d  steps %d  fitness %0.4f  solved %d  idx %d' % (g.id, steps, g.fitness, g.solved, idx))
        plt.savefig(filename+'.png', dpi=300, bbox_inches='tight')
        plt.close()


    def _plot_state(self, X, X_label, filename):
        n_X = np.shape(X)[1]
        fig, axx = plt.subplots(n_X, figsize=(10,16), sharex=True, sharey=False)
        for i in range(n_X):
            ax = fig.axes[i]
            ax.plot(X[:,i])
            ax.set_ylabel(X_label[i])
        fig.subplots_adjust(hspace=0.2)
        plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)
        plt.savefig(filename+'.png', bbox_inches='tight')
        plt.close()


if __name__ == '__main__':

    task = RacetrackTaskNew()

    if len(sys.argv) > 1:
        if len(sys.argv) > 2:
            filename = sys.argv[2]
        else:
            filename = './results/RacetrackTask_3/best-net-397.json'
            filename = './results/RacetrackTask_10/leader-099-051-04671-net.json'
            filename = './results/RacetrackTaskNew_1/leader-000-017-00855-net.json'

        if sys.argv[1] == 'simulate':
            print('here we are')
            network = Network(None,filename=filename)
            task.dump(network, 'racetrack_sim')
        if sys.argv[1] == 'visualize':
            network = Network(None,filename=filename)
            network.genotype = Object()
            network.genotype.id = 0
            fitness, solved = task.evaluate(network)
            network.genotype.fitness = fitness
            network.genotype.solved = solved
            network.visualize('racetrack_net')
            task.visualize(network, 'racetrack_sim')
        if sys.argv[1] == 'plot':

            filename = './results/RacetrackTaskNew_1/leader-000-017-00855-sim.json'
            
            data = loadf(filename)
            steps, states, actions, idx_all, d_all, idx_v, idx_d = data
            X = np.array(states)
            U = np.array(actions)

            X_label = ['x', 'y', 'v', 'beta', 'psi', 'omega', 'x_dot', 'y_dot', 'psi_dot', 'varphi_dot']
            U_label = ['delta', 'G', 'Fb', 'zeta', 'phi']

            task._plot_racetrack(data, 'racetrack_plot')
            task._plot_state(X, X_label, 'racetrack_plot_X')
            task._plot_state(U, U_label, 'racetrack_plot_U')

        else:
            pass
    else:
        ga = GeneticAlgorithm(task)
        #ga.population_size = 120
        ga.population_size = 120
        ga.target_species = 12
        ga.compatibility_threshold = 1.0
        ga.number_generation_allowed_to_not_improve = 50
        ga.visualization_type = VisualizationType.ALL

        task.net_statistic = NetStatistic(task.n_inputs, task.n_outputs)

        for i in range(500):
            ga.epoch()
            print(task.net_statistic)

    sys.exit()
