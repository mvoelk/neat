#!/usr/bin/python

import pprint
pp = pprint.PrettyPrinter(indent=4, width=120, depth=3)

import sys
import numpy as np
import pygame as pg

from genotype import *
from phenotype import *
from neat import *


# from numeric_3d
def in2pi(a):
    """ Brings an angle in the range between -pi and +pi """
    if a > np.pi:
        a = a - int((a+np.pi)/(2*np.pi))*2*np.pi
    if a < -np.pi:
        a = a - int((a-np.pi)/(2*np.pi))*2*np.pi
    return a

in2piV = np.vectorize(in2pi)


class PoleBalanceTask(object):
    def __init__(self):
        self.g  = 9.81  # gravity
        self.mc = 1.0   # cart_mass
        self.mp = np.array([0.1]) # pole_mass
        self.l  = np.array([0.5]) # pole_length
        self.h  = 2.4   # track_limit
        self.r  = 0.628 # failure_angle = 2*pi/10
        self.f  = 10.0  # force_magnitude
        self.dt = 0.01  # timestep
        self.velocities           = True # False
        self.penalize_oscillation = True
        self.max_steps            = 1000

        x, dx, theta, dtheta = 0.0, 0.0, np.random.normal(0, 0.02, self.l.size), np.zeros(self.l.size)
        self.initial_state = (x, dx, theta, dtheta)

    @property
    def n_inputs(self):
        return (1+self.l.shape[0])*(1+int(self.velocities))
    @property
    def n_outputs(self):
        return 1


    def _simulation_step_multipole(self, action, state):
        # state is a tuple of (x, dx, (p1, p2), (dp1, dp2))
        x, dx, theta, dtheta = state

        #f = (min(1.0, max(-1.0, action)) - 0.5) * self.f * 2.0;

        # Alternate equations
        # fi = self.mp * self.l * dtheta**2 * np.sin(theta) + (3.0/4) * self.mp * np.cos(theta) * self.g * np.sin(theta)
        # mi = self.mp * (1 - (3.0/4) * np.cos(theta)**2)
        # ddx = f + np.sum(fi) / (self.mc + np.sum(mi))
        # ddtheta = (- 3.0 / (4 * self.l)) * (ddx * np.cos(theta) + self.g * np.sin(theta))

        # Equations from "THE POLE BALANCING PROBLEM"
        # _ni = (-f - self.mp * self.l * dtheta**2 * np.sin(theta))
        # m = self.mc + np.sum(self.mp)
        # _n = self.g * np.sin(theta) + np.cos(theta) * (_ni / m)
        # _d = self.l * (4./3. - (self.mp * np.cos(theta)**2) / m)
        # ddtheta = (_n / _d)
        # ddx = (f + np.sum(self.mp * self.l * np.floor(dtheta**2 * np.sin(theta) - ddtheta * np.cos(theta)))) / m

        x += self.dt * dx
        dx += self.dt * ddx
        theta += self.dt * dtheta
        dtheta += self.dt * ddtheta
        return (x, dx, theta, dtheta)


    def _simulation_step(self, action, state):
        # single pole, no friction, point mass at the end
        # -f + (mc+mp)*ddx - mp*l*sin(theta)*dtheta**2 + mp*l*cos(theta)*ddtheta
        # -mp*l**2*ddtheta + mp*g*l*sin(theta) - mp*l*cos(theta)*ddx
        # leads to
        # ddx = (dtheta**2*l*mp*sin(theta) + f - g*mp*sin(2*theta)/2)/(mc + mp*sin(theta)**2)
        # ddtheta = (g*(mc + mp)*sin(theta) - (dtheta**2*l*mp*sin(theta) + f)*cos(theta))/(l*(mc + mp*sin(theta)**2))

        x, dx, theta, dtheta = state
        f = action
        mc = self.mc
        mp = self.mp
        l = self.l
        g = self.g

        s = np.sin(theta)
        c = np.cos(theta)
        ddx = (dtheta**2*l*mp*s + f - g*mp*s*c)/(mc + mp*s**2)
        ddtheta = (g*(mc + mp)*s - (dtheta**2*l*mp*s + f)*c)/(l*(mc + mp*s**2))

        state_new = state + np.hstack([dx, ddx, dtheta, ddtheta]) * self.dt
        state_new[2] = in2pi(state_new[2])
        return state_new

    def _step(self, network, state): # evaluate network and simulate one step
        if self.velocities:
            # Divide velocities by 2.0 because that is what neat-python does
            #net_input = np.hstack((x/self.h, dx/2.0, theta/self.r, dtheta/2.0))
            net_input = state / np.array([self.h, 2.0, self.r, 2.0])
        else:
            #net_input = np.hstack((x/self.h, theta/self.r))
            net_input = state[::2]

        net_output = network.feed(net_input)

        #print('net_input  ' + str(net_input))
        #print('net_output ' + str(net_output))

        action = net_output * self.f
        state = self._simulation_step(action, state)

        return (action, state)


    def _loop(self, network, initial_state, max_steps): # evaluate network and simulate all steps
        # state = [x, dx, theta1, dtheta1, thetat2, dthetat2...]

        if not hasattr(self,'steps_all') or self.steps_all.shape[0] < max_steps:
            self.steps_all = np.arange(max_steps)
            self.time_all = self.steps_all*self.dt
            self.tolerance_all = 2*np.pi*np.exp(-0.4*self.steps_all*self.dt) + self.r

        steps  = 0
        states = []
        actions = []
        state = initial_state
        #while (steps < max_steps and np.abs(x) < self.h and ((np.abs(theta) < self.r).all() or steps < 200)):
        #while steps < max_steps and np.abs(state[0]) < self.h and (np.abs(state[2::2]) < self.tolerance_all[steps]).all():
        while steps < max_steps and np.abs(state[0]) < self.h:
            steps += 1
            action, state = self._step(network, state)
            states.append(state)
            actions.append(action)
            #print(states[-1])

        return steps, np.array(states), np.array(actions)


    def evaluate(self, network):
        initial_state = np.hstack(self.initial_state)

        steps, states, actions = self._loop(network, initial_state, self.max_steps)

        x = states[:,0]
        theta = states[:,2]

        #score= np.sum( np.abs(theta) < np.pi/4. ) / float(self.max_steps)
        score_x = np.sum(1.-np.exp(-0.4*self.h/np.abs(x))) / float(self.max_steps)
        score_theta = np.sum(1.-np.exp(-0.12*np.pi/np.abs(theta))) / float(self.max_steps)
        score = score_theta * (1+0.5*score_x)
        #print('score_x %f  score_theta %f  score %f' % (score_x, score_theta, score) )

        #score = steps/float(self.max_steps)
        #if self.penalize_oscillation:
        #    #penalty = 1.0e3/(sum( abs(dx)/self.dt for (x, dx, theta, dtheta) in states))
        #    #score = steps/float(self.max_steps) - penalty
        #    dx = states[:,0]
        #    ddx = states[:,1]
        #    dx_mean = np.mean(np.abs(np.array(dx)))/self.dt
        #    ddx_mean = np.mean(np.abs(np.diff(np.array(ddx))))/(self.dt**2)
        #    bonus_dx = (2.-np.exp(-1e2/dx_mean))    # bonus for low velocity
        #    bonus_ddx = (2.-np.exp(-1e2/ddx_mean))  # bonus for low acceleration
        #    score = score * bonus_dx * bonus_ddx
        #    #print('raw_score %f  bonus_dx %f  bonus_ddx %f  score %f  dx_mean %f  ddx_mean %f' % (steps/float(self.max_steps), bonus_dx, bonus_ddx, score, dx_mean, ddx_mean) )

        solved = int(steps >= self.max_steps)

        return [score, solved]


    # based on code from PEAS
    def visualize(self, network, filename):
        """ Visualize a solution strategy by the given individual
        """

        import matplotlib
        matplotlib.use('Agg',warn=False)
        import matplotlib.pyplot as plt

        initial_state = np.hstack(self.initial_state)
        steps, states, actions = self._loop(network, initial_state, self.max_steps)
        actions = np.array(actions)
        #print('%5d'%actions.size, np.histogram(actions)[0], ' min %s max %s'%(min(actions), max(actions)))

        g = network.genotype

        x = states[:,0]
        dx = states[:,1]
        theta = states[:,2::2]
        dtheta = states[:,3::2]

        setps_all = self.steps_all[:x.shape[0]]
        toleranc_all = self.tolerance_all[:x.shape[0]]

        fig = plt.figure()

        top = fig.add_subplot(211) # The top plot (cart position)
        top.fill_between(setps_all, -self.h, self.h, facecolor='green', alpha=0.3)
        top.plot(x, label=r'$x$')
        top.plot(dx, label=r'$\delta x$')
        top.legend(loc='lower left', ncol=4, bbox_to_anchor=(0, 0, 1, 1))

        foo = 1.-np.exp(-0.4*self.h/np.abs(x))
        top.plot(foo*self.h,'k')

        bottom = fig.add_subplot(212) # The bottom plot (pole angles)
        bottom.plot((0,steps),(0,0), 'c--' )
        #bottom.plot((0,steps),(2*np.pi,2*np.pi), 'c--' )
        #bottom.plot((0,steps),(-2*np.pi,-2*np.pi), 'c--' )
        bottom.plot((0,steps),(np.pi,np.pi), 'r--' )
        bottom.plot((0,steps),(-np.pi,-np.pi), 'r--' )
        #bottom.fill_between(setps_all, -toleranc_all, toleranc_all, facecolor='green', alpha=0.3)

        for i in range(1):
            bottom.plot(theta, label=r'$\theta_%d$'%i)
            bottom.plot(dtheta, ls='--', label=r'$\delta \theta_%d$'%i)
        bottom.legend(loc='lower left', ncol=4, bbox_to_anchor=(0, 0, 1, 1))

        bottom.plot(np.abs(theta) < np.pi/4., 'r' )
        foo = 1.-np.exp(-0.12*np.pi/np.abs(theta))
        bottom.plot(foo,'k')
        #bottom.plot(np.cumsum(foo),'b')

        fig.text(0.02,0.02,'genome_id %04d  steps %d  fitness %0.4f  solved %d' % (g.id, steps, g.fitness, g.solved))
        fig.savefig(filename)
        plt.close()


class World:
    def __init__(self, task, network):
        self.task = task
        self.network = network

        self.fps = 30
        self.display_width = 1200
        self.display_height = 400
        self.meter_pixel_ratio = 100
        self.cart_size = np.array([0.3, 0.1]) * self.meter_pixel_ratio

        if not hasattr(self.task,'name'):
            self.task.name = type(task).__name__

        # initialize pygame
        pg.init()

        self.display_size = np.array([self.display_width, self.display_height])
        self.init_pos = self.display_size/2
        self.display = pg.display.set_mode(self.display_size)

        pg.display.set_caption(self.task.name)

        self.clock = pg.time.Clock()
        self.background_color = (255, 255, 255)

        # initialize world
        self.exit = False
        self.reset = True

        self.world_loop()

    def world_loop(self):

        while not self.exit:

            if self.reset:
                #state = np.hstack(self.task.initial_state)
                state = np.array([0.0, 0.0, 2*np.pi*np.random.random(), 0.0])
                self.reset = False

            # handle events per frame
            for event in pg.event.get():
                #print(event)
                if event.type == pg.QUIT:
                    self.exit = True
                if event.type == pg.KEYDOWN:
                    if event.key == pg.K_ESCAPE or event.key == pg.K_q:
                        self.exit = True
                    if event.key == pg.K_r:
                        self.reset = True

            # update physics
            action, state = self.task._step(self.network, state)

            #print('action ' + str(action))
            #print('state  ' + str(state))

            (x, dx, theta, dtheta) = state

            if abs(x) > self.task.h:
                self.reset = True

            # redraw world
            self.display.fill(self.background_color)

            pos = (self.init_pos[0] + x*self.meter_pixel_ratio, self.init_pos[1])
            rect = pg.draw.rect(self.display, (0,255,0), (pos[0] - self.cart_size[0]/2, pos[1], self.cart_size[0], self.cart_size[1]), 3)

            l = self.task.l[0] * self.meter_pixel_ratio
            end_pos = (pos[0]+l*np.sin(theta), pos[1]-l*np.cos(theta) )
            pg.draw.line(self.display, (255,0,0), pos, end_pos, 3)

            xb = self.task.h*self.meter_pixel_ratio
            pg.draw.line(self.display, (0,0,255), (self.init_pos[0]-xb, self.init_pos[1]+self.cart_size[1]), (self.init_pos[0]+xb, self.init_pos[1]+self.cart_size[1]) )

            #pg.draw.line(self.display, (255,0,255), pos, (pos[0]+100,pos[1]-100), 3)

            pg.display.update()
            self.clock.tick(self.fps)

        if self.exit:
            pg.quit()
            quit()


if __name__ == '__main__':

    task = PoleBalanceTask()

    if len(sys.argv) > 1:
        if sys.argv[1] == 'single':
            task.max_steps = 2000
            task.initial_state = np.array([0.0, 0.0, 1.6, 0.01])
            #task.initial_state = np.array([0.0, 0.0, 1.0, 0.01])
            #task.initial_state = np.array([0.0, 0.0, 0.517, 0.01])
            task.h = 5.0

        if sys.argv[1] == 'double':
            task.name = 'DoublePoleBalanceTask'
            task.mp = np.array([0.1, 0.01])
            task.l = np.array([0.5, 0.05])
            task.max_steps = 1000
            x, dx  = 0.0, 0.0
            theta = np.array([0.017, 0.0]) # Long pole starts at a fixed 1 degree angle.
            dtheta = np.array([0.0, 0.0])
            task.initial_state = (x, dx, theta, dtheta)

        if sys.argv[1] == 'tumbler':
            task.name = 'TumblerPoleBalanceTask'
            task.dt = 0.01
            task.max_steps = 2000
            x, dx, theta, dtheta  = 0.0, 0.0, np.array([np.pi]), np.array([0.0])
            task.initial_state = (x, dx, theta, dtheta)


    if len(sys.argv) > 2:
        if len(sys.argv) > 3:
            filename = sys.argv[3]
        else:
            filename = './results/PoleBalanceTask/net-001-002.json'

        if sys.argv[2] == 'simulate':
            network = Network(None,filename=filename)
            world = World(task, network)
            sys.exit()
        if sys.argv[2] == 'visualize':
            network = Network(None,filename=filename)
            network.genotype = Object()
            fitness, solved = task.evaluate(network)
            network.genotype.fitness = fitness
            network.genotype.solved = solved
            network.visualize('net.png')
            task.visualize(network, 'sim.png')
            sys.exit()


    ga = GeneticAlgorithm(task)
    ga.visualization_type = VisualizationType.BEST
    for i in range(500):
        #import cProfile
        #p = cProfile.Profile()
        #p.enable()

        ga.epoch()

        #p.disable()
        #p.print_stats('tottime')

    pp.pprint(ga.best_ever.__dict__)

    sys.exit()


# TODO
# doble plole, serial and parallel
# draw parallel side by side
# use friction
# plot energy
# show fitness
# js eovlution tree
# disturbance, force
# norm input for tumbler into 2*pi
# compare to human
# test sets
# visualize without extra calculation
# run multiple simulation and use min score as fitness
