#!/usr/bin/python

import pprint
pp = pprint.PrettyPrinter(indent=4, width=120, depth=3)

import sys
import time

from genotype import *
from phenotype import *
from neat import *

class XorTask(object):
    @property
    def n_inputs(self):
        return np.shape(self.input_data)[1]
    @property
    def n_outputs(self):
        return np.shape(self.output_data)[1]

    def __init__(self):
        self.EPSILON = 1e-100
        self.input_data = np.array( [(0,0), (0,1), (1,0), (1,1)] , dtype=float)
        self.output_data = np.array( [(-1,), (1,), (1,), (-1,)] , dtype=float)
        self.solved = False

    def evaluate(self, network):
        mse = 0.0
        for (input, target) in zip(self.input_data, self.output_data):
            output = network.feed(input)
            err = target - output
            err[abs(err) < self.EPSILON] = 0
            mse += (err ** 2).mean()

        rmse = np.sqrt(mse / len(self.input_data))
        score = 1/(1+rmse) # fitness score
        solved = score > 0.99
        return [score, solved]


if __name__ == '__main__':
    task = XorTask()
    #pp.pprint(task.__dict__)

    if len(sys.argv) > 1:
        if sys.argv[1] == 'benchmark':
            a = np.array([])
            t0 = time.time()
            for j in range(50):
                ga = GeneticAlgorithm(task)
                ga.visualization_type = VisualizationType.NONE
                for i in range(500):
                    #import cProfile
                    #p = cProfile.Profile()
                    #p.enable()
                    if ga.epoch(): # solved?
                        a = np.append(a,ga.generation)
                        t = time.time() - t0
                        print('average generations %.2f std %.2f over %d runs in %.2f' % (a.mean(), a.std(), a.size, t))
                        break
                    #p.disable()
                    #p.print_stats('cumtime')
                    
        if sys.argv[1] == 'visualize':
            filename = './results/XorTask_1/net-032-014.json'
            network = Network(None,filename=filename)
            network.genotype = Object()
            fitness, solved = task.evaluate(network)
            network.genotype.fitness = fitness
            network.genotype.solved = solved
            network.visualize('net.png')

            from innovation import InnovationDB
            innovation_db = InnovationDB()
            genotype = Genome(111, innovation_db, phenotype=network)
            print("GENOTYPE")
            pp.pprint(genotype.__dict__)

            print("INNOVATION_DB")
            pp.pprint(innovation_db.__dict__)

        if sys.argv[1] == 'hotstart':
            filename = './results/XorTask_2/net-000-003.json'
            filename = './results/XorTask_1/net-032-014.json'
            ga = GeneticAlgorithm(task)
            ga.substrat = Network(None,filename=filename)
            ga.visualization_type = VisualizationType.ALL
            for i in range(500):
                if ga.epoch():
                    break
        if sys.argv[1] == 'multicore':
            ga = GeneticAlgorithm(task)
            ga.visualization_type = VisualizationType.BEST
            ga.max_cores = 3
            for i in range(500):
                if ga.epoch():
                    break
    else:
        ga = GeneticAlgorithm(task)
        ga.visualization_type = VisualizationType.ALL
        for i in range(500):
            if ga.epoch():
                break
        #pp.pprint(ga.best_ever.__dict__)
