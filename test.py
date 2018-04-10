#!/usr/bin/python

import pprint
pp = pprint.PrettyPrinter(indent=4, width=120, depth=3)

from sys import exit
import numpy as np

#from tasks import *
from run_xor import XorTask
from genotype import *
from phenotype import *
from neat import *

print('\ntask')
t = XorTask()
pp.pprint(t.__dict__)

ga = GeneticAlgorithm(t)

print('\nepoch')
ga.epoch()

print('\nGenteticAlgorithm')
pp.pprint(ga.__dict__)
print('\nGenome')
g = ga.genomes[0]
pp.pprint(g.__dict__)
print('\nNeuron')
pp.pprint(g.neurons[1].__dict__)
print('\nLink')
pp.pprint(g.links[1].__dict__)
print('\nInnovation')
pp.pprint(ga.innovation_db.innovations[1].__dict__)

print('')
print('exist neuron %s %s' % (1, g.exist_neuron(1)))
print('exist neuron %s %s' % (100, g.exist_neuron(100)))
print('exist link %s-%s %s' % (2, 4, g.exist_link(2,4)))
print('exist link %s-%s %s' % (2, 100, g.exist_link(2,100)))
print('input neurons')
pp.pprint(g.get_input_neurons())
print('hidden neurons')
pp.pprint(g.get_hidden_neurons())
print('output neurons')
pp.pprint(g.get_output_neurons())

print('\nadd links')
# l = g.add_link()
ga.genomes[0]
for _ in range(5):
    l = g.add_link()
    if l == None:
        print(str(l))
    else:
        pp.pprint(l.__dict__)

print('\nadd neuron')
n,l1,l2 = g.add_neuron()
pp.pprint(n.__dict__)
pp.pprint(l1.__dict__)
pp.pprint(l2.__dict__)

print('genomes_len %s' %(len(ga.genomes)))
g = ga.genomes[2]
for _ in range(5):
    if np.random.rand() < 0.5:
        l = g.add_link()
        if l == None:
            print(str(l))
        else:
            pp.pprint(l.__dict__)
    else:
        n,l1,l2 = g.add_neuron()
        pp.pprint(n.__dict__)
        pp.pprint(l1.__dict__)
        pp.pprint(l2.__dict__)

print('\ncompatibility score')
print(ga.compatibility_score(ga.genomes[0],ga.genomes[1]))
print(ga.compatibility_score(ga.genomes[1],ga.genomes[2]))
print(ga.compatibility_score(ga.genomes[0],ga.genomes[2]))


print('\ncrossover')
g3 = ga.crossover(ga.genomes[0],ga.genomes[1])
pp.pprint(g3.__dict__)

print('\nepoch')
ga.epoch()

print('\nphenotype')
pp.pprint(g.__dict__)
p = g.create_phenotype()
pp.pprint(p.__dict__)
pp.pprint(p.neurons[1].__dict__)
pp.pprint(p.neurons[2].__dict__)
pp.pprint(p.links[1].__dict__)
pp.pprint(p.links[2].__dict__)


o = p.feed(t.input_data[2])
pp.pprint(o)
