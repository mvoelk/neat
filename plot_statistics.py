#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import json
import sys
import os

if __name__ == '__main__':
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        filename = sys.argv[1]
        f = open(filename, 'r')
        data = json.loads(f.read())
        f.close()

        species = data['species']
        n_species = len(species)
        n_generations = 1
        for i in range(n_species):
            s = np.array(species[i])
            n_generations = int(max(n_generations, np.max(s[:,0])+1))


        min_max_gen = np.zeros([n_species,2])
        sizes = np.zeros([n_generations,n_species])
        fitness = np.zeros([n_generations,n_species])
        for i in range(n_species):
            s = np.array(species[i])
            #print(s)
            min_gen = int(s[0,0])
            max_gen = int(s[-1,0])
            min_max_gen[i,:] = np.array([min_gen, max_gen])
            sizes[min_gen:max_gen+1,i] = s[:,1]


        plt.figure()
        plt.subplot(3, 1, 1)
        plt.barh(np.arange(n_species), min_max_gen[:,1]-min_max_gen[:,0], left=min_max_gen[:,0], alpha=0.8, height=1.0, facecolor='b', edgecolor='k')
        plt.grid(True)
        plt.ylabel('Species')
        plt.subplot(3, 1, 2)
        for i in range(n_species):
            s = np.array(species[i])
            plt.plot(s[:,0],s[:,2])
        plt.grid(True)
        plt.ylabel('Fitness')
        plt.subplot(3, 1, 3)
        plt.stackplot(np.arange(n_generations),sizes.T, alpha=0.4)
        plt.ylabel('Size')
        plt.xlabel('Generation')
        plt.show()

    else:
        print('nothing found')

# ./run_xor.py && ./plot_statistics.py ./results/XorTask/statistics.json
# ./plot_statistics.py results/XorTask_4/statistics.json
