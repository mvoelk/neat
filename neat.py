import numpy as np
from copy import deepcopy
from random import randint, choice
import os
import sys
import time
import shutil
import multiprocessing

from innovation import *
from genotype import *
from species import *

import pprint
pp = pprint.PrettyPrinter(indent=4, width=120, depth=3)


class Object(object):
    pass

class VisualizationType:
    NONE, BEST, ALL = np.arange(3)

class VisualizationOutput:
    JSON, GENOTYPE, PHENOTYPE, STATISTICS = np.arange(1,5)**2


# have to be on top level so it can be pickled
def evaluate(args):
    np.seterr(over='ignore')
    task, genome = args
    print('evaluate '+str(genome.id))
    network = genome.create_phenotype()
    fitness, solved = task.evaluate(network)
    return network, fitness, solved, genome.id

class GeneticAlgorithm:
    def __init__(self, task):
        self.genomes = [] # population
        self.species = []
        self.bests = [] # champions, best individuals
        self.best_ever = None
        self.best_ever_old = None
        self.innovation_db = InnovationDB()
        self.task = task
        self.generation = 0
        self.next_genome_id = 0
        self.next_species_id = 0
        self.init_time = self.last_time = time.time()
        self.compatibility_threshold = 3.0

        self.population_size = 120 # number of individuals
        self.target_species = 30
        self.number_generation_allowed_to_not_improve = 20
        self.crossover_rate = 0.7
        self.survival_rate = 0.5
        self.trys_in_tournament_selection = 3
        self.elitism = True
        self.min_elitism_size = 1# 5
        self.young_age_threshhold = 10
        self.young_age_fitness_bonus = 1.3
        self.old_age_threshold = 50
        self.old_age_fitness_penalty = 0.7
        #self.feedforward = True
        self.visualization_type = VisualizationType.ALL
        self.visualization_output = VisualizationOutput.JSON|VisualizationOutput.GENOTYPE|VisualizationOutput.PHENOTYPE|VisualizationOutput.STATISTICS

        self.max_cores = 1
        self.substrat = None # phenotype

    def init(self):
        if not hasattr(self.task,'name'):
            self.task.name = type(self.task).__name__

        use_cores = min(self.max_cores, multiprocessing.cpu_count()-1)
        if use_cores > 1:
            self.pool = multiprocessing.Pool(processes=use_cores, maxtasksperchild=5)
        else:
            self.pool = None

        self.results_path = './results/' + self.task.name
        if os.path.exists(self.results_path):
            shutil.rmtree(self.results_path)
        os.mkdir(self.results_path)

        self.statistics = {"species":[], "generation":[]}


    def epoch(self): # evolve, evaluator, fittnes
        # main stuff is going on here

        if not hasattr(self,'statistics'):
            self.init()


        # numbe of offsprings each species should spawn
        total_average = sum(s.average_fitness for s in self.species)
        for s in self.species:
            s.spawns_required = int(round(self.population_size * s.average_fitness / total_average))
        # NEAT-Sweepers calculates average_fittnes over all genomes,
        # assignes spawn_amount=adjusted_fittnes/average_fittnes
        # and sums it for each species

        # remove stagnated species and species with no offsprings
        species = []
        for s in self.species:
            if s.generations_not_improved < self.number_generation_allowed_to_not_improve \
            and s.spawns_required > 0:
                species.append(s)
            # in NEAT-Sweepers and PEAS specie with best is not removed
        self.species[:] = species

        # reproduction
        for s in self.species:
            #s.members.sort(key=lambda x: x.fitness, reverse=True)
            # let only the best survive
            k = max(1, int(round(len(s.members) * self.survival_rate)))
            pool = s.members[:k]
            s.members[:] = []

            #print("pool_size " + str(k))
            # if elitism keep best
            if self.elitism and len(pool) > self.min_elitism_size:
                s.add_member(s.leader)
                #print("elitism add")
                #for g in pool[:int(k*0.2)]:
                #    s.add_member(g)
                #    print("elitism add")

            #co_ids = []
            while len(s.members) < s.spawns_required:
                #print('species members %s' %(len(s.members)))
                n = min(len(pool), self.trys_in_tournament_selection)
                # NEAT-Sweepers n = self.population_size/5
                # parents by turnament selection
                g1 = self.tournament_selection(pool, n)
                g2 = self.tournament_selection(pool, n)
                child = self.crossover(g1, g2, self.next_genome_id)
                self.next_genome_id += 1
                child.mutate()
                s.add_member(child)
            #print('crossover  ' + ' '.join('%3d' %(k) for k in co_ids))


        self.genomes[:] = []
        for s in self.species:
            #s.representative = choice(s.members) # TODO leader vs representative
            # PEAS chooses random representative instead of leader
            self.genomes.extend(s.members)
            s.members[:] = []
            s.age += 1

        # create basic population / inital birth
        while len(self.genomes) < self.population_size:
            #print('initial birth %s' % self.next_genome_id)
            if self.substrat != None:
                genome = Genome(self.next_genome_id, self.innovation_db, phenotype=self.substrat)
            else:
                genome = Genome(self.next_genome_id, self.innovation_db, None, None, self.task.n_inputs, self.task.n_outputs)
            self.genomes.append(genome)
            self.next_genome_id += 1

        # create new phenotypes and evaluate network
        if self.pool is not None:
            print("Running in %d processes." % self.pool._processes)
            to_eval = [(self.task, g) for g in self.genomes]
            res = self.pool.map(evaluate, to_eval) # oh no, we get pickled here!
            print('complete')
            for i in range(len(res)):
                network, fitness, solved, genome_id = res[i]
                g = self.genomes[i]
                g.phenotype = network
                g.fitness = fitness
                g.solved = int(solved)
            # TODO: it hangs some times, why?
        else:
            for g in self.genomes:
                network = g.create_phenotype()
                fitness, solved = self.task.evaluate(network)
                g.fitness = fitness
                g.solved = int(solved)


        # sort genomes by fitness
        self.genomes.sort(key=lambda x: x.fitness, reverse=True)
        # update best
        self.best_ever_old = self.best_ever
        if self.best_ever == None or self.best_ever.fitness < self.genomes[0].fitness:
            self.best_ever = self.genomes[0]

        # assigne genomes to species
        for g in self.genomes:
            added = False
            #pp.pprint(self.species)
            for s in self.species:
                # add to existing specie if compatible
                #print('compatibility %s' % self.compatibility_score(s.leader, s.leader) )
                compatibility = self.compatibility_score(g, s.leader) # TODO leader vs representative
                #print('compatibility %s ' % compatibility)
                #print('species %s leader %s genome %s  compatibility %s' % (s.id, s.leader.id, g.id, compatibility))
                if compatibility <= self.compatibility_threshold:
                    s.add_member(g)
                    added = True
                    break
            if not added:
                # create new specie if not comatible with any other
                s = Species(g, self.next_species_id)
                self.next_species_id += 1
                self.species.append(s)
                #print('new species %s' % s.id)

        # remove empty species
        self.species[:] = filter(lambda s: len(s.members) > 0, self.species)

        # adjust compatibility_threshold
        if len(self.species) < self.target_species:
            self.compatibility_threshold *= 0.95
        elif len(self.species) > self.target_species:
            self.compatibility_threshold *= 1.05
        # in PEAS compytibility_trashold is adjusteted by +/- compatibility_threshold_delta
        # in NEAT-Sweepers copytibility_trashold is fixed

        # sort members by fitness, adjust fitness, average_fitness and max_fitness
        for s in self.species:
            #s.adjust_fitnesses()

            # sort members by fitness
            s.members.sort(key=lambda x: x.fitness, reverse=True)
            s.leader_old = s.leader
            s.leader = s.members[0]

            if s.leader.fitness > s.max_fitness:
                s.generations_not_improved = 0
            else:
                s.generations_not_improved += 1
            s.max_fitness = s.leader.fitness

            # adjust fitness
            sum_fitness = 0.0
            for m in s.members:
                fitness = m.fitness
                #print('fitness %s' % fitness)
                sum_fitness += fitness
                # boost young species
                if s.age < self.young_age_threshhold:
                    fitness *= self.young_age_fitness_bonus
                # punish old species
                if s.age > self.old_age_threshold:
                    fitness *= self.old_age_fitness_penalty
                # apply fitness sharing to adjusted fitnesses
                m.adjusted_fitness = fitness/len(s.members)

            s.average_fitness = sum_fitness/len(s.members)

            #s.has_best = s.champions[-1] in specie.members

            if s.leader.fitness < s.leader_old.fitness and False:
                print('ALARM leader fitness')
                print('species_id %s' % s.id)
                for ss in self.species:
                    for mm in ss.members:
                        #print(s.leader_old.id, mm.id)
                        if mm.id == s.leader_old.id:
                            print('old leader found in species_id %s' % ss.id)

        for s in self.species:
            while len(self.statistics['species']) <= s.id:
                self.statistics['species'].append([])
            data = [self.generation, len(s.members), s.leader.fitness, s.leader.solved]
            self.statistics['species'][s.id].append(data)


        print(self)

        self._dump_statistics()

        # visualize
        if self.visualization_type == VisualizationType.ALL:
            for s in self.species:
                if s.leader != s.leader_old or s.age == 0 and s.leader.fitness != 0.0:# and s.leader.solved:
                    #print('VISUALIZE ALL  species_id %d  leadere_id %d  leader_fitness %f' % (s.id, s.leader.id, s.leader.fitness))
                    self._visualize(s.leader, 'leader')
        elif self.visualization_type == VisualizationType.BEST:
            if self.best_ever != self.best_ever_old:
                #print('VISUALIZE BEST  best_ever_id %d  best_ever_fitness %f' % (self.best_ever.id, self.best_ever.fitness))
                self._visualize(self.best_ever, 'best')

        self.generation += 1
        return self.best_ever.solved


    def __str__(self):
        # used for printing
        b = self.best_ever
        s = '\nGeneration %s' %(self.generation)
        s += '\nBest id %s fitness %0.5f neurons %s links %s depth %s' % (b.id, b.fitness, len(b.neurons), len(b.links), b.phenotype.depth)
        s += '\nspecies_id  ' + ' '.join('%4d' %(s.id) for s in self.species)
        s += '\nspawns_req  ' + ' '.join('%4d' %(s.spawns_required) for s in self.species)
        s += '\nmembers_len ' + ' '.join('%4d' %(len(s.members)) for s in self.species)
        s += '\nage         ' + ' '.join('%4d' %(s.age) for s in self.species)
        s += '\nnot_improved' + ' '.join('%4d' %(s.generations_not_improved) for s in self.species)
        s += '\nmax_fitness ' + ' '.join('%0.2f' %(s.max_fitness) for s in self.species)
        s += '\navg_fitness ' + ' '.join('%0.2f' %(s.average_fitness) for s in self.species)
        s += '\nleader      ' + ' '.join('%4d' %(s.leader.id) for s in self.species)
        s += '\nsolved      ' + ' '.join('%4d' %(s.leader.solved) for s in self.species)
        s += '\npopulation_len %s  species_len %s  compatibility_threshold %0.2f' %(len(self.genomes), len(self.species), self.compatibility_threshold)
        now = time.time()
        s += '\ntime_total %0.3f  time_per_epoch %0.3f' %(now-self.init_time, now-self.last_time)
        self.last_time = now
        s += '\n'
        return s

    def _visualize(self, genome, prefix):
        filename = '%s/%s-%03d-%03d-%05d' % (self.results_path, prefix, genome.species_id, self.generation, genome.id)
        if self.visualization_output & VisualizationOutput.JSON:
            genome.phenotype.dump(filename+'-net')
        if self.visualization_output & VisualizationOutput.GENOTYPE:
            genome.phenotype.visualize(filename+'-net')
        if self.visualization_output & VisualizationOutput.PHENOTYPE:
            if hasattr(self.task,'visualize') and callable(getattr(self.task,'visualize')):
                self.task.visualize(genome.phenotype, filename+'-sim')

    def _dump_statistics(self):
        if self.visualization_output & self.visualization_output & VisualizationOutput.JSON:
            import json
            f = open(self.results_path+'/statistics.json', 'w')
            f.write(json.dumps(self.statistics))
            f.close()


    @staticmethod
    def tournament_selection(genomes, number_to_compare):
        champion = None
        for _ in range(number_to_compare):
            g = genomes[randint(0,len(genomes)-1)]
            if champion == None or g.fitness > champion.fitness:
                champion = g
        return champion

    @staticmethod
    def crossover(mum, dad, baby_id=None): # genome1, genome2
        n_mum = len(mum.links)
        n_dad = len(dad.links)

        if mum.fitness == dad.fitness:
            if n_mum == n_dad:
                better = (mum,dad)[randint(0,1)]
            elif n_mum < n_dad:
                better = mum
            else:
                better = dad
        elif mum.fitness > dad.fitness:
            better = mum
        else:
            better = dad

        baby_neurons = []   # neuron genes
        baby_links = []     # link genes

        # iterate over parent genes
        i_mum = i_dad = 0
        neuron_ids = set()
        while i_mum < n_mum or i_dad < n_dad:
            mum_gene = mum.links[i_mum] if i_mum < n_mum else None
            dad_gene = dad.links[i_dad] if i_dad < n_dad else None
            selected_gene = None
            if mum_gene and dad_gene:
                if mum_gene.innovation_id == dad_gene.innovation_id:
                    # same innovation number, pick gene randomly from mom or dad
                    idx = randint(0,1)
                    selected_gene = (mum_gene,dad_gene)[idx]
                    selected_genome = (mum,dad)[idx]
                    i_mum += 1
                    i_dad += 1
                elif dad_gene.innovation_id < mum_gene.innovation_id:
                    # dad has lower innovation number, pick dad's gene, if they are better
                    if better == dad:
                        selected_gene = dad.links[i_dad]
                        selected_genome = dad
                    i_dad += 1
                elif mum_gene.innovation_id < dad_gene.innovation_id:
                    # mum has lower innovation number, pick mum's gene, if they are better
                    if better == mum:
                        selected_gene = mum_gene
                        selected_genome = mum
                    i_mum += 1
            elif mum_gene == None and dad_gene:
                # end of mum's genome, pick dad's gene, if they are better
                if better == dad:
                    selected_gene = dad.links[i_dad]
                    selected_genome = dad
                i_dad += 1
            elif mum_gene and dad_gene == None:
                # end of dad's genome, pick mum's gene, if they are better
                if better == mum:
                    selected_gene = mum_gene
                    selected_genome = mum
                i_mum += 1

            # add gene only when it has not already been added
            # TODO: in which case do we need this?
            if selected_gene and len(baby_links) and baby_links[len(baby_links)-1].innovation_id == selected_gene.innovation_id:
                print('GENE HAS ALREADY BEEN ADDED')
                selected_gene = None

            if selected_gene != None:
                # inherit link
                baby_links.append(deepcopy(selected_gene))

                # inherit neurons
                if not selected_gene.from_neuron_id in neuron_ids:
                    neuron = selected_genome.exist_neuron(selected_gene.from_neuron_id)
                    if neuron != None:
                        baby_neurons.append(deepcopy(neuron))
                        neuron_ids.add(selected_gene.from_neuron_id)
                if not selected_gene.to_neuron_id in neuron_ids:
                    neuron = selected_genome.exist_neuron(selected_gene.to_neuron_id)
                    if neuron != None:
                        baby_neurons.append(deepcopy(neuron))
                        neuron_ids.add(selected_gene.to_neuron_id)

            # in NEAT-Sweepers the baby neurons are created from innovations...
            # we lose activation_mutation in this case

        # add in- and output neurons if they are not connected
        for neuron in mum.get_bias_input_output_neurons():
            if not neuron.id in neuron_ids:
                baby_neurons.append(deepcopy(neuron))
                neuron_ids.add(neuron.id)

        #print('\nCROSSOVER')
        if all([l.disabled for l in baby_links]):
            # in case of same innovation id, we choose random and can end up with all links disabled
            choice(baby_links).disabled = False
            #print('ALL BABY LINKS DISABLED')
        #if better == dad:
        #    print('DAD BETTER')
        #if better == mum:
        #    print('MUM BETTER')
        #print('mum links disabled ' + str([l.disabled for l in mum.links]))
        #print('dad links disabled ' + str([l.disabled for l in dad.links]))
        #print('baby links disabled ' + str([l.disabled for l in baby_links]))
        #print('mum links ' + str(n_mum))
        #print('dad links ' + str(n_dad))
        #print('mum neurons ' + str(len(mum.neurons)))
        #print('dad neurons ' + str(len(dad.neurons)))
        #print('baby neurons ' + str(len(baby.neurons)))



        innovation_db = mum.innovation_db
        n_inputs = mum.n_inputs
        n_outputs = mum.n_outputs
        baby = Genome(baby_id, innovation_db, baby_neurons, baby_links, n_inputs, n_outputs)

        return baby


    @staticmethod
    def compatibility_score(genome1, genome2): # distance
        # for speciation
        n_match = n_disjoint = n_excess = 0
        weight_difference = 0

        n_g1 = len(genome1.links)
        n_g2 = len(genome2.links)
        i_g1 = i_g2 = 0

        while i_g1 < n_g1 or i_g2 < n_g2:
            # excess
            if i_g1 == n_g1:
                n_excess += 1
                i_g2 += 1
                continue

            if i_g2 == n_g2:
                n_excess += 1
                i_g1 += 1
                continue

            link1 = genome1.links[i_g1]
            link2 = genome2.links[i_g2]

            # match
            if link1.innovation_id == link2.innovation_id:
                #print('match')
                n_match += 1
                i_g1 += 1
                i_g2 += 1
                weight_difference = weight_difference + abs(link1.weight-link2.weight)
                # do we need activation_response_difference?
                continue

            # disjoint
            if link1.innovation_id < link2.innovation_id:
                #print('disjoint')
                n_disjoint += 1
                i_g1 += 1
                continue

            if link1.innovation_id > link2.innovation_id:
                #print('disjoint')
                n_disjoint += 1
                i_g2 += 1
                continue
        n_match += 1 # if not fully connected match can be zero and dividing by zero is bad
        score = (1.0*n_excess + 1.0*n_disjoint)/max(n_g1,n_g2) + 0.4*weight_difference/n_match
        return score
