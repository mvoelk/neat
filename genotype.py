from random import random, randint, choice
from math import sqrt
import numpy as np

from innovation import *
from phenotype import Network

import pprint
pp = pprint.PrettyPrinter(indent=4, width=120, depth=3)


def rand_clamp():
    return random()*2-1

class NeuronType:
    INPUT, OUTPUT, HIDDEN, BIAS = range(4)

class NeuronGen: # Vertices, Nodes
    def __init__(self, neuron_id, neuron_type, pos_x, pos_y, activation_response=None):
        if activation_response == None:
            activation_response = 1/4.924273 # 1.0  # curvature of sigmoid function
        self.id = neuron_id
        self.type = neuron_type
        #self.recurrent = recurrent
        self.activation_response = activation_response
        self.pos_x = pos_x
        self.pos_y = pos_y
        # innovation_id ???

class LinkGen: # Edge, Connections
    def __init__(self, neuron1_id, neuron2_id, innovation_id, disabled=False, weight=None, recurrent=False):
        if weight == None:
            weight = rand_clamp()
        self.from_neuron_id = neuron1_id
        self.to_neuron_id = neuron2_id
        self.weight = weight
        self.disabled = disabled
        self.recurrent = recurrent
        self.innovation_id = innovation_id


class Genome: # Genotype of a general recurrent network

    def __init__(self, genome_id, innovation_db, neurons=None, links=None, n_inputs=2, n_outputs=1, phenotype=None):
        self.id = genome_id
        self.innovation_db = innovation_db
        self.neurons = neurons # neuron genome
        self.links = links # link genome
        self.n_inputs = n_inputs # record of number of in- and outputs
        self.n_outputs = n_outputs
        self.fitness = 0.0 # raw fitness score
        self.solved = False
        self.adjusted_fitness = 0.0 # after it has been placed into a species
        self.amount_to_spawn = 0 # number of offsprings this individual is required to spawn for the next generation
        self.species_id = None
        self.phenotype = None

        self.max_neurons = float('inf')
        #self.max_depth = 5
        self.weight_mutation_rate = 0.8
        self.activation_mutation_rate = 0.1
        #self.max_weight_perturbation = 0.5
        self.max_activation_perturbation = 0.1
        self.tries_to_find_old_link = 5
        #self.tries_to_find_none_recurrent_link = 5
        self.tries_to_find_unlinked_neurons = 5
        self.chance_to_add_link = 0.3 #0.07
        self.chance_to_add_neuron = 0.03
        self.chance_to_add_recurrent_link = 0.05
        self.chance_to_reset_weight = 0.1
        #self.chance_to_disable_link = 0.01
        #self.chance_to_reenable_link = 0.01
        #self.chance_to_delete_neuron = 0.01

        #self.weight_range = (-50., 50.)
        #self.weight_range = (-3., 3.) # for xor
        self.weight_range = (-10., 10.)
        self.stdev_weight = 2.0
        self.stdev_mutate_weight = 1.5
        self.stdev_mutate_response = 1.0

        self.fsneat = False

        # with neurons and links we are done
        if neurons != None:
            if np.all([l.disabled for l in links]):
                pass
                #None**3
            # sort neurons by innovation number
            self.neurons.sort(key=lambda x: x.id)
            for i in range(len(self.neurons)):
                self.neurons[i].idx = i
            return

        # create genome from phenotype
        if phenotype != None:
            n_inputs = 0
            n_outputs = 0
            next_neuron_id = 0
            self.neurons = []
            # build substrat
            for x in phenotype.neurons:
                neuron = NeuronGen(x.id, x.type, x.pos_x, x.pos_y, x.activation_response)
                if neuron.type == NeuronType.INPUT:
                    n_inputs += 1
                if neuron.type == NeuronType.OUTPUT:
                    n_outputs += 1
                self.neurons.append(neuron)
                next_neuron_id = max(next_neuron_id, x.id)
            next_neuron_id += 1
            innovation_db.next_neuron_id = max(innovation_db.next_neuron_id, next_neuron_id)

            self.links = []
            # create links
            for x in phenotype.links:
                input_neuron_id = x.input_neuron.id
                output_neuron_id = x.output_neuron.id
                innovation = innovation_db.get_innovation(InnovationType.LINK, in_neuron_id=input_neuron_id, out_neuron_id=output_neuron_id)
                link = LinkGen(input_neuron_id, output_neuron_id, innovation.id, weight=x.weight)
                self.links.append(link)
            return


        # crate genome based on number of in- and outputs
        input_pos_x = 1./(n_inputs+1)
        output_pos_x = 1./(n_outputs)
        next_neuron_id = 0
        self.neurons = []
        # create bias gene
        self.neurons.append(NeuronGen(next_neuron_id, NeuronType.BIAS, 0.5*input_pos_x, 0.0))
        next_neuron_id += 1
        # creat input neuron genes
        for i in range(n_inputs):
            self.neurons.append(NeuronGen(next_neuron_id, NeuronType.INPUT, (i+1+0.5)*input_pos_x, 0.0))
            next_neuron_id += 1
        # creat output neuron genes
        for i in range(n_outputs):
            self.neurons.append(NeuronGen(next_neuron_id, NeuronType.OUTPUT, (i+0.5)*output_pos_x, 1.0))
            next_neuron_id += 1
        # innovation_db.next_neuron_id = next_neuron_id # TODO: we do this in every creation of a net inital genome
        innovation_db.next_neuron_id = max(innovation_db.next_neuron_id, next_neuron_id)

        # create link genes
        self.links = []

        if self.fsneat == True:
            # connect random one input to one output, known as Feature Selection NEAT (FS-NEAT)
            i = choice(self.get_input_neurons())
            o = choice(self.get_output_neurons())
            innovation = innovation_db.get_innovation(InnovationType.LINK, in_neuron_id=i.id, out_neuron_id=o.id)
            weight = np.random.normal(0, self.stdev_weight)
            self.links.append(LinkGen(i.id, o.id, innovation.id, weight=weight))
        else:
            # fully connected
            for i in self.get_bias_input_neurons():
                for o in self.get_output_neurons():
                    innovation = innovation_db.get_innovation(InnovationType.LINK, in_neuron_id=i.id, out_neuron_id=o.id)
                    weight = np.random.normal(0, self.stdev_weight)
                    self.links.append(LinkGen(i.id, o.id, innovation.id, weight=weight))
        #print(self)


    def get_bias_neurons(self):# we can done faster, we know where they are
        return [ x for x in self.neurons if x.type == NeuronType.BIAS ]

    def get_input_neurons(self):
        return [ x for x in self.neurons if x.type == NeuronType.INPUT ]

    def get_output_neurons(self):
        return [ x for x in self.neurons if x.type == NeuronType.OUTPUT ]

    def get_hidden_neurons(self):
        return [ x for x in self.neurons if x.type == NeuronType.HIDDEN ]

    def get_bias_input_neurons(self):
        return [ x for x in self.neurons if x.type == NeuronType.INPUT or x.type == NeuronType.BIAS ]

    def get_bias_input_output_neurons(self):
        return [ x for x in self.neurons if x.type == NeuronType.INPUT or x.type == NeuronType.BIAS ]

    def exist_link(self, neuron1_id, neuron2_id):
        for link in self.links:
            if link.from_neuron_id == neuron1_id and link.to_neuron_id == neuron2_id:
                return link
        return None

    def exist_neuron(self, neuron_id):
        for neuron in self.neurons:
            if neuron.id == neuron_id:
                return neuron
        return None

    def add_link(self):
        # forward, recurrent, looped recurrent
        neuron1 = neuron2 = None

        # TODO: chance to add recurrentl link, in neuron selection loop?
        #if random() < self.chance_to_add_recurrent_link:
        #    # try to find a neuron that is not an input or bias neuron and does not have a loopback
        #    for _ in range(self.tries_to_find_none_recurrent_link):
        #        tmp_neuron = self.neurons[randint(1+self.n_inputs,len(self.neurons)-1)]
        #        #if not tmp_neuron.recurrent and tmp_neuron.type != NeuronType.BIAS and tmp_neuron.type != NeuronType.INPUT:
        #        if tmp_neuron.type != NeuronType.BIAS and tmp_neuron.type != NeuronType.INPUT:
        #            neuron1 = neuron2 = tmp_neuron
        #            recurrent = neuron1.recurrent = True
        #            break

        if neuron1 == None:
            # tries to find tow unlinked neurons
            for _ in range(self.tries_to_find_unlinked_neurons):
                tmp_neuron1 = self.neurons[randint(0,len(self.neurons)-1)]
                tmp_neuron2 = self.neurons[randint(1+self.n_inputs,len(self.neurons)-1)]

                #if not self.exist_link(tmp_neuron1.id, tmp_neuron2.id) and tmp_neuron1.id != tmp_neuron2.id:
                if not self.exist_link(tmp_neuron1.id, tmp_neuron2.id):
                    if tmp_neuron1.pos_y >= tmp_neuron2.pos_y:
                        if random() < self.chance_to_add_recurrent_link:
                            neuron1 = tmp_neuron1
                            neuron2 = tmp_neuron2
                            recurrent = True
                            break
                    else:
                        neuron1 = tmp_neuron1
                        neuron2 = tmp_neuron2
                        recurrent = False
                        break

        if neuron1 == None or neuron2 == None:
            #print('NO NEURONS to add link')
            return None

        # is link recurrent
        #recurrent = neuron1.pos_y >= neuron2.pos_y # TODO: do we need a recurrent flag?
        #if recurrent:
        #    print("ADD RECURRENT LINK")

        innovation = self.innovation_db.get_innovation(InnovationType.LINK, neuron1.id, neuron2.id)

        weight = np.random.normal(0, self.stdev_weight)
        link = LinkGen(neuron1.id, neuron2.id, innovation.id, weight=weight, recurrent=recurrent)
        self.links.append(link)
        #print('genome %s  innovation %s  link %s --> %s' %(self.id, innovation.id, neuron1.id, neuron2.id))
        return link

    def add_neuron(self):

        # find link to split
        link = None
        # if genome is less then 5 hidden neurons, it is considdered to be to samll to select a link at random
        # and we prefere older links to prevent chaining effects TODO
        size_threshold = self.n_inputs + self.n_outputs + 5
        if len(self.links) < size_threshold:
            for _ in range(self.tries_to_find_old_link):
                tmp_link = self.links[randint(0,len(self.links)-1-int(sqrt(len(self.links)-1)))]
                if not tmp_link.disabled and not tmp_link.recurrent and self.exist_neuron(tmp_link.from_neuron_id).type != NeuronType.BIAS:
                    link = tmp_link
                    break
            if link == None:
                return
        else:
            while link == None:
                tmp_link = self.links[randint(0,len(self.links)-1)]
                if not tmp_link.disabled and not tmp_link.recurrent and self.exist_neuron(tmp_link.from_neuron_id).type != NeuronType.BIAS:
                    link = tmp_link


        from_neuron = self.exist_neuron(link.from_neuron_id)
        to_neuron = self.exist_neuron(link.to_neuron_id)

        split_x = (from_neuron.pos_x + to_neuron.pos_x) / 2
        split_y = (from_neuron.pos_y + to_neuron.pos_y) / 2
        recurrent = from_neuron.pos_y > to_neuron.pos_y

        innovation = self.innovation_db.get_innovation(InnovationType.NEURON, from_neuron.id, to_neuron.id)

        # TODO: test for existance while selection?
        neuron = self.exist_neuron(innovation.neuron_id)
        if neuron == None: # can exist due to crossover
            neuron = NeuronGen(innovation.neuron_id, NeuronType.HIDDEN, split_x, split_y)
            #print('genome %s  innovation %s  neuron %s --> %s (%s)' %(self.id, innovation.id, link.from_neuron_id, link.to_neuron_id, neuron.id))
            self.neurons.append(neuron)

            innovation1 = self.innovation_db.get_innovation(InnovationType.LINK, from_neuron.id, neuron.id)
            link1 = LinkGen(from_neuron.id, neuron.id, innovation1.id, weight=1.0, recurrent=recurrent)
            self.links.append(link1)

            innovation2 = self.innovation_db.get_innovation(InnovationType.LINK, neuron.id, link.to_neuron_id)
            link2 = LinkGen(neuron.id, to_neuron.id, innovation2.id, weight=link.weight, recurrent=recurrent)
            self.links.append(link2)

            link.disabled = True
            #print('links disabled ' + str([l.disabled for l in self.links]))

            return (neuron, link1, link2)
        else:
            return None

    def mutate(self):
        # add neuron
        if random() < self.chance_to_add_neuron and len(self.neurons) < self.max_neurons:
            self.add_neuron()
            #pass

        # add link
        if random() < self.chance_to_add_link:
            self.add_link()

        # mutate weights
        for link in self.links:
            if random() < self.weight_mutation_rate:
                if random() < self.chance_to_reset_weight:
                    #link.weight = rand_clamp()
                    link.weight = np.random.normal(0, self.stdev_weight)
                else:
                    link.weight += np.random.normal(0, self.stdev_mutate_weight)
                    # NEAT-Sweepers uses uniform distribution and max_weight_perturbation = 0.5
                    #link.weight += rand_clamp() * self.max_weight_perturbation
                    link.weight = np.clip(link.weight, self.weight_range[0], self.weight_range[1])

        # mutate activation response
        for neuron in self.neurons:
            if random() > self.activation_mutation_rate:
                #if self.stdev_mutate_response > 0.0:
                #    neuron.activation_response += np.random.normal(0, self.stdev_mutate_response)
                neuron.activation_response += rand_clamp() * self.max_activation_perturbation

    def create_phenotype(self):
        self.phenotype = Network(self)
        return self.phenotype

    def __str__(self):
        s = 'genome %s %s' %(self.id, self.fitness)
        s += '\nn_inputs ' + str(self.n_inputs)
        s += '\nn_outputs ' + str(self.n_outputs)
        s += '\nlen_bias ' + str(len(self.get_bias_neurons()))
        s += '\nlen_input ' + str(len(self.get_input_neurons()))
        s += '\nlen_hidden ' + str(len(self.get_hidden_neurons()))
        s += '\nlen_output ' + str(len(self.get_output_neurons()))
        return s
