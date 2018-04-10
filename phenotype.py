import numpy as np

import pprint
pp = pprint.PrettyPrinter(indent=4, width=120, depth=3)

class Object(object):
    pass

class NeuronType:
    INPUT, OUTPUT, HIDDEN, BIAS = range(4)

def sigmoid(x, response=1.0):
    return (1.0 / (1.0 + np.exp(-x/response)))

def sigmoid2(x, response=1/4.924273):
    # response=0.5 for tanh
    # response=1/4.924273 to get -1 at x = -1 and 1 at x = 1
    return (1.0 / (1.0 + np.exp(-x/response))) * 2.0 - 1.0

class RunType:
    SNAPSHOT, ACTIVE = range(2)

class Link:
    def __init__(self, neurons, link_gen):
        #pp.pprint(link_gen.__dict__)
        self.input_neuron = next(filter(lambda n: n.id == link_gen.from_neuron_id, neurons))
        self.output_neuron = next(filter(lambda n: n.id == link_gen.to_neuron_id, neurons))
        self.input_neuron.output_links.append(self)
        self.output_neuron.input_links.append(self)
        self.weight = link_gen.weight
        #self.recurrent = link_gen.recurrent

class Neuron:
    def __init__(self, neuron_gen):
        self.id = neuron_gen.id
        self.type = neuron_gen.type
        self.input_links = []
        self.output_links = []
        self.sum_activation = 0
        self.value = 0 # value/output?
        self.activation_response = neuron_gen.activation_response
        self.pos_x = neuron_gen.pos_x
        self.pos_y = neuron_gen.pos_y

class Network:
    def __init__(self, genome, filename=None):
        # translate genotype into phenotype
        self.genotype = genome
        self.filename = filename

        if filename != None:
            # load from file
            import json
            f = open(filename, 'r')
            data = json.loads(f.read())
            f.close()

            #pp.pprint(data)

            self.neurons = []
            splits = set()
            for d in data['neurons']:
                o = Object()
                o.id = d['id']
                o.type = d['type']
                o.activation_response = d['activation_response']
                o.pos_x = d['pos_x']
                o.pos_y = d['pos_y']
                self.neurons.append(Neuron(o))
                splits.add(o.pos_y)
            self.depth = len(splits)

            self.links = []
            for d in data['links']:
                o = Object()
                o.from_neuron_id = d['input_neuron'] # TODO: input_neuron_id
                o.to_neuron_id = d['output_neuron']
                o.weight = d['weight']
                self.links.append(Link(self.neurons, o))

        else:
            self.neurons = []
            splits = set()
            for neuron_gen in genome.neurons:
                self.neurons.append(Neuron(neuron_gen))
                splits.add(neuron_gen.pos_y)
            self.depth = len(splits)

            self.links = []
            for link_gen in genome.links:
                if not link_gen.disabled:
                    self.links.append(Link(self.neurons, link_gen))
            #        print('ENABLED LINK')
            #    else:
            #        print('DISABLED LINK')
            #
            #if len(self.links) == 0:
            #    print('NO LINKS ALARM') # we only have one disabled link in genotype
            #    pp.pprint(genome.__dict__)
            #print('-----')


    def feed(self, inputs, run_type=RunType.SNAPSHOT):

        if run_type == RunType.SNAPSHOT:
            flush_count = self.depth
            # flush to prevent dependencies on the order of training data
            for neuron in self.neurons:
                neuron.value = 0
        else:
            flush_count = 1

        for i in range(flush_count):
            #print('eval net')
            outputs = []
            i_input = 0
            i_bias = 0
            for neuron in self.neurons:
                #print('neuron type %s' % neuron.type)
                if neuron.type == NeuronType.INPUT:
                    neuron.value = inputs[i_input]
                    i_input += 1
                elif neuron.type == NeuronType.BIAS:
                    neuron.value = 1
                    i_bias += 1
                else:
                    sum = 0.0
                    for link in neuron.input_links:
                        sum = sum + link.weight * link.input_neuron.value
                    value = sigmoid2(sum, neuron.activation_response)
                    #value = sigmoid2(sum)
                    neuron.value = value
                    if neuron.type == NeuronType.OUTPUT:
                        outputs.append(value)

        return np.array(outputs)

    def visualize(self, filename):
        #print('fooo')
        #if len(self.links) == 0:
        #    print('NO LINKS %s' % self.genotype.id )
        #    print(self.links)
        #    pp.pprint(self.genotype.__dict__)
        #    pp.pprint(self.genotype.links[0].__dict__)
        import pygraphviz
        G = pygraphviz.AGraph(directed=True)
        G.graph_attr['label']=self.genotype.fitness
        for neuron in self.neurons:
            G.add_node(neuron.id)
            n = G.get_node(neuron.id)
            n.attr['pos'] = '%s,%s!' % (neuron.pos_x*10,neuron.pos_y*10)
            n.attr['style'] = 'filled'
            n.attr['shape'] = 'circle'
            if neuron.type == NeuronType.INPUT:
                #n.attr['shape'] = 'doublecircle'
                n.attr['fillcolor'] = '#ffaaaa'
            elif neuron.type == NeuronType.OUTPUT:
                #n.attr['shape'] = 'doublecircle'
                n.attr['fillcolor'] = '#aaaaff'
            else:
                n.attr['fillcolor'] = '#ffffff'
        max_weight = max(abs(l.weight) for l in self.links)
        for link in self.links:
            if abs(link.weight) > 0.001:
                G.add_edge(link.input_neuron.id, link.output_neuron.id)
                e = G.get_edge(link.input_neuron.id, link.output_neuron.id)
                e.attr['penwidth'] = abs(link.weight/max_weight*4)
                e.attr['color'] = 'blue' if link.weight > 0 else 'red'
                #e.attr['label'] = str(link.weight)
        G.draw(filename+'.png', prog='neato') # format='png',

    def dump(self, filename):
        import json
        data = {'neurons': [], 'links': []}
        for neuron in self.neurons:
            d = {   'id': neuron.id,
                    'type': neuron.type,
                    'value': neuron.value,
                    'activation_response': neuron.activation_response,
                    'pos_x': neuron.pos_x,
                    'pos_y': neuron.pos_y  }
            data['neurons'].append(d)
        for link in self.links:
            d = {   'input_neuron': link.input_neuron.id,
                    'output_neuron': link.output_neuron.id,
                    'weight': link.weight  }
            data['links'].append(d)
        f = open(filename+'.json', 'w')
        f.write(json.dumps(data))
        f.close()
