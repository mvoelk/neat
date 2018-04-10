
import pprint
pp = pprint.PrettyPrinter(indent=4, width=120, depth=3)


class InnovationType:
    NEURON, LINK = range(2)

class Innovation:
    def __init__(self, innovation_id, innovation_type, in_neuron_id=None, out_neuron_id=None, neuron_id=None, link_id=None):
        self.id = innovation_id
        self.type = innovation_type # new neuron or new link
        self.in_neuron_id = in_neuron_id
        self.out_neuron_id = out_neuron_id
        self.neuron_id = neuron_id
        self.link_id = link_id
        #self.hidden = False # ???

class InnovationDB:
    def __init__(self): # do we need number of input output neurons here?? Substrat?
        self.innovations = []
        self.next_innovation_id = 0 # innovation_id and innovation_idx are the same
        self.next_neuron_id = 0
        self.next_link_id = 0

    def exist_innovation(self, innovation_type, in_neuron_id, out_neuron_id):
        # returns innovation if exist or None if not
        for innovation in self.innovations:
            if innovation.type == innovation_type and innovation.in_neuron_id == in_neuron_id and innovation.out_neuron_id == out_neuron_id:
                return innovation
        return None

    def create_innovation(self, innovation_type, in_neuron_id=None, out_neuron_id=None):
        if in_neuron_id == None or out_neuron_id == None:
            return None
        if innovation_type == InnovationType.NEURON:
            innovation = Innovation(self.next_innovation_id, innovation_type, in_neuron_id, out_neuron_id, neuron_id=self.next_neuron_id)
            self.next_neuron_id += 1
        elif innovation_type == InnovationType.LINK:
            innovation = Innovation(self.next_innovation_id, innovation_type, in_neuron_id, out_neuron_id, link_id=self.next_link_id)
            self.next_link_id += 1
        else:
            return None
        self.innovations.append(innovation)
        self.next_innovation_id += 1
        return innovation

    def get_innovation(self, innovation_type, in_neuron_id=None, out_neuron_id=None):
        # creates innovation if not exist
        innovation = self.exist_innovation(innovation_type, in_neuron_id, out_neuron_id)
        if innovation == None:
            innovation = self.create_innovation(innovation_type, in_neuron_id, out_neuron_id)
        return innovation
