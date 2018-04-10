
import numpy as np
import json


# linear algebra

def angle2(v1, v2):
    """ Returns the signed angle between two vectors 'v1' and 'v2' """
    return np.arccos(np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))) * np.sign(np.cross(v1,v2))


# json stuff

class NumPyArrayJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist() # or map(int, obj)
        return json.JSONEncoder.default(self, obj)

def loadf(filename):
    f = open(filename, 'r')
    data = json.loads(f.read())
    f.close()
    return data

def dumpf(filename, data):
    f = open(filename, 'w')
    f.write(json.dumps(data, cls=NumPyArrayJSONEncoder))
    f.close()


# statistics

class NetStatistic():
    def __init__(self, n_inputs, n_outputs):
        self.i = 1
        self.input_min = np.zeros(n_inputs)
        self.input_max = np.zeros(n_inputs)
        self.input_mean = np.zeros(n_inputs)
        self.input_var = np.zeros(n_inputs)
        self.output_min = np.zeros(n_outputs)
        self.output_max = np.zeros(n_outputs)
        self.output_mean = np.zeros(n_outputs)
        self.output_var = np.zeros(n_outputs)

    def update(self, net_input, net_output):
        i = self.i
        self.input_min = np.min(np.vstack([self.input_min, net_input]),0)
        self.input_max = np.max(np.vstack([self.input_max, net_input]),0)
        self.input_mean = (i-1.)/i * self.input_mean + 1./i * net_input
        if i > 1: self.input_var = (i-1.)/i * self.input_var + 1./(i-1) * (net_input-self.input_mean)**2
        self.output_min = np.min(np.vstack([self.output_min, net_output]),0)
        self.output_max = np.max(np.vstack([self.output_max, net_output]),0)
        self.output_mean = (i-1.)/i * self.output_mean + 1./i * net_output
        if i > 1: self.output_var = (i-1.)/i * self.output_var + 1./(i-1) * (net_output-self.output_mean)**2
        self.i += 1

    def __str__(self):
        s = ''
        for a in ['input_min', 'input_max', 'input_mean', 'input_var', 'output_min', 'output_max', 'output_mean', 'output_var']:
            d = getattr(self, a)
            s += a.ljust(12) + ' '.join(["{0: 0.6f}".format(i) for i in d]) + '\n'
        return s
