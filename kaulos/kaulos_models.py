from .compact_dependencies import *
from keras.layers.recurrent import RNN
from .kaulos_engine import _KaulosModel

_BACKEND = 'tensorflow'

def round_differentiable(x):
    """Differentiable round function. Gradient skips the function.
    # Arguments:
        x (tensor): Input tensor.
    """
    logical = K.round(x)
    return x + K.stop_gradient(logical - x)

def greater_differentiable(x,y):
    """Differentiable greater function. Gradient skips the function.
    # Arguments:
        x (tensor): Input tensor.
    """
    logical = K.cast(K.greater(x,y), 'float32')
    return x + K.stop_gradient(logical - x)

def less_differentiable(x,y):
    """Differentiable less function. Gradient skips the function.
    # Arguments:
        x (tensor): Input tensor.
    """
    logical = K.cast(K.less(x,y), 'float32')
    return x + K.stop_gradient(logical - x)

class IdealIAF(_KaulosModel):
    """Ideal Integrate and Fire neuron model. This model has a spiking threshold and a capacitance.
    """
    params = OrderedDict([('threshold', 1.0), ('C', 1.0)])
    alters = OrderedDict([('V', 0.0), ('spike', 0.0)])
    inters = OrderedDict([])
    accesses = ['I']
    def kaulos_step(self):
        V = self.V + self.dt * self.I / self.C
        spike = K.round(V / (2.0 * self.threshold))
        V = V - self.threshold * round_differentiable(V / (2.0 * self.threshold))
        self.V = V
        self.spike = spike

IdealIntegrateAndFire = IdealIAF

class LeakyIAF(_KaulosModel):
    """Leaky Integrate and Fire neuron model. This model has a spiking threshold, capacitance, and resistance.
    """
    params = OrderedDict([('threshold', 1.0), ('R', 1.0), ('C', 1.0)])
    alters = OrderedDict([('V', 0.0), ('spike', 0.0)])
    inters = OrderedDict([])
    accesses = ['I']
    def kaulos_step(self):
        V = self.V + self.dt * self.I / self.C - self.V / (self.R * self.C)
        spike = K.round(V / (2.0 * self.threshold))
        V = V - self.threshold * round_differentiable(V / (2.0 * self.threshold))
        self.V = V
        self.spike = spike

LeakyIntegrateAndFire = LeakyIAF

class Integrator(_KaulosModel):
    """An input current integrator.
    """
    params = OrderedDict([('gain', 1.0)])
    alters = OrderedDict([('V', 0.0)])
    inters = OrderedDict([])
    accesses = ['I']
    def kaulos_step(self):
        V = self.V + self.dt * self.I * self.gain
        self.V = V

class Differentiator(_KaulosModel):
    """An input current differentiator.
    """
    params = OrderedDict([('gain', 1.0)])
    alters = OrderedDict([('V', 0.0)])
    inters = OrderedDict([('Vprevious',0.0)])
    accesses = ['I']
    def kaulos_step(self):
        V = self.V - self.I
        Vprevious = self.V
        self.Vprevious = Vprevious
        self.V = V

"""
class VoltageAmplifier(_KaulosModel):
    params = OrderedDict([])
    alters = OrderedDict([('V', 0.0)])
    inters = OrderedDict([])
    accesses = ['V']
    def kaulos_step(self):
        I = self.V
        self.I = I
"""

class CurrentToVoltage(_KaulosModel):
    """Maps current to voltage. Useful depending on model construction.
    """
    params = OrderedDict([])
    alters = OrderedDict([('V', 0.0)])
    inters = OrderedDict([])
    accesses = ['I']
    def kaulos_step(self):
        V = self.I
        self.V = V

class VoltageToCurrent(_KaulosModel):
    """An input current differentiator.
    """
    params = OrderedDict([])
    alters = OrderedDict([('I', 0.0)])
    inters = OrderedDict([])
    accesses = ['V']
    def kaulos_step(self):
        I = self.V
        self.I = I

class HodgkinHuxley(_KaulosModel):
    """Hodgkin-Huxley neuron model with some default channel parameters.
    """
    params = OrderedDict([('g_K', 36.0),('g_Na', 120.0),('g_l', 0.3),('E_K', -12.),
              ('E_Na', 115.), ('E_l', 10.613)])
    alters = OrderedDict([('V', 0.0),('spike', 0.0)])
    inters = OrderedDict([('n', 0.0),('m', 0.0),('h', 1.0),('Vprev1', 0.0),('Vprev2', 0.0)])
    accesses = ['I']
    def kaulos_step(self):
        self.Vprev2 = self.Vprev1
        self.Vprev1 = self.V
        a_n = (25.0-self.V)/(10.0*(K.exp((25.0-self.V)/10.0)-1.0))
        a_m = (10.0-self.V)/(100.0*(K.exp((10.0-self.V)/10.0)-1.0))
        a_h = 0.07*K.exp(-self.V/20.0)

        b_n = 4.0*K.exp(-1.0 * self.V/18.0)
        b_m = 0.125*K.exp(-1.0 * self.V/80.0)
        b_h = 1.0/(K.exp((30.0-self.V)/10.0)+1.0)

        n = self.n + self.dt*(a_n*(1.0-self.n) - b_n*self.n)
        m = self.m + self.dt*(a_m*(1.0-self.m) - b_m*self.m)
        h = self.h + self.dt*(a_h*(1.0-self.h) - b_h*self.h)
        self.m = m
        self.n = n
        self.h = h
        I_K = self.g_K*K.pow(self.m, 4)*(self.V-self.E_K)
        I_Na = self.g_Na*K.pow(self.n, 3)*h*(self.V-self.E_Na)
        I_l = self.g_l*(self.V-self.E_l)
        I_channels = I_K + I_Na + I_l
        V = self.V + self.dt*(self.I - I_channels)
        self.V = V
        spike = K.cast(greater_differentiable(self.Vprev1, self.V), 'float32') * K.cast(less_differentiable(self.Vprev2,self.Vprev1), 'float32') * K.cast(greater_differentiable(self.V, -30.), 'float32')
        self.spike = spike


class AlphaSynapse(_KaulosModel):
    """An alpha synapse model.
    """
    params = OrderedDict([('ar', 4.0),('ad', 4.0), ('gmax', 100.), ('V_reverse_default', 100.)])
    alters = OrderedDict([('g', 0.0), ('V_reverse', 100.)])
    inters = OrderedDict([('a_0', 0.0),('a_1', 0.0),('a_2', 0.0)])
    accesses = ['spike']
    def kaulos_step(self):
        new_a_0 = K.maximum(0., self.a_0 + self.dt*self.a_1)
        new_a_1 = self.a_1 + self.dt*self.a_2 + self.ar*self.ad*self.spike
        new_a_2 = -(self.ar + self.ad)*self.a_1 - self.ar * self.ad * self.a_0
        g = K.minimum(self.gmax, self.gmax * new_a_0)
        self.a_0 = new_a_0
        self.a_1 = new_a_1
        self.a_2 = new_a_2
        self.g = g
        self.gm = g
        self.V_reverse = self.V_reverse * 0.0 + self.V_reverse_default

class AggregatorDendrite(_KaulosModel):
    """An aggregator "dendrite" that converts conductance and voltage inputs to input current.
    """
    params = OrderedDict([('ar', 4.0),('ad', 4.0), ('gmax', 100.)])
    alters = OrderedDict([('I', 0.0), ('g_modulated', 0.0)])
    inters = OrderedDict([])
    accesses = ['g','V','V_reverse']
    def kaulos_step(self):
        I = self.g * (self.V - self.V_reverse)
        self.I = I
        self.g_modulated = self.g
