from compact_dependencies import *
from keras.layers.recurrent import RNN
from kaulos_engine import _KaulosModel

_BACKEND = 'theano'
class LeakyIAF(_KaulosModel):
    params = OrderedDict([('threshold', 1.0), ('R', 1.0), ('C', 1.0)])
    alters = OrderedDict([('V', 0.0), ('s', 0.0)])
    inters = OrderedDict([])
    accesses = ['I']
    def kaulos_step(self):
        V = self.V + self.I
        if (_BACKEND == 'tensorflow'):
            import tensorflow as tf
            V = tf.where(K.greater(V, self.threshold), 0. * V, V)
        else:
            s = K.round(V / (2.0 * self.threshold))
            V = V - self.threshold * K.round(V / (2.0 * self.threshold))
        self.V = V
        self.s = s
        print("V: " + str(self.V))

class HodgkinHuxley(_KaulosModel):
    params = OrderedDict([('g_K', 36.0),('g_Na', 120.0),('g_l', 0.3),('E_K', -12.),
              ('E_Na', 115.), ('E_l', 10.613)])
    alters = OrderedDict([('V', 0.0),('s', 0.0)])
    inters = OrderedDict([('n', 0.0),('m', 0.0),('h', 1.0)])
    accesses = ['I']
    def kaulos_step(self):
        a_n = (25.0-self.V)/(10.0*(K.exp((25.0-self.V)/10.0)-1.0))
        a_m = (10.0-self.V)/(100.0*(K.exp((10.0-self.V)/10.0)-1.0))
        a_h = 0.07*K.exp(-self.V/20.0)

        b_n = 4.0*K.exp(-1.0 * self.V/18.0)
        b_m = 0.125*K.exp(-1.0 * self.V/80.0)
        b_h = 1.0/(K.exp((30.0-self.V)/10.0)+1.0)

        n = self.n + self.dt*(a_n*(1.0-self.n) - b_n*self.n)
        m = self.m + self.dt*(a_m*(1.0-self.m) - b_m*self.m)
        h = self.h + self.dt*(a_h*(1.0-self.h) - b_h*self.h)

        I_K = self.g_K*K.pow(m, 4)*(self.V-self.E_K)
        I_Na = self.g_Na*K.pow(n, 3)*h*(self.V-self.E_Na)
        I_l = self.g_l*(self.V-self.E_l)
        I = I_K + I_Na + I_l
        self.V = self.V + self.dt*(self.I - I)
        self.s = self.V
        self.m = m
        self.n = n
        self.h = h
        self.m = m

class AlphaSynapse(_KaulosModel):
    params = OrderedDict([('ar', 1.0),('ad', 1.0), ('gmax', 100.)])
    alters = OrderedDict([('g', 0.0)])
    inters = OrderedDict([('a_0', 0.0),('a_1', 0.0),('a_2', 1.0)])
    accesses = ['s']
    def __init__(self,**kwargs):
        super(AlphaSynapse, self).__init__(**kwargs)
    def build(self, input_shape):
        print(input_shape)
        super(AlphaSynapse, self).build(input_shape)
    def call(self, s_ext):
        new_a_0 = K.maximum( 0. , self.a_0 + self.dt*self.a_1 )
        new_a_1 = self.a_1 + self.dt*self.a_2
        new_a_1 = new_a_1 + self.ar*self.ad*s_ext
        new_a_2 = -( self.ar + self.ad )*self.a_1 - self.ar * self.ad * self.a_0
        g = new_a_0*self.gmax
        updates = []
        updates.append((self.a_0, new_a_0))
        updates.append((self.a_1, new_a_1))
        updates.append((self.a_2, new_a_2))
        updates.append((self.g, g))
        self.add_update(updates)
        return self.call_outs
