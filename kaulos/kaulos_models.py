from compact_dependencies import *
from kaulos_engine import _KaulosModel

_BACKEND = 'theano'
class LeakyIAF(_KaulosModel):
    def __init__(self, dt, threshold, alpha, **kwargs):
        self.threshold = threshold
        self.alpha = alpha
        super(LeakyIAF, self).__init__(**kwargs)
    def build(self, input_shape):
        self.input_shape_m = input_shape
        print(input_shape[1:])
        self.V = self.add_weight(
            shape=(1,1),
            name='V',
            initializer='zeros',
            trainable=False)
        super(LeakyIAF, self).build(input_shape)
    def call(self, x):
        V = self.V + x
        if (_BACKEND == 'tensorflow'):
            import tensorflow as tf
            V = tf.where(K.greater(V, self.threshold), 0. * V, V)
        else:
            V = K.switch(K.greater(V, self.threshold), 0. * V, V)
        updates = []
        updates.append((self.V, V))
        self.add_update(updates)
        return self.V
    def compute_output_shape(self, input_shape):
        return input_shape


class HodgkinHuxley(_KaulosModel):
    def __init__(self, dt, threshold, alpha, **kwargs):
        self.dt = dt
        self.threshold = threshold
        self.alpha = alpha
        self.params = {
            'g_K': 36.0,
            'g_Na': 120.0,
            'g_l': 0.3,
            'E_K': -77.,
            'E_Na': 50.,
            'E_l': -54.4
        }
        super(HodgkinHuxley, self).__init__(**kwargs)
    def build(self, input_shape):
        self.input_shape_m = input_shape
        print(input_shape[1:])
        self.V = self.add_weight(
            shape=(1,1),
            name='V',
            initializer='zeros',
            trainable=False)
        self.n = self.add_weight(
            shape=(1,1),
            name='n',
            initializer='zeros',
            trainable=False)
        self.m = self.add_weight(
            shape=(1,1),
            name='m',
            initializer='zeros',
            trainable=False)
        self.h = self.add_weight(
            shape=(1,1),
            name='h',
            initializer='ones',
            trainable=False)
        super(HodgkinHuxley, self).build(input_shape)
    def call(self, I_ext):
        a_n = (25.0-self.V)/(10.0*(K.exp((25.0-self.V)/10.0)-1.0))
        a_m = (10.0-self.V)/(100.0*(K.exp((10.0-self.V)/10.0)-1.0))
        a_h = 0.07*K.exp(-self.V/20.0)

        b_n = 4.0*K.exp(-1 * self.V/18.0)
        b_m = 0.125*K.exp(-1 * self.V/80.0)
        b_h = 1.0/(K.exp((30.0-self.V)/10.0)+1.0)

        n = self.n + self.dt*(a_n*(1.0-self.n) - b_n*self.n)
        m = self.m + self.dt*(a_m*(1.0-self.m) - b_m*self.m)
        h = self.h + self.dt*(a_h*(1.0-self.h) - b_h*self.h)

        I_K = self.g_K*K.pow(m, 4)*(self.V-self.E_K)
        I_Na = self.g_Na*K.pow(n, 3)*h*(self.V-self.E_Na)
        I_l = self.g_l*(self.V-self.E_l)
        I = I_K + I_Na + I_l
        V = self.V + self.dt*(I_ext - I)
        updates = []
        updates.append((self.m, m))
        updates.append((self.n, n))
        updates.append((self.h, h))
        updates.append((self.V, V))
        self.add_update(updates)
        return V
    def compute_output_shape(self, input_shape):
        return input_shape

class AlphaSynapse(_KaulosModel):
    def __init__(self, dt, ar, ad, **kwargs):
        self.dt = dt
        self.ar = ar
        self.ad = ad
        self.gmax = 10
        self.V_reverse = 100
        super(AlphaSynapse, self).__init__(**kwargs)
    def build(self, input_shape):
        self.input_shape_m = input_shape
        print(input_shape[1:])
        self.a_0 = self.add_weight(
            shape=(1,1),
            name='a_0',
            initializer='zeros',
            trainable=False)
        self.a_1 = self.add_weight(
            shape=(1,1),
            name='a_1',
            initializer='zeros',
            trainable=False)
        self.a_2 = self.add_weight(
            shape=(1,1),
            name='a_2',
            initializer='zeros',
            trainable=False)
        self.g = self.add_weight(
            shape=(1,1),
            name='g',
            initializer='zeros',
            trainable=False)
        super(AlphaSynapse, self).build(input_shape)
    def call(self, s_ext):
        new_a_0 = K.maximum( 0. , self.a_0 + self.dt*self.a_1 )
        new_a_1 = self.a_1 + self.dt*self.a_2
        new_a_1 = new_a_1 + self.ar*self.ad*s_ext
        new_a_2 = -( self.ar+self.ad )*self.a_1 - self.ar*self.ad*self.a_0

        g = new_a_0*self.gmax
        updates = []
        updates.append((self.a_0, new_a_0))
        updates.append((self.a_1, new_a_1))
        updates.append((self.a_2, new_a_2))
        updates.append((self.g, g))
        self.add_update(updates)
        return self.g
    def compute_output_shape(self, input_shape):
        return input_shape
