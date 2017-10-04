from compact_dependencies import *

class _KaulosModel(Layer):
    def __init__(self, **kwargs):
        self.lpu_attributes = LPU_Attr()
        self.lpu_attributes.params = self.params
        self.lpu_attributes.alters = self.alters
        self.lpu_attributes.states = self.states
        self.lpu_attributes.accesses = self.accesses
        super(_KaulosModel, self).__init__(**kwargs)
        self.update_lpu_attrs(**kwargs)
        self.lpu_attributes.params['dt'] = 1e-3
        self.call_outs = []
    def __getattr__(self, key):
        if key in self.lpu_attributes.params:
            return self.lpu_attributes.params[key]
        if key in self.lpu_attributes.alters:
            return self.lpu_attributes.alters[key]
        if key in self.lpu_attributes.states:
            return self.lpu_attributes.states[key]
        else:
            return super(_KaulosModel, self).__getattr__(key)
    def build(self, input_shape):
        for a in self.lpu_attributes.alters:
            self.lpu_attributes.alters[a] = self.add_weight(
                shape=(1,1),
                name=a,
                initializer=Constant(value=self.lpu_attributes.alters[a]),
                trainable=False)
        for a in self.lpu_attributes.states:
            self.lpu_attributes.states[a] = self.add_weight(
                shape=(1,1),
                name=a,
                initializer=Constant(value=self.lpu_attributes.states[a]),
                trainable=False)
        for a in self.lpu_attributes.alters:
            self.call_outs.append(self.lpu_attributes.alters[a])
        super(_KaulosModel, self).build(input_shape)
    def compute_output_shape(self, input_shape):
        return len(self.call_outs) * [input_shape]
    def update_lpu_attrs(self, **kwargs):
        for a,b in kwargs.iteritems():
            if a in self.lpu_attributes.params.keys():
                self.lpu_attributes.params[a] = b
            if a in self.lpu_attributes.alters.keys():
                self.lpu_attributes.alters[a] = b
            if a in self.lpu_attributes.states.keys():
                self.lpu_attributes.states[a] = b

class LPU_Attr():
    def __init__(self, **kwargs):
        self.accesses = []
        self.params = OrderedDict()
        self.alters = OrderedDict()
        self.states = OrderedDict()
