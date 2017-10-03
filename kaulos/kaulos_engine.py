from compact_dependencies import *

class _KaulosModel(Layer):
    def __init__(self, **kwargs):
        self.lpu_attributes = LPU_Attr()
        super(_KaulosModel, self).__init__(**kwargs)
        self.update_lpu_attrs(**kwargs)
    def build(self, input_shape):
        super(_KaulosModel, self).build(input_shape)
        self.update_lpu_attrs
    def update_lpu_attrs(self, **kwargs):
        for a,b in kwargs.iteritems():
            if a in self.lpu_attributes.params.keys():
                self.lpu_attributes.params[a] = b
            if a in self.lpu_attributes.updates.keys():
                self.lpu_attributes.updates[a] = b
            if a in self.lpu_attributes.states.keys():
                self.lpu_attributes.states[a] = b

class LPU_Attr():
    def __init__(self, **kwargs):
        self.accesses = []
        self.params = {}
        self.updates = {}
        self.states = {}
