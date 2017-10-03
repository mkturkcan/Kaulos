from compact_dependencies import *

class _KaulosModel(Layer):
    def __init__(self, **kwargs):
        self.lpu_attributes = LPU_Attr()
        super(_KaulosModel, self).__init__(**kwargs)
    def build(self, input_shape):
        pass

class LPU_Attr():
    def __init__(self, **kwargs):
        self.params = {}
        self.updates = {}
        self.accesses = []
        self.states = {}
