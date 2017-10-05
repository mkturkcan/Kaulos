from compact_dependencies import *

class _KaulosModel(Layer):
    def __init__(self, **kwargs):
        self.lpu_attributes = LPU_Attr()
        self.lpu_attributes.params = self.params
        self.lpu_attributes.alters = self.alters
        self.lpu_attributes.inters = self.inters
        self.lpu_attributes.accesses = self.accesses
        self.lpu_attributes.accesses_tensors = OrderedDict()
        for i in self.accesses:
            self.lpu_attributes.accesses_tensors[i] = None
        self.update_lpu_attrs(**kwargs)
        self.lpu_attributes.params['dt'] = 1e-3
        self.call_outs = []
        self.units = max(int(len(self.accesses)), int(len(self.alters)))
        if len(self.inters)>0:
            self.state_size = [self.units, int(len(self.inters))]
        else:
            self.state_size = [self.units]
        super(_KaulosModel, self).__init__(**kwargs)
    def __getattr__(self, key):
        if key in self.lpu_attributes.params:
            return self.lpu_attributes.params[key]
        if key in self.lpu_attributes.alters:
            return self.lpu_attributes.alters[key]
        if key in self.lpu_attributes.inters:
            return self.lpu_attributes.inters[key]
        if key in self.lpu_attributes.accesses_tensors:
            return self.lpu_attributes.accesses_tensors[key]
        else:
            return super(_KaulosModel, self).__getattr__(key)
    def build(self, input_shape):
        for a in self.lpu_attributes.alters:
            self.call_outs.append(self.lpu_attributes.alters[a])
        super(_KaulosModel, self).build(input_shape)
    def compute_output_shape(self, input_shape):
        return (None, len(self.alters))
    def update_lpu_attrs(self, **kwargs):
        for a,b in kwargs.iteritems():
            if a in self.lpu_attributes.params.keys():
                self.lpu_attributes.params[a] = b
            elif a in self.lpu_attributes.alters.keys():
                self.lpu_attributes.alters[a] = b
            elif a in self.lpu_attributes.inters.keys():
                self.lpu_attributes.inters[a] = b
            else:
                raise KeyError("Unrecognized attribute %r for %r" %
                    (a, self.__class__))
    def acquire(self, I, S):
        for i, a in enumerate(self.lpu_attributes.alters):
            print(a, i)
            self.lpu_attributes.alters[a] = S[0][:,i:i+1]
        for i, a in enumerate(self.lpu_attributes.inters):
            print(a, i)
            self.lpu_attributes.inters[a] = S[1][:,i:i+1]
        for i, a in enumerate(self.lpu_attributes.accesses_tensors):
            print(a, i)
            self.lpu_attributes.accesses_tensors[a] = I[:,i:i+1]
        self.Ot = S[0]
        if len(self.inters)>0:
            self.St = S[1]
    def distribute(self):
        for i, a in enumerate(self.lpu_attributes.alters):
            print(a, i, self.lpu_attributes.alters[a])
            if len(self.lpu_attributes.alters)>1:
                self.Ot = T.set_subtensor(self.Ot[:,i:i+1], vars(self)[a])
            else:
                self.Ot = T.set_subtensor(self.Ot[:,:], vars(self)[a])
        if len(self.inters)>0:
            for i, a in enumerate(self.lpu_attributes.inters):
                print(a, i)
                self.St = T.set_subtensor(self.St[:,i:i+1], vars(self)[a])
    def call(self, I, S):
        self.acquire(I, S)
        self.kaulos_step()
        self.distribute()
        if len(self.inters)>0:
            return self.Ot, [self.Ot, self.St]
        else:
            return self.Ot, [self.Ot]
    def kaulos_step():
        pass
class LPU_Attr():
    def __init__(self, **kwargs):
        self.accesses = []
        self.params = OrderedDict()
        self.alters = OrderedDict()
        self.inters = OrderedDict()



class KaulosWrapperCell(keras.layers.Layer):
    def __init__(self, layers, W = None, **kwargs):
        self.units = 0
        self.unit_sizes = []
        self.state_size = [0, 0]
        self.state_sizes = []
        self.layers = layers
        self.state_ind_len = []
        for i in self.layers:
            self.units += i.units
            self.unit_sizes.append(i.units)
            if len(i.state_size)>1:
                self.state_size[0] += i.state_size[0]
                self.state_size[1] += i.state_size[1]
                self.state_ind_len.append(i.state_size[1])
            else:
                self.state_size[0] += i.state_size[0]
                self.state_ind_len.append(0)
            self.state_sizes.append(i.state_size)
        if self.state_size[1]==0:
            self.state_size = [self.state_size[0]]
        if len(self.state_size) == 1:
            self.state_size = self.state_size[0]
        print("Units: " + str(self.units))
        print("State Size: " + str(self.state_size))
        print("Unit Size per Layer: " + str(self.unit_sizes))
        print("State Size per Layer: " + str(self.state_sizes))
        self.W = W
        super(KaulosWrapperCell, self).__init__(**kwargs)
    def build(self, input_shape):
        for i in self.layers:
            i.build(input_shape)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.units, self.units),
                                      initializer='identity',
                                      trainable=False)
        if self.W is not None:
            self.set_weights([self.W])
        self.built = True
    def call(self, inputs, states):
        # Update connectivities
        inputs = K.dot( inputs, self.kernel)
        # Initialize circuit
        out_states = []
        outs = []
        ii = 0
        print(self.state_ind_len)
        # Loop through components and find the correct indices from the inputs
        # and the states that belong to them; call them and collect the results
        for i in self.layers:
            print(range(int(sum(self.unit_sizes[:ii])),int(sum(self.unit_sizes[:ii+1]))))
            print(range(int(sum(self.state_ind_len[:ii])),int(sum(self.state_ind_len[:ii+1]))))
            unit_range = range(int(sum(self.unit_sizes[:ii])),int(sum(self.unit_sizes[:ii+1])))
            state_range = range(int(sum(self.state_ind_len[:ii])),int(sum(self.state_ind_len[:ii+1])))
            call_states = []
            if len(unit_range)>1:
                call_states.append(states[0][:,unit_range])
                if len(self.state_sizes[ii])>1:
                    call_states.append(states[1][:,state_range])
                a, b = i.call(inputs[:,unit_range],call_states)
            else:
                call_states.append(states[0][:,unit_range[0]:unit_range[0]+1])
                if len(self.state_sizes[ii])>1:
                    call_states.append(states[1][:,state_range[0]:state_range[0]+1])
                a, b = i.call(inputs[:,unit_range[0]:unit_range[0]+1],call_states)
            out_states.append(b)
            outs += [a]
            ii += 1
        # Combine all outputs into a single tensor
        output = outs[0]
        inters_exist = False
        if len(outs)>1:
            for i in range(len(outs)-1):
                output = K.concatenate([output, outs[i+1]], axis=-1)
        print(out_states)
        for i in range(len(out_states)):
            if len(out_states[i])>1:
                if inters_exist == False:
                    inters_exist = True
                    inters = out_states[i][1]
                else:
                    inters = K.concatenate([inters, out_states[i][1]], axis=-1)

        # Finally, add the outputs to the output states
        out_states = [output] + out_states
        print("Input Tensor: " + str(inputs))
        print("Input State Tensor: " + str(states))
        print("Output Tensor: " + str(output))
        print("Output State Tensor: " + str(out_states))
        if type(self.state_size) is list:
            return output, [output, inters]
        else:
            return output, [output]
