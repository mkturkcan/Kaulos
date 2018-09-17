from .compact_dependencies import *

_BACKEND = keras.backend.backend()

class _KaulosModel(Layer):
    """Kaulos Model Layer class. This class implements the step function for a specific component type in the circuit.
    # Attributes:
        lpu_attributes (dict): The data structure that holds params, alters (output state variables) and inters (hidden state variables).
        units (int): Number of variables in the model.
        state_size (int): Size of the state variable matrix.
    """
    def __init__(self, component_units = 1, **kwargs):
        """Initialization function for the _KaulosModel class.
        # Arguments:
            component_units (int): Number of units in the Layer.
            kwargs (dict of OrderedDicts) : Contains the list of trainable parameters, parameters, state variables and initial values.
        """
        self._COMPONENT_UNITS = component_units
        self.lpu_attributes = LPU_Attr()
        self.lpu_attributes.params = self.params
        self.lpu_attributes.alters = self.alters
        self.lpu_attributes.inters = self.inters
        self.lpu_attributes.accesses = self.accesses
        self.lpu_attributes.accesses_tensors = OrderedDict()
        if 'dt' not in self.lpu_attributes.params.keys():
            self.lpu_attributes.params['dt'] = 1e-3
        if 'params_trainable' not in kwargs.keys():
            kwargs['params_trainable'] = []
        for i in self.accesses:
            self.lpu_attributes.accesses_tensors[i] = None

        self.call_outs = []
        self.units = max(int(self._COMPONENT_UNITS * len(self.accesses)), int(self._COMPONENT_UNITS * len(self.alters)))
        if len(self.inters)>0:
            self.state_size = [self.units, int(self._COMPONENT_UNITS * len(self.inters))]
        else:
            self.state_size = [self.units]
        super(_KaulosModel, self).__init__()
        self.update_lpu_attrs(**kwargs)
        # self.add_param_weights(**kwargs)
    def __getattr__(self, key):
        """Overrides certain getters for simplification purposes, especially in regards to model specification.
        # Arguments:
            key (any): The key to access.
        """
        if key in self.lpu_attributes.params:
            return self.lpu_attributes.params[key]
        if key in self.lpu_attributes.params_trainable:
            return self.lpu_attributes.params_trainable[key]
        if key in self.lpu_attributes.alters:
            return self.lpu_attributes.alters[key]
        if key in self.lpu_attributes.inters:
            return self.lpu_attributes.inters[key]
        if key in self.lpu_attributes.accesses_tensors:
            return self.lpu_attributes.accesses_tensors[key]
        else:
            return object.__getattribute__(self,key)
    def build(self, input_shape):
        """Builds the model using the given input_shape; from the Keras model specification.
        # Arguments:
            input_shape (tuple of ints): The input shape to the layer.
        """
        for a in self.lpu_attributes.alters:
            self.call_outs.append(self.lpu_attributes.alters[a])

        super(_KaulosModel, self).build(input_shape)
    def compute_output_shape(self, input_shape):
        """Computes the output shape; from the Keras model specification.
        # Arguments:
            input_shape (tuple of ints): The input shape to the layer.
        """
        return (None, len(self.alters))
    def update_lpu_attrs(self, **kwargs):
        """Creates the model LPU attributes.
        # Arguments:
            kwargs (dictionary): The extra inputs to the class constructor.
        """
        if 'params_trainable' in kwargs.keys():
            for a in kwargs['params_trainable']:
                if a in self.lpu_attributes.params.keys():
                    self.lpu_attributes.params_trainable[a] = True
        if self._COMPONENT_UNITS>1:
            for a,b in kwargs.items():
                if a == 'dt':
                    self.lpu_attributes.params[a] = b
                    self.lpu_attributes.params_trainable[a] = False
                elif a in self.lpu_attributes.params.keys():
                    if a in kwargs['params_trainable']:
                        self.lpu_attributes.params[a] = self.add_weight(name=a,
                                                      shape=(1,self._COMPONENT_UNITS),
                                                      initializer=Constant(value=float(self.lpu_attributes.params[a])),
                                                      trainable=self.lpu_attributes.params_trainable[a])
                    else:
                        if a not in self.lpu_attributes.params_trainable.keys():
                            self.lpu_attributes.params_trainable[a] = False
                        self.lpu_attributes.params[a] = self.add_weight(name=a,
                                                      shape=(1,self._COMPONENT_UNITS),
                                                      initializer=Constant(value=float(b)),
                                                      trainable=self.lpu_attributes.params_trainable[a])
                        self.lpu_attributes.params_trainable[a] = False
                if a in self.lpu_attributes.alters.keys():
                    if a not in self.lpu_attributes.params_trainable.keys():
                        self.lpu_attributes.params_trainable[a] = False
                    self.lpu_attributes.alters[a] = self.add_weight(name=a,
                                                  shape=(1,self._COMPONENT_UNITS),
                                                  initializer=Constant(value=float(b)),
                                                  trainable=self.lpu_attributes.params_trainable[a])
                if a in self.lpu_attributes.inters.keys():
                    if a not in self.lpu_attributes.params_trainable.keys():
                        self.lpu_attributes.params_trainable[a] = False
                    self.lpu_attributes.inters[a] = self.add_weight(name=a,
                                                  shape=(1,self._COMPONENT_UNITS),
                                                  initializer=Constant(value=float(b)),
                                                  trainable=self.lpu_attributes.params_trainable[a])
        else:
            for a,b in kwargs.items():
                if a in self.lpu_attributes.params.keys():
                    self.lpu_attributes.params[a] = b
                    self.lpu_attributes.params_trainable[a] = False
                if a in self.lpu_attributes.alters.keys():
                    self.lpu_attributes.alters[a] = b
                if a in self.lpu_attributes.inters.keys():
                    self.lpu_attributes.inters[a] = b
                if a == 'dt':
                    self.lpu_attributes.params[a] = b
                    self.lpu_attributes.params_trainable[a] = False
    def add_param_weights(self, **kwargs):
        """Deprecated function for adding trainable parameters.
        # Arguments:
            kwargs (dictionary): The extra inputs to the class constructor.
        """
        if 'params_trainable' in kwargs.keys():
            for a in kwargs['params_trainable']:
                if a in self.lpu_attributes.params.keys():
                    self.lpu_attributes.params_trainable[a] = True
            for a in kwargs['params_trainable']:
                if a in self.lpu_attributes.params.keys():
                    self.lpu_attributes.params[a] = self.add_weight(name=a,
                                                  shape=(1,self._COMPONENT_UNITS),
                                                  initializer=Constant(value=float(self.lpu_attributes.params[a])),
                                                  trainable=self.lpu_attributes.params_trainable[a])
    def acquire(self, I, S):
        """Slices and indexes the inputs and states.
        # Arguments:
            I (tensor): Input tensor.
            S (tensor): State tensor.
        """
        if _BACKEND == "theano":
            i = 0
            for a in self.lpu_attributes.alters:
                #print(a, i)
                self.lpu_attributes.alters[a] = S[0][:,i:i+self._COMPONENT_UNITS]
                i+=1
            i = 0
            for a in self.lpu_attributes.inters:
                #print(a, i)
                self.lpu_attributes.inters[a] = S[1][:,i:i+self._COMPONENT_UNITS]
                i+=1
            i = 0
            for a in self.lpu_attributes.accesses_tensors:
                #print(a, i)
                self.lpu_attributes.accesses_tensors[a] = I[:,i:i+self._COMPONENT_UNITS]
                i+=1
        elif _BACKEND == "tensorflow":
            i = 0
            for a in self.lpu_attributes.alters:
                #print(a, i)
                self.lpu_attributes.alters[a] = S[0][:,i:i+self._COMPONENT_UNITS]
                i+=self._COMPONENT_UNITS
            i = 0
            for a in self.lpu_attributes.inters:
                #print(a, i)
                self.lpu_attributes.inters[a] = S[1][:,i:i+self._COMPONENT_UNITS]
                i+=self._COMPONENT_UNITS
            i = 0
            for a in self.lpu_attributes.accesses_tensors:
                #print(a, i)
                self.lpu_attributes.accesses_tensors[a] = I[:,i:i+self._COMPONENT_UNITS]
                i+=self._COMPONENT_UNITS
        else: # Idea for TF splits
            all_vars = tf.split(S[0], [1 for i in range(len(self.lpu_attributes.alters.keys()) + len(self.lpu_attributes.inters.keys()))], self._COMPONENT_UNITS)
            all_ins = tf.split(S[0], [1 for i in range(max(int(len(self.accesses)), int(len(self.alters))))], self._COMPONENT_UNITS)
            i = 0
            for a in self.lpu_attributes.alters:
                #print(a, i)
                self.lpu_attributes.alters[a] = all_vars[i]
                i+=1
            i = 0
            for a in self.lpu_attributes.inters:
                #print(a, i)
                self.lpu_attributes.inters[a] = all_vars[i]
                i+=1
            i = 0
            for a in self.lpu_attributes.accesses_tensors:
                #print(a, i)
                self.lpu_attributes.accesses_tensors[a] = all_ins[i]
                i+=1
        self.Ot = S[0]
        if len(self.inters)>0:
            self.St = S[1]
    def distribute(self):
        """Reformats the step function output back into the Keras format.
        """
        if _BACKEND == "theano":
            i = 0
            for a in self.lpu_attributes.alters:
                #print(a, i, self.lpu_attributes.alters[a])
                if len(self.lpu_attributes.alters)>1:
                    self.Ot = T.set_subtensor(self.Ot[:,i:i+self._COMPONENT_UNITS], vars(self)[a])
                else:
                    self.Ot = T.set_subtensor(self.Ot[:,:], vars(self)[a])
                i += 1
            if len(self.inters)>0:
                i = 0
                for a in self.lpu_attributes.inters:
                    #print(a, i)
                    self.St = T.set_subtensor(self.St[:,i:i+self._COMPONENT_UNITS], vars(self)[a])
                    i += 1
        else:
            i = 0
            outs_list = []
            #for a in self.lpu_attributes.alters:
                #print(a, i)
                #self.lpu_attributes.alters[a]
            for a in self.lpu_attributes.alters:
                #print('Added to output from alters: ', a, i, self.lpu_attributes.alters[a])
                outs_list.append(vars(self)[a])
                #outs_list.append(self.lpu_attributes.alters[a])
                #if len(self.lpu_attributes.alters)>1:
                #    self.Ot = T.set_subtensor(self.Ot[:,i:i+1], vars(self)[a])
                #else:
                #    self.Ot = T.set_subtensor(self.Ot[:,:], vars(self)[a])
                i += 1
            #zero = tf.constant(1., dtype=tf.int32, name="kaulos_concat_zero")
            self.Ot = tf.concat(outs_list,-1)
            state_list = []
            if len(self.inters)>0:
                i = 0
                for a in self.lpu_attributes.inters:
                    #print('Added to output from inters: ', a, i)
                    #self.St = T.set_subtensor(self.St[:,i:i+1], vars(self)[a])
                    state_list.append(vars(self)[a])
                    i += 1
                self.St = tf.concat(state_list,-1)
            state_list = []
    def kaulos_step():
        """Placeholder step update function for the model; gets overridden by the model.
        """
        pass
    def call(self, I, S):
        """Wraps acquire, kaulos_step and call into one; from the Keras model specification format.
        # Arguments:
            I (tensor): Input tensor.
            S (tensor): State tensor.
        """
        self.acquire(I, S)
        self.kaulos_step()
        self.distribute()
        if len(self.inters)>0:
            return self.Ot, [self.Ot, self.St]
        else:
            return self.Ot, [self.Ot]


class LPU_Attr():
    """Basic class for LPU ("local processing unit", an abstraction for dynamical systems) attributes.
    """
    def __init__(self, **kwargs):
        self.accesses = []
        self.params = OrderedDict()
        self.params_trainable = OrderedDict()
        self.alters = OrderedDict()
        self.inters = OrderedDict()



class KaulosWrapperCell(keras.layers.Layer):
    """Kaulos Cell Layer class. This class turns circuit into a cell for Keras RNNs.
    # Attributes:
        lpu_attributes (dict): The data structure that holds params, alters (output state variables) and inters (hidden state variables).
        units (int): Number of variables in the model.
        state_size (int): Size of the state variable matrix.
    """
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
        """Builds the model using the given input_shape; from the Keras model specification.
        # Arguments:
            input_shape (tuple of ints): The input shape to the layer.
        """
        for i in self.layers:
            i.build(input_shape)
            self.trainable_weights += i.trainable_weights
        #self.kernel = self.add_weight(name='kernel',
        #                              shape=(self.units, self.units),
        #                              initializer='identity',
        #                              trainable=False)
        #if self.W is not None:
        #    self.set_weights([self.W])
        #self.built = True
        super(KaulosWrapperCell, self).build(input_shape)
    def call(self, inputs, states):
        """Call function that handles the wiring between all the different component layers; from the Keras model specification format.
        # Arguments:
            inputs (tensor): Input tensor.
            states (list of tensors): List of state tensors.
        """
        # Update connectivities
        #inputs = K.dot( inputs, self.kernel)
        # Initialize circuit
        out_states = []
        outs = []
        ii = 0
        #print(self.state_ind_len)
        # Loop through components and find the correct indices from the inputs
        # and the states that belong to them; call them and collect the results
        #if _BACKEND == 'tensorflow':
        #    tf.concat(states[1],a,[states[0].get_shape()[0], b])

        for i in self.layers:
            #print(range(int(sum(self.unit_sizes[:ii])),int(sum(self.unit_sizes[:ii+1]))))
            #print(range(int(sum(self.state_ind_len[:ii])),int(sum(self.state_ind_len[:ii+1]))))
            unit_range = range(int(sum(self.unit_sizes[:ii])),int(sum(self.unit_sizes[:ii+1])))
            state_range = range(int(sum(self.state_ind_len[:ii])),int(sum(self.state_ind_len[:ii+1])))
            call_states = []
            if _BACKEND == 'theano':
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
            else:
                if len(unit_range)>1:
                    a = tf.constant(np.array([0, unit_range[0]]), dtype=tf.int32, name="kaulos_slice_begin")
                    #print(unit_range)
                    b = tf.constant(np.array([-1, len(unit_range)]), dtype=tf.int32, name="kaulos_slice_step")
                    #print(state_range)

                    #zero_ph = tf.placeholder("int32")
                    call_states.append(tf.slice(states[0], a, b))
                    if len(self.state_sizes[ii])>1:
                        c = tf.constant(np.array([0, state_range[0]]), dtype=tf.int32, name="kaulos_slice_begin_t")
                        d = tf.constant(np.array([-1, len(state_range)]), dtype=tf.int32, name="kaulos_slice_step_t")
                        #a = tf.constant([0,state_range[0]], dtype=tf.int32, name="kaulos_slice_begin")
                        #b = tf.constant(state_range[1:], dtype=tf.int32, name="kaulos_slice_step")
                        call_states.append(tf.slice(states[1], c, d))
                    #a = tf.constant([0,unit_range[0]], dtype=tf.int32, name="kaulos_slice_begin")
                    #b = tf.constant(unit_range[1:], dtype=tf.int32, name="kaulos_slice_step")
                    a, b = i.call(tf.slice(inputs, a, b),call_states)
                else:
                    a = tf.constant(np.array([0, unit_range[0]]), dtype=tf.int32, name="kaulos_slice_begin")
                    b = tf.constant(np.array([-1, 1]), dtype=tf.int32, name="kaulos_slice_step")

                    call_states.append(tf.slice(states[0], a, b))
                    if len(self.state_sizes[ii])>1:
                        c = tf.constant(np.array([0, state_range[0]]), dtype=tf.int32, name="kaulos_slice_begin_t")
                        d = tf.constant(np.array([-1, 1]), dtype=tf.int32, name="kaulos_slice_step_t")
                        call_states.append(tf.slice(states[1], c, d))
                    a, b = i.call(tf.slice(inputs, a, b),call_states)
                out_states.append(b)
                outs += [a]
                ii += 1

        # Combine all outputs into a single tensor
        output = outs[0]
        inters_exist = False
        if len(outs)>1:
            for i in range(len(outs)-1):
                output = K.concatenate([output, outs[i+1]], axis=-1)
        #print(out_states)
        for i in range(len(out_states)):
            if len(out_states[i])>1:
                if inters_exist == False:
                    inters_exist = True
                    inters = out_states[i][1]
                else:
                    inters = K.concatenate([inters, out_states[i][1]], axis=-1)

        # Finally, add the outputs to the output states
        out_states = [output] + out_states
        #print("Input Tensor: " + str(inputs))
        #print("Input State Tensor: " + str(states))
        #print("Output Tensor: " + str(output))
        #if type(self.state_size) is list:
        #    print("Input State Tensor: " + str([output, inters]))
        if type(self.state_size) is list:
            return output, [output, inters]
        else:
            return output, [output]
