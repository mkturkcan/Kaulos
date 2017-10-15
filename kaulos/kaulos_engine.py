from .compact_dependencies import *
from functools import wraps

_BACKEND = keras.backend.backend()



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

    def __setattr__(self, key, value):
        if hasattr(self, 'lpu_attributes'):
            la = self.lpu_attributes
            for p in (la.accesses_tensors, la.params, la.alters, la.inters):
                if key in p:
                    p[key] = value
                    return
        super(_KaulosModel, self).__setattr__(key, value)

    def __getattr__(self, key):
        for p in ('accesses_tensors', 'params', 'alters', 'inters'):
            attr = getattr(self.lpu_attributes, p)
            if key == p:
                return attr
            if key in attr:
                return attr[key]
        return super(_KaulosModel, self).__getattribute__(key)

    def backend_dependent(func):
        """A decorator for binding backend-specific function.
        """
        backend_dependent_func = "_%s_%s" % (_BACKEND, func.__name__)
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            """The backend-speicific function will be evoked throgh the wrapper
            function, and replace the wrapper the first time the wrapper is
            called.
            """
            new_func =  getattr(self, backend_dependent_func)
            setattr(self, func.__name__, new_func)
            return new_func(*args, **kwargs)
        return wrapper

    def build(self, input_shape):
        for a in self.lpu_attributes.alters:
            self.call_outs.append(self.lpu_attributes.alters[a])
        super(_KaulosModel, self).build(input_shape)

    def compute_output_shape(self, input_shape):
        return (None, len(self.alters))

    def update_lpu_attrs(self, **kwargs):
        for a,b in kwargs.items():
            if a in self.lpu_attributes.params.keys():
                self.lpu_attributes.params[a] = b
            if a in self.lpu_attributes.alters.keys():
                self.lpu_attributes.alters[a] = b
            if a in self.lpu_attributes.inters.keys():
                self.lpu_attributes.inters[a] = b

    @backend_dependent
    def acquire(self, I, S):
        """
        This function will be overloaded by backend-speicific implementation:
            1. _theano_acquire
            2. _tensorflow_acquire
        """
        pass

    def _theano_acquire(self, I, S):
        for i, a in enumerate(self.lpu_attributes.alters):
            self.lpu_attributes.alters[a] = S[0][:,i:i+1]
        for i, a in enumerate(self.lpu_attributes.inters):
            self.lpu_attributes.inters[a] = S[1][:,i:i+1]
        for i, a in enumerate(self.lpu_attributes.accesses_tensors):
            self.lpu_attributes.accesses_tensors[a] = I[:,i:i+1]
        self.Ot = S[0]
        if len(self.inters)>0:
            self.St = S[1]

    def _tensorflow_acquire(self, I, S):
        for i, a in enumerate(self.lpu_attributes.alters):
            self.lpu_attributes.alters[a] = S[0][:,i:i+1]
        for i, a in enumerate(self.lpu_attributes.inters):
            self.lpu_attributes.inters[a] = S[1][:,i:i+1]
        for i, a in enumerate(self.lpu_attributes.accesses_tensors):
            self.lpu_attributes.accesses_tensors[a] = I[:,i:i+1]
        self.Ot = S[0]
        if len(self.inters)>0:
            self.St = S[1]

    @backend_dependent
    def distribute(self):
        """Distrubte default value of parameters to state variables

        This function will be overloaded by backend-speicific implementation:
            1. _theano_distribute
            2. _tensorflow_distribute
        """
        pass

    def _theano_distribute(self):
        for i, v in enumerate(self.lpu_attributes.alters.values()):
            if len(self.lpu_attributes.alters)>1:
                self.Ot = T.set_subtensor(self.Ot[:,i:i+1], v)
            else:
                self.Ot = T.set_subtensor(self.Ot[:,:], v)
        if len(self.inters)>0:
            for i, v in enumerate(self.lpu_attributes.inters.values()):
                self.St = T.set_subtensor(self.St[:,i:i+1], v)

    def _tensorflow_distribute(self):
        # i = 0
        # outs_list = []
        # for a in self.lpu_attributes.alters:
        #     #print('Added to output from alters: ', a, i, self.lpu_attributes.alters[a])
        #     outs_list.append(vars(self)[a])
        #     #if len(self.lpu_attributes.alters)>1:
        #     #    self.Ot = T.set_subtensor(self.Ot[:,i:i+1], vars(self)[a])
        #     #else:
        #     #    self.Ot = T.set_subtensor(self.Ot[:,:], vars(self)[a])
        #     i += 1
        # #zero = tf.constant(1., dtype=tf.int32, name="kaulos_concat_zero")
        # self.Ot = tf.concat(outs_list,1)
        # state_list = []
        # if len(self.inters)>0:
        #     i = 0
        #     for a in self.lpu_attributes.inters:
        #         #print('Added to output from inters: ', a, i)
        #         #self.St = T.set_subtensor(self.St[:,i:i+1], vars(self)[a])
        #         state_list.append(vars(self)[a])
        #         i += 1
        #         self.St = tf.concat(state_list,1)

        outs_list = [v for v in self.lpu_attributes.alters.values()]
        self.Ot = tf.concat(outs_list,1)

        if len(self.inters)>0:
            state_list = [v for v in self.lpu_attributes.inters.values()]
            self.St = tf.concat(state_list,1)
        #     state_list = []
        #     i = 0
        #     for a in self.lpu_attributes.inters:
        #         #print('Added to output from inters: ', a, i)
        #         #self.St = T.set_subtensor(self.St[:,i:i+1], vars(self)[a])
        #         state_list.append(vars(self)[a])
        #         i += 1
        #         self.St = tf.concat(state_list,1)


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
        #for i in self.layers:
        #    i.build(input_shape)
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
                    #a = tf.constant([0,unit_range[0]], dtype=tf.int32, name="kaulos_slice_begin")
                    #b = tf.constant([None, np.array(unit_range[1:])], dtype=tf.int32, name="kaulos_slice_step")
                    call_states.append(tf.slice(states[0], np.array([0, unit_range[0]]), np.array([-1, len(unit_range)])))
                    if len(self.state_sizes[ii])>1:
                        #a = tf.constant([0,state_range[0]], dtype=tf.int32, name="kaulos_slice_begin")
                        #b = tf.constant(state_range[1:], dtype=tf.int32, name="kaulos_slice_step")
                        call_states.append(tf.slice(states[1], np.array([0,state_range[0]]), np.array([-1, len(state_range)])))
                    #a = tf.constant([0,unit_range[0]], dtype=tf.int32, name="kaulos_slice_begin")
                    #b = tf.constant(unit_range[1:], dtype=tf.int32, name="kaulos_slice_step")
                    a, b = i.call(tf.slice(inputs, np.array([0,unit_range[0]]), np.array([-1, len(unit_range)])),call_states)
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
