from kaulos import *
import time

def create_model():
    M = 2
    T = 100
    x_train = np.abs(np.random.randn(1,T,M)) * 0.1

    component = LeakyIAF()
    cell = KaulosWrapperCell([component])
    x = keras.Input(x_train.shape[1:])
    layer = RNN(cell, return_sequences = True, unroll = True)
    y = layer(x)

    model = Model(inputs=x, outputs=y)

    optimizer = Adam()
    model.compile(loss='mse', optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model, x_train

def get_model_spikes():
    model, x_train = create_model()
    model_output = model.predict(x_train)
    return np.sum(model_output[:,:,1])

def test_spikes():
    assert get_model_spikes() == 7.0
