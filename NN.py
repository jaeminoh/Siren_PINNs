from jax import random
import jax.numpy as np


def relative_l2(predicted, solution):
    numerator = np.linalg.norm(predicted - solution, 2)
    denominator = np.linalg.norm(solution, 2)
    return numerator/denominator


def MLP(layers, activation):
    def init(rng_key):
        def init_layer(key, d_in, d_out):
            k1, _ = random.split(key)
            stddev = 1./np.sqrt((d_in+d_out)/2.)
            W = stddev*random.normal(k1, (d_in, d_out))
            b = np.zeros(d_out)
            return W, b
        _, *keys = random.split(rng_key, len(layers))
        params = list(map(init_layer, keys, layers[:-1], layers[1:]))
        return params
    def apply(params, inputs):
        for W, b in params[:-1]:
            outputs = np.dot(inputs, W) + b
            inputs = activation(outputs)
        W, b = params[-1]
        outputs = np.dot(inputs, W) + b
        return outputs 
    return init, apply


def Siren(layers, w0=30., activation=np.sin):
    def siren_init(key, d_in, d_out, is_first=False):
        if is_first == True:
            variance = 1/d_in
            W = random.uniform(key, (d_in, d_out), minval=-variance, maxval=variance)
            std = 1/np.sqrt(d_in)
            b = random.uniform(key, (d_out,), minval=-std, maxval=std)
            return W, b
        else:
            variance = np.sqrt(6/d_in)/w0
            W = random.uniform(key, (d_in, d_out), minval=-variance, maxval=variance)
            std = 1/np.sqrt(d_in)
            b = random.uniform(key, (d_out,), minval=-std, maxval=std)
            return W, b
    
    def init(rng_key):
        _, *keys = random.split(rng_key, len(layers))
        params = [siren_init(keys[0], layers[0], layers[1], is_first=True)] + \
            list(map(siren_init, keys[1:], layers[1:-1], layers[2:]))
        return params
    
    def apply(params, y):
        #Forward Pass
        for W, b in params[:-1]:
            pre_activation = np.dot(y*w0, W) + b
            y = activation(pre_activation)
        #Final inner product
        W, b = params[-1]
        outputs = np.dot(y*w0, W) +b
        return outputs
    return init, apply