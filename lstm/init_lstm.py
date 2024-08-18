import numpy as np
from lstm.utils import init_orthogonal

def init_lstm(hidden_size, vocab_size, z_size):
    W_f = np.random.randn(hidden_size, z_size)
    b_f = np.zeros((hidden_size, 1))

    W_i = np.random.randn(hidden_size, z_size)
    b_i = np.zeros((hidden_size, 1))

    W_g = np.random.randn(hidden_size, z_size)
    b_g = np.zeros((hidden_size, 1))

    W_o = np.random.randn(hidden_size, z_size)
    b_o = np.zeros((hidden_size, 1))

    W_v = np.random.randn(vocab_size, hidden_size)
    b_v = np.zeros((vocab_size, 1))
    
    W_f = init_orthogonal(W_f)
    W_i = init_orthogonal(W_i)
    W_g = init_orthogonal(W_g)
    W_o = init_orthogonal(W_o)
    W_v = init_orthogonal(W_v)

    return W_f, W_i, W_g, W_o, W_v, b_f, b_i, b_g, b_o, b_v
