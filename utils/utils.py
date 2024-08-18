import numpy as np

def sigmoid(x, derivative=False):
    sig = 1 / (1 + np.exp(-x))
    if derivative:
        return sig * (1 - sig)
    return sig

def tanh(x, derivative=False):
    t = np.tanh(x)
    if derivative:
        return 1 - t ** 2
    return t

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis=0, keepdims=True)

def init_orthogonal(W):
    if W.ndim == 2:
        rows, cols = W.shape
        if rows < cols:
            W = W.T
        u, _, v = np.linalg.svd(W, full_matrices=False)
        W = u if u.shape == W.shape else v
        if rows < cols:
            W = W.T
    return W

def one_hot_encode_sequence(seq, vocab_size):
    one_hot_encoded = np.zeros((len(seq), vocab_size))
    for i, index in enumerate(seq):
        one_hot_encoded[i, index] = 1
    return one_hot_encoded

def clip_gradient_norm(grads, max_norm=1.0):
    norm = np.sqrt(sum(np.sum(grad ** 2) for grad in grads))
    if norm > max_norm:
        grads = [grad * max_norm / norm for grad in grads]
    return grads

def update_parameters(params, grads, lr):
    W_f, W_i, W_g, W_o, W_v, b_f, b_i, b_g, b_o, b_v = params
    W_f_d, W_i_d, W_g_d, W_o_d, W_v_d, b_f_d, b_i_d, b_g_d, b_o_d, b_v_d = grads

    W_f -= lr * W_f_d
    b_f -= lr * b_f_d

    W_i -= lr * W_i_d
    b_i -= lr * b_i_d

    W_g -= lr * W_g_d
    b_g -= lr * b_g_d

    W_o -= lr * W_o_d
    b_o -= lr * b_o_d

    W_v -= lr * W_v_d
    b_v -= lr * b_v_d

    return W_f, W_i, W_g, W_o, W_v, b_f, b_i, b_g, b_o, b_v
