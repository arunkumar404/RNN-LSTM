import numpy as np
from lstm.utils import sigmoid, tanh, softmax

def forward(inputs, h_prev, C_prev, p, hidden_size):
    assert h_prev.shape == (hidden_size, 1)
    assert C_prev.shape == (hidden_size, 1)

    W_f, W_i, W_g, W_o, W_v, b_f, b_i, b_g, b_o, b_v = p
    
    x_s, z_s, f_s, i_s = [], [], [], []
    g_s, C_s, o_s, h_s = [], [], [], []
    v_s, output_s = [], []
    
    h_s.append(h_prev)
    C_s.append(C_prev)
    
    for x in inputs:
        z = np.row_stack((h_prev, x))
        z_s.append(z)
        
        f = sigmoid(np.dot(W_f, z) + b_f)
        f_s.append(f)
        
        i = sigmoid(np.dot(W_i, z) + b_i)
        i_s.append(i)
        
        g = tanh(np.dot(W_g, z) + b_g)
        g_s.append(g)
        
        C_prev = f * C_prev + i * g 
        C_s.append(C_prev)
        
        o = sigmoid(np.dot(W_o, z) + b_o)
        o_s.append(o)
        
        h_prev = o * tanh(C_prev)
        h_s.append(h_prev)

        v = np.dot(W_v, h_prev) + b_v
        v_s.append(v)
        
        output = softmax(v)
        output_s.append(output)

    return z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, output_s
