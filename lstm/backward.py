import numpy as np
from lstm.utils import sigmoid, tanh, clip_gradient_norm

def backward(z, f, i, g, C, o, h, v, outputs, targets, p, hidden_size):
    W_f, W_i, W_g, W_o, W_v, b_f, b_i, b_g, b_o, b_v = p

    W_f_d = np.zeros_like(W_f)
    b_f_d = np.zeros_like(b_f)

    W_i_d = np.zeros_like(W_i)
    b_i_d = np.zeros_like(b_i)

    W_g_d = np.zeros_like(W_g)
    b_g_d = np.zeros_like(b_g)

    W_o_d = np.zeros_like(W_o)
    b_o_d = np.zeros_like(b_o)

    W_v_d = np.zeros_like(W_v)
    b_v_d = np.zeros_like(b_v)
    
    dh_next = np.zeros_like(h[0])
    dC_next = np.zeros_like(C[0])
        
    loss = 0
    
    for t in reversed(range(len(outputs))):
        loss += -np.mean(np.log(outputs[t]) * targets[t])
        C_prev = C[t-1]
        
        dv = np.copy(outputs[t])
        dv[np.argmax(targets[t])] -= 1

        W_v_d += np.dot(dv, h[t].T)
        b_v_d += dv

        dh = np.dot(W_v.T, dv)        
        dh += dh_next
        do = dh * tanh(C[t])
        do = sigmoid(o[t], derivative=True) * do
        
        W_o_d += np.dot(do, z[t].T)
        b_o_d += do

        dC = np.copy(dC_next)
        dC += dh * o[t] * tanh(tanh(C[t]), derivative=True)
        dg = dC * i[t]
        dg = tanh(g[t], derivative=True) * dg
        
        W_g_d += np.dot(dg, z[t].T)
        b_g_d += dg

        di = dC * g[t]
        di = sigmoid(i[t], True) * di
        W_i_d += np.dot(di, z[t].T)
        b_i_d += di

        df = dC * C_prev
        df = sigmoid(f[t]) * df
        W_f_d += np.dot(df, z[t].T)
        b_f_d += df

        dz = (np.dot(W_f.T, df) + np.dot(W_i.T, di) + np.dot(W_g.T, dg) + np.dot(W_o.T, do))
        dh_prev = dz[:hidden_size, :]
        dC_prev = f[t] * dC
        
    grads = W_f_d, W_i_d, W_g_d, W_o_d, W_v_d, b_f_d, b_i_d, b_g_d, b_o_d, b_v_d
    grads = clip_gradient_norm(grads)
    
    return loss, grads
