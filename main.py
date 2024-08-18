import numpy as np
from lstm.train import train_lstm
from lstm.forward import forward
from lstm.pytorch_lstm import train_pytorch_lstm, test_pytorch_lstm
from lstm.utils import one_hot_encode_sequence

hidden_size = 50
vocab_size = len(word_to_idx)
num_epochs = 50
lr = 3e-4

# 'custom' or 'pytorch'
implementation = 'pytorch'  

if implementation == 'custom':
    params, training_loss, validation_loss = train_lstm(training_set, validation_set, vocab_size, hidden_size, num_epochs, lr)
    
    inputs, targets = test_set[1]
    inputs_one_hot = one_hot_encode_sequence(inputs, vocab_size)
    h = np.zeros((hidden_size, 1))
    c = np.zeros((hidden_size, 1))
    z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, outputs = forward(inputs_one_hot, h, c, params, hidden_size)

    output_sentence = [idx_to_word[np.argmax(output)] for output in outputs]
    print('Input sentence:', inputs)
    print('Target sequence:', targets)
    print('Predicted sequence:', output_sentence)
    
elif implementation == 'pytorch':
    net, training_loss, validation_loss = train_pytorch_lstm(training_set, validation_set, vocab_size, hidden_size, num_epochs, lr, word_to_idx)
    
    test_pytorch_lstm(net, test_set, vocab_size, word_to_idx, idx_to_word)
