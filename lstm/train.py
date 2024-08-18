import numpy as np
from lstm.init_lstm import init_lstm
from lstm.forward import forward
from lstm.backward import backward
from lstm.utils import one_hot_encode_sequence, update_parameters

def train_lstm(training_set, validation_set, vocab_size, hidden_size, num_epochs, lr):
    z_size = hidden_size + vocab_size
    params = init_lstm(hidden_size=hidden_size, vocab_size=vocab_size, z_size=z_size)
    
    training_loss, validation_loss = [], []

    for i in range(num_epochs):
        epoch_training_loss = 0
        epoch_validation_loss = 0
        
        for inputs, targets in validation_set:
            inputs_one_hot = one_hot_encode_sequence(inputs, vocab_size)
            targets_one_hot = one_hot_encode_sequence(targets, vocab_size)

            h = np.zeros((hidden_size, 1))
            c = np.zeros((hidden_size, 1))

            z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, outputs = forward(inputs_one_hot, h, c, params, hidden_size)
            loss, _ = backward(z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, outputs, targets_one_hot, params, hidden_size)
            epoch_validation_loss += loss
        
        for inputs, targets in training_set:
            inputs_one_hot = one_hot_encode_sequence(inputs, vocab_size)
            targets_one_hot = one_hot_encode_sequence(targets, vocab_size)

            h = np.zeros((hidden_size, 1))
            c = np.zeros((hidden_size, 1))

            z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, outputs = forward(inputs_one_hot, h, c, params, hidden_size)
            loss, grads = backward(z_s, f_s, i_s, g_s, C_s, o_s, h_s, v_s, outputs, targets_one_hot, params, hidden_size)
            params = update_parameters(params, grads, lr)
            epoch_training_loss += loss

        training_loss.append(epoch_training_loss / len(training_set))
        validation_loss.append(epoch_validation_loss / len(validation_set))

        if i % 5 == 0:
            print(f'Epoch {i}, training loss: {training_loss[-1]}, validation loss: {validation_loss[-1]}')

    return params, training_loss, validation_loss
