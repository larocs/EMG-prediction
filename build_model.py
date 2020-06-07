import numpy as np
from keras.layers import Input, Dense, Flatten, Activation, Dropout, LSTM, RepeatVector, TimeDistributed, ConvLSTM2D, GRU
from keras.layers import Add, Subtract, Multiply, ReLU, ThresholdedReLU, Concatenate, GlobalAveragePooling1D, GlobalMaxPooling1D, GlobalAvgPool1D, LeakyReLU
from keras.layers.wrappers import Bidirectional
from keras.layers.convolutional import Conv1D, MaxPooling1D, UpSampling1D
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l1, l2, l1_l2
from keras.models import Sequential, Model
from keras import optimizers, regularizers
from sklearn.utils import class_weight
from keras import backend as K
from loss_functions import emg_error, correlation_loss, mean_absolute_percentage_error, emg_plus_fft, loss_fft, loss_fft_filter, emg_on_fft

def swish(x):
    return K.sigmoid(x) * x

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def create_model(params):
    model_name = params['model']
    model_func = switcher[model_name]
    return model_func(params)

def rnn_lstm_envelope(params):
    
    """BuildLSTM model on top of Keras and Tensorflow"""

    # set input and network params
    input_dim = params['data_dim']
    timesteps = params['timesteps']
    prediction_timesteps = params['prediction_timesteps']

    # optimization parameters
    loss_function = params['loss_function']
    optimizer = params['optimizer']
    activation_function = params['activation_function']
    dropout = params['dropout_keep_prob']

    inputs = Input(shape=(params['timesteps'], input_dim))
    outputs = LSTM(20,input_shape=(timesteps, input_dim), activation=activation_function, dropout=dropout)(inputs)
    outputs = Dense(prediction_timesteps)(outputs)
    model = Model(inputs=[inputs], outputs=outputs)

    model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model

def rnn_lstm(params):
    
    """Build a stacked LSTM model on top of Keras and Tensorflow"""

    # set input and network params
    input_dim = params['data_dim']
    hidden_layers = params['hidden_units']
    lstm_layers = len(hidden_layers)
    dense_units = params['dense_units']
    dense_layers = len(dense_units)
    timesteps = params['timesteps']

    # regularization params
    dropout = params['dropout_keep_prob']
    regularization_l1 = params['regularization_l1']
    regularization_l2 = params['regularization_l2']

    # optimization parameters
    loss_function = params['loss_function']
    optimizer = params['optimizer']
    activation_function = params['activation_function']
    
    if activation_function == 'leakyrelu':
        leaky_alpha =  params['leaky_alpha']
        activation = LeakyReLU(alpha=leaky_alpha)
        activation.__name__ = 'LeakyReLu'
    else:
        activation = activation_function

    inputs = Input(shape=(params['timesteps'], input_dim))

    if lstm_layers > 1:
        output = LSTM(hidden_layers[0], dropout=dropout, return_sequences=True)(inputs)
        for i in range(lstm_layers-1):
            if i == (lstm_layers-2):
                output = LSTM(hidden_layers[i], dropout=dropout, return_sequences=False)(output)
            else:
                output = LSTM(hidden_layers[i], dropout=dropout, return_sequences=True)(output)
    
    else:
        output = LSTM(hidden_layers[0], dropout=dropout, return_sequences=False)(inputs)
       
    for i in range(dense_layers):
        if dropout > 0:
            output = Dropout(dropout)(output)
        output = Dense(dense_units[i], activation=activation_function)(output)
    
    model = Model(inputs=[inputs], outputs=[output])

    model.compile(optimizer=optimizer, loss=loss_function, metrics = ['accuracy',rmse])
    model.summary()
    return model

def multi_lstm(params):
    
    """Build a parallel LSTM model on top of Keras and Tensorflow"""

    # set input and network params
    input_dim = params['data_dim']
    hidden_layers = params['hidden_units']
    lstm_layers = len(hidden_layers)
    dense_units = params['dense_units']
    dense_layers = len(dense_units)
    timesteps = params['timesteps']

    # optimization parameters
    loss_function = params['loss_function']
    optimizer = params['optimizer']
    activation_function = params['activation_function']
    dropout = params['dropout_keep_prob']

    inputs = Input(shape=(params['timesteps'], input_dim))

    lstms = []
        
    for i in range(len(lstm_layers)-1):
        new_lstm = LSTM(hidden_layers[i], dropout=dropout, return_sequences=False)(inputs)
        lstms.append(new_lstm)
    
    output = Concatenate()(lstms)
    
    for i in range(len(dense_layers)-1):
        if dropout_keep_prob > 0:
            output = Dropout(dropout_keep_prob)(output)
        output = Dense(dense_layers[i], activation=activation_function)(output)

    model = Model(inputs=[inputs], outputs=[output])

    model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])
    model.summary()
    return model

def bi_lstm(params):
	
	"""BuildLSTM model on top of Keras and Tensorflow"""

	# set input and network params
	input_dim = params['data_dim']
	hidden_layers = params['hidden_units']
	dense_layers = params['dense_units']
	timesteps = params['timesteps']

	# optimization parameters
	loss_function = params['loss_function']
	optimizer = params['optimizer']
	activation_function = params['activation_function']
	dropout = params['dropout_keep_prob']

	inputs = Input(shape=(params['timesteps'], input_dim))

	lstm_1 = Bidirectional(LSTM(400, activation=activation_function, dropout=dropout))(inputs)
	lstm_2 = Bidirectional(LSTM(100, activation=activation_function, dropout=dropout))(inputs)
	lstm_3 = Bidirectional(LSTM(10, activation=activation_function, dropout=dropout))(inputs)
	output = Concatenate()([lstm_1,lstm_2,lstm_3])
	output = Dense(1)(output)

	model = Model(inputs=[inputs], outputs=[output])

	#adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	#rmsprop = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
	#model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

	model.compile(loss=loss_function, optimizer=optimizer)
	model.summary()
	return model

def rnn_gru(params):
	
	"""BuildLSTM model on top of Keras and Tensorflow"""

	# set input and network params
	input_dim = params['data_dim']
	hidden_layers = params['hidden_units']
	dense_layers = params['dense_units']
	timesteps = params['timesteps']

	# optimization parameters
	loss_function = params['loss_function']
	optimizer = params['optimizer']
	activation_function = params['activation_function']
	dropout = params['dropout_keep_prob']

	inputs = Input(shape=(800,1))

	output = GRU(200,input_shape=(800,1), dropout=dropout, return_sequences=True)(inputs)
	output = GRU(200, dropout=dropout, return_sequences=True)(output)
	output = GRU(200, dropout=dropout)(output)
	output = Dense(100)(output)
	output = Dense(1)(output)

	model = Model(inputs=[inputs], outputs=[output])

	#adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	#rmsprop = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
	#model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

	model.compile(loss=loss_function, optimizer=optimizer, metrics=['accuracy'])
	model.summary()
	return model

def predict_next_timestamp(model, history):
	"""Predict the next time stamp given a sequence of history data"""

	prediction = model.predict(history)
	#prediction = np.reshape(prediction, (prediction.size,))
	return prediction

def predict_next_window(model, history, window_size):
	"""Predict the next time stamp given a sequence of history data"""

	# Include DENSE Layers
	prediction = []
	for j in range((int)(len(history)/window_size) - 1):
		x = history[j*window_size]
		x = np.array(x).reshape(1,100,5)
		y = []
		for i in range(window_size):
			y = model.predict_on_batch(x)
			x = x[:,1:,:]
			x = np.insert(x, len(x), y, axis=1)
		prediction.append(y)
		
	return prediction

def predict_sequence_full(model, history, window_size):
	#Shift the window by 1 new prediction each time, re-run predictions on new window
	curr_frame = history[0]
	predicted = []
	for i in range(len(history)):
		predicted.append(model.predict(curr_frame[newaxis,:,:])[0,0])
		curr_frame = curr_frame[1:]
		curr_frame = np.insert(curr_frame, [window_size-2], predicted[-1], axis=0)
	return predicted

def lstm_autoencoder(params):
	
	"""BuildLSTM model on top of Keras and Tensorflow"""
	
	input_dim = params['data_dim']
	hidden_layers = params['hidden_units']
	dense_layers = params['dense_units']

	timesteps = params['timesteps']
	prediction_timesteps = params['prediction_timesteps']
	activation_function = params['activation_function']
	dropout_keep_prob = params['dropout_keep_prob']

	inputs = Input(shape=(timesteps, input_dim))

	# define encoder
	encoder = LSTM(100, activation=activation_function, return_sequences=False, dropout=dropout_keep_prob)(inputs)
	# define reconstruct decoder
	repeater = RepeatVector(timesteps)(encoder)
	decoder1 = LSTM(100, activation=activation_function)(repeater)
	decoder1 = Dense(4000)(decoder1)
	# define predict decoder
	#decoder2 = RepeatVector(prediction_timesteps)(encoder)
	#decoder2 = LSTM(20, activation=activation_function, return_sequences=True)(decoder2)
	#decoder2 = TimeDistributed(Dense(1))(decoder2)
	# tie it together
	#output_layers = [decoder1, decoder2]
	output_layers = [decoder1]
	model = Model(inputs=inputs, outputs=output_layers)
	model.compile(loss=params['loss_function'], optimizer=params['optimizer'], metrics=['accuracy'])

	encoder = Model(inputs, encoder)
	return model, encoder

def lstm_decoder(params):
	
	"""BuildLSTM model on top of Keras and Tensorflow"""
	hidden_layers = params['hidden_units']
	dense_layers = params['dense_units']

	timesteps = params['timesteps']
	prediction_timesteps = params['prediction_timesteps']
	activation_function = params['activation_function']
	dropout_keep_prob = params['dropout_keep_prob']

	inputs = Input(shape=(100, ))

	# define reconstruct decoder
	repeater = RepeatVector(timesteps)(inputs)
	decoder1 = LSTM(100, activation=activation_function)(repeater)
	decoder1 = Dense(400)(decoder1)
	# define predict decoder
	#decoder2 = RepeatVector(prediction_timesteps)(encoder)
	#decoder2 = LSTM(20, activation=activation_function, return_sequences=True)(decoder2)
	#decoder2 = TimeDistributed(Dense(1))(decoder2)
	# tie it together
	#output_layers = [decoder1, decoder2]
	output_layers = [decoder1]
	model = Model(inputs=inputs, outputs=output_layers)
	model.compile(loss=params['loss_function'], optimizer=params['optimizer'], metrics=['accuracy'])

	return model

def lstm_encoder_simple(params):
	# define input sequence
	input_dim = params['data_dim']
	output_dim = params['output_dim']	
	activation_function = params['activation_function']
	dropout = params['dropout_keep_prob']	
	timesteps = params['timesteps']
	prediction_timesteps = params['prediction_timesteps']
	optimizer = params['optimizer']
	loss_function = params['loss_function']

	# define model
	model = Sequential()
	model.add(LSTM(200, activation=activation_function, input_shape=(timesteps, input_dim)))
	model.add(RepeatVector(prediction_timesteps))
	model.add(LSTM(200, activation=activation_function, dropout=dropout, return_sequences=True))
	model.add(TimeDistributed(Dense(200,activation=activation_function)))
	model.add(TimeDistributed(Dense(1)))
	model.compile(optimizer=optimizer, loss=loss_function)
	return model

def mlp_swish(params):
	
	# set input and network params
	input_dim = params['data_dim']
	output_dim = params['output_dim']
	hidden_layers = params['hidden_units']
	dense_layers = params['dense_units']
	timesteps = params['timesteps']

	# optimization parameters
	loss_function = params['loss_function']
	optimizer = params['optimizer']
	activation_function = params['activation_function']
	dropout_keep_prob = params['dropout_keep_prob']
		
	# this is the size of our encoded representations (compress informatino to 2000 points - 10% of original vector)
	encoding_dim = 400

	# this is our input placeholder
	input_window = Input(shape=(input_dim,))
	encoded = Dense(2000, activation = swish, kernel_initializer = 'uniform')(input_window)
	encoded = Dense(400, activation = swish, kernel_initializer = 'uniform')(encoded)
	decoded = Dense(800, activation = swish, kernel_initializer = 'uniform')(encoded)
	decoded = Dense(2400, activation = swish, kernel_initializer = 'uniform')(decoded)
	
	# this model maps an input to its reconstruction
	autoencoder = Model(input_window, decoded)
	encoder = Model(input_window, encoded)
	# create the decoder model
	#decoder = Model(encoded_input, decoder)
	autoencoder.compile(optimizer=optimizer, loss=loss_function, metrics = ['accuracy'])
	return autoencoder, encoder#, decoder

def autoencoder_relu(params):
	
	# set input and network params
	input_dim = params['data_dim']
	#output_dim = params['output_dim']
	hidden_layers = params['hidden_units']
	dense_layers = params['dense_units']
	timesteps = params['timesteps']

	# optimization parameters
	loss_function = params['loss_function']
	optimizer = params['optimizer']
	activation_function = params['activation_function']
	dropout_keep_prob = params['dropout_keep_prob']
		
	# this is the size of our encoded representations (compress informatino to 2000 points - 10% of original vector)
	encoding_dim = 400

	# this is our input placeholder
	input_window = Input(shape=(input_dim,))
	encoded = Dense(2000, activation=activation_function)(input_window)	
	encoded = Dense(800, activation=activation_function)(encoded)
	encoded = Dense(400, activation=activation_function)(encoded)
	decoded = ReLU(threshold=0.05)(encoded)	
	decoded = Dense(400)(encoded)
	decoded = ReLU(threshold=0.05)(decoded)	
	decoded = Dense(400)(encoded)
	decoded = ReLU(threshold=0.05)(decoded)
	decoded = Dense(400, activation=activation_function)(decoded)
	
	# this model maps an input to its reconstruction
	autoencoder = Model(input_window, decoded)
	encoder = Model(input_window, encoded)
	# create the decoder model
	#decoder = Model(encoded_input, decoder)
	autoencoder.compile(optimizer=optimizer, loss=loss_function, metrics = ['accuracy'])
	return autoencoder, encoder#, decoder

def autoencoder(params):
	
	# set input and network params
	input_dim = params['data_dim']
	#output_dim = params['output_dim']
	hidden_layers = params['hidden_units']
	dense_layers = params['dense_units']
	timesteps = params['timesteps']

	# optimization parameters
	loss_function = params['loss_function']
	optimizer = params['optimizer']
	activation_function = params['activation_function']
	dropout_keep_prob = params['dropout_keep_prob']
		
	# this is the size of our encoded representations (compress informatino to 2000 points - 10% of original vector)
	encoding_dim = 100

	# this is our input placeholder
	input_window = Input(shape=(input_dim,))
	encoded = Dense(1600, activation=activation_function)(input_window)
	encoded = Dropout(dropout_keep_prob)(encoded)
	encoded = Dense(1600, activation=activation_function)(encoded)
	encoded = Dropout(dropout_keep_prob)(encoded)
	encoded = Dense(200, activation=activation_function)(encoded)
	
	#auto-decoder
	decoded_1 = Dense(200, activation=activation_function)(encoded)
	decoded_1 = Dropout(dropout_keep_prob)(decoded_1)
	decoded_2 = Dense(1600, activation=activation_function)(decoded_1)
	decoded_2 = Dropout(dropout_keep_prob)(decoded_2)
	decoded_3 = Dense(800, activation=activation_function)(decoded_2)
	outputs = [decoded_3]
	#outputs = [decoded,predicted]
	
	# this model maps an input to its reconstruction
	model = Model(input_window, outputs=outputs)
	encoder = Model(input_window, encoded)
	
	input_decoder = Input(shape=(200,))
	decoder_layers = model.layers[6:11]
	decoder = input_decoder
	for decoder_layer in decoder_layers:
		decoder = decoder_layer(decoder)
	# decoder
	decoder = Model(input_decoder, decoder)
	# create the decoder model
	#decoder = Model(encoded_input, decoder)
	model.compile(optimizer=optimizer, loss='mse', metrics = ['accuracy'])
	model.summary()
	encoder.summary()
	decoder.summary()
	return model, encoder, decoder

def decoder_envelope(params):
	
	# set input and network params
	input_dim = params['data_dim']
	#output_dim = params['output_dim']
	hidden_layers = params['hidden_units']
	dense_layers = params['dense_units']
	timesteps = params['timesteps']

	# optimization parameters
	loss_function = params['loss_function']
	optimizer = params['optimizer']
	activation_function = params['activation_function']
	dropout_keep_prob = params['dropout_keep_prob']
		
	# this is the size of our encoded representations (compress informatino to 2000 points - 10% of original vector)
	encoding_dim = 800

	# this is our input placeholder
	input_window = Input(shape=(encoding_dim,))
	decoded = Dense(800, activation=activation_function)(input_window)
	#decoded = Dropout(dropout_keep_prob)(decoded)
	decoded = Dense(800, activation=activation_function)(decoded)
	#decoded = Dropout(dropout_keep_prob)(decoded)
	decoded = Dense(400)(decoded)
	outputs = [decoded]
	
	# this model maps an input to its reconstruction
	model = Model(input_window, outputs=outputs)
	# create the decoder model
	#decoder = Model(encoded_input, decoder)
	model.compile(optimizer=optimizer, loss=loss_function, metrics = ['accuracy'])
	model.summary()
	return model#, decoder

def decoder_multiply(params):
	
	# set input and network params
	input_dim = params['data_dim']
	#output_dim = params['output_dim']
	hidden_layers = params['hidden_units']
	dense_layers = params['dense_units']
	timesteps = params['timesteps']

	# optimization parameters
	loss_function = params['loss_function']
	optimizer = params['optimizer']
	activation_function = params['activation_function']
	dropout_keep_prob = params['dropout_keep_prob']
		
	# this is the size of our encoded representations (compress informatino to 2000 points - 10% of original vector)
	encoding_dim = 800

	# this is our input placeholder
	input_window = Input(shape=(encoding_dim,))
	decoded = Dense(800, activation=activation_function)(input_window)	
	#decoded = Dropout(dropout_keep_prob)(decoded)
	decoded_2 = Dense(800, activation=activation_function)(decoded)
	#decoded_2 = Dropout(dropout_keep_prob)(decoded_2)
	multiply_outputs = Multiply()([decoded_2, decoded])
	multiply_outputs_2 = Multiply()([multiply_outputs, decoded])	
	multiply_outputs_3 = Multiply()([multiply_outputs_2, decoded])	
	multiply_outputs_4 = Multiply()([multiply_outputs_3, decoded])	
	multiply_outputs_5 = Multiply()([multiply_outputs_4, decoded])
	add_outputs = Concatenate()([decoded,multiply_outputs,multiply_outputs_2,multiply_outputs_3,multiply_outputs_4,multiply_outputs_5])
	decoded_6 = Dense(100, activation=activation_function, input_shape=(6*encoding_dim,))(add_outputs)
	#decoded_6 = Dense(100, activation=activation_function)(decoded_6)

	outputs = [decoded_6]
	#outputs = [decoded,predicted]
	
	# this model maps an input to its reconstruction
	model = Model(input_window, outputs=outputs)
	# create the decoder model
	#decoder = Model(encoded_input, decoder)
	model.compile(optimizer=optimizer, loss=loss_function, metrics = ['accuracy'])
	model.summary()
	return model#, decoder

def decoder_mlp_operations(params):
	
	# set input and network params
	input_dim = params['data_dim']
	#output_dim = params['output_dim']
	hidden_layers = params['hidden_units']
	dense_layers = params['dense_units']
	timesteps = params['timesteps']

	# optimization parameters
	optimizer = params['optimizer']
	activation_function = params['activation_function']
	dropout_keep_prob = params['dropout_keep_prob']
		
	# this is the size of our encoded representations (compress informatino to 2000 points - 10% of original vector)
	encoding_dim = 800

	# this is our input placeholder
	input_window = Input(shape=(encoding_dim,))
	decoded = Dense(800, activation=activation_function)(input_window)	
	decoded = Dropout(dropout_keep_prob)(decoded)
	decoded_0 = Dense(400, activation=activation_function)(decoded)
	decoded_1 = Dense(400, activation=activation_function)(decoded)
	decoded_2 = Add()([decoded_0,decoded_1])
	decoded_3 = Dense(400, activation=activation_function)(decoded)
	decoded_4 = Dense(400, activation=activation_function)(decoded)
	decoded_5 = Subtract()([decoded_3,decoded_4])
	decoded_6 = Dense(400, activation=activation_function)(decoded)
	decoded_7 = Dense(400, activation=activation_function)(decoded)
	decoded_8 = Multiply()([decoded_6,decoded_7])
	decoded_9 = Add()([decoded_2,decoded_5,decoded_8])
	decoded_10 = Dense(400,activation=activation_function)(decoded_9)

	outputs = [decoded_10]
	#outputs = [decoded,predicted]
	
	# this model maps an input to its reconstruction
	model = Model(input_window, outputs=outputs)
	# create the decoder model
	#decoder = Model(encoded_input, decoder)
	model.compile(optimizer=optimizer, loss=emg_error, metrics = ['accuracy'])
	model.summary()
	return model#, decoder

def simple_mlp(params):
    
    # set input and network params
    input_dim = params['data_dim']
    hidden_layers = params['hidden_units']
    dense_layers = params['dense_units']
    timesteps = params['timesteps']
    
    # regularization params
    dropout_keep_prob = params['dropout_keep_prob']
    regularization_l1 = params['regularization_l1']
    regularization_l2 = params['regularization_l2']

    # optimization parameters
    optimizer = params['optimizer']
    activation_function = params['activation_function']
    
    if activation_function == 'leakyrelu':
        leaky_alpha =  params['leaky_alpha']
        activation = LeakyReLU(alpha=leaky_alpha)
        activation.__name__ = 'LeakyReLu'
    else:
        activation = activation_function
    
    loss_function = params['loss_function']
        
    # this is our input placeholder
    input_window = Input(shape=(timesteps,))
    #if dropout_keep_prob > 0.0:
    ##   output = Dropout(dropout_keep_prob)(input_window)
    #else:
    output = input_window
    output = Dense(dense_layers[0], activation=activation, input_shape=(timesteps,), kernel_regularizer=regularizers.l1_l2(l1=regularization_l1, l2=regularization_l2))(output)
    
    for i in range(len(dense_layers)-1):
        if dropout_keep_prob > 0.0:
            output = Dropout(dropout_keep_prob)(output)
        output = Dense(dense_layers[i+1], activation=activation, kernel_regularizer=regularizers.l1_l2(l1=regularization_l1, l2=regularization_l2))(output)

    outputs = [output]
    
    model = Model(input_window, outputs=outputs)

    model.compile(optimizer=optimizer, loss=loss_function, metrics = ['accuracy',rmse])
    model.summary()
    return model

def decoder_mlp_mae(params):
	
	# set input and network params
	input_dim = params['data_dim']
	#output_dim = params['output_dim']
	hidden_layers = params['hidden_units']
	dense_layers = params['dense_units']
	timesteps = params['timesteps']

	# optimization parameters
	optimizer = params['optimizer']
	activation_function = params['activation_function']
	dropout_keep_prob = params['dropout_keep_prob']
		
	# this is the size of our encoded representations (compress informatino to 2000 points - 10% of original vector)
	encoding_dim = 4000

	# this is our input placeholder
	input_window = Input(shape=(encoding_dim,))
	decoded = Dense(4000, activation=activation_function)(input_window)
	decoded = Dense(2000, activation=activation_function)(decoded)
	decoded = Dense(1000, activation=activation_function)(decoded)
	decoded = Dense(800, activation=activation_function)(decoded)
	decoded = Dense(600, activation=activation_function)(decoded)
	decoded = Dense(400, activation=activation_function)(decoded)

	outputs = [decoded]
	#outputs = [decoded,predicted]
	
	# this model maps an input to its reconstruction
	model = Model(input_window, outputs=outputs)
	# create the decoder model
	#decoder = Model(encoded_input, decoder)
	model.compile(optimizer=optimizer, loss=emg_error, metrics = ['accuracy'])
	model.summary()
	return model#, decoder

def autoencoder_operations(params):
	
	# set input and network params
	input_dim = params['data_dim']
	#output_dim = params['output_dim']
	hidden_layers = params['hidden_units']
	dense_layers = params['dense_units']
	timesteps = params['timesteps']

	# optimization parameters
	loss_function = params['loss_function']
	optimizer = params['optimizer']
	activation_function = params['activation_function']
	dropout_keep_prob = params['dropout_keep_prob']
		
	# this is the size of our encoded representations (compress informatino to 2000 points - 10% of original vector)
	encoding_dim = 400

	# this is our input placeholder
	input_window = Input(shape=(input_dim,))
	encoded = Dense(2000, activation=activation_function)(input_window)	
	encoded = Dense(1000, activation=activation_function)(encoded)
	encoded = Dense(400, activation=activation_function)(encoded)
	decoded_0 = Dense(400, activation=activation_function)(encoded)
	#decoded_0 = Dropout(dropout_keep_prob)(decoded_0)
	decoded_1 = Dense(400, activation=activation_function)(encoded)
	#decoded_1 = Dropout(dropout_keep_prob)(decoded_1)
	sub_outputs = Subtract()([decoded_0, decoded_1])
	decoded_2 = Dense(400, activation=activation_function)(encoded)
	#decoded_2 = Dropout(dropout_keep_prob)(decoded_2)
	decoded_3 = Dense(400, activation=activation_function)(encoded)
	#decoded_3 = Dropout(dropout_keep_prob)(decoded_3)
	add_outputs = Add()([decoded_2, decoded_3])
	decoded_4 = Dense(400, activation=activation_function)(encoded)
	#decoded_4 = Dropout(dropout_keep_prob)(decoded_1)
	decoded_5 = Dense(400, activation=activation_function)(encoded)
	#decoded_5 = Dropout(dropout_keep_prob)(decoded_2)	
	multiply_outputs = Multiply()([decoded_4, decoded_5])
	last_outputs = [sub_outputs,add_outputs,multiply_outputs]
	
	# this model maps an input to its reconstruction
	autoencoder = Model(input_window, decoded_6)
	encoder = Model(input_window, encoded)
	# create the decoder model
	#decoder = Model(encoded_input, decoder)
	autoencoder.compile(optimizer=optimizer, loss=loss_function, metrics = ['accuracy'])
	return autoencoder, encoder#, decoder

def autoencoder_inject(params):
	
	# set input and network params
	input_dim = params['data_dim']
	#output_dim = params['output_dim']
	hidden_layers = params['hidden_units']
	dense_layers = params['dense_units']
	timesteps = params['timesteps']

	# optimization parameters
	loss_function = params['loss_function']
	optimizer = params['optimizer']
	activation_function = params['activation_function']
	dropout_keep_prob = params['dropout_keep_prob']
		
	# this is the size of our encoded representations (compress informatino to 2000 points - 10% of original vector)
	encoding_dim = 400

	# this is our input placeholder
	input_window = Input(shape=(input_dim,))
	encoded = Dense(2000, activation=activation_function)(input_window)	
	encoded = Dense(1000, activation=activation_function)(encoded)
	encoded = Dense(400, activation=activation_function)(encoded)
	decoded_0 = Dense(400, activation=activation_function)(encoded)
	decoded_0 = Add()([decoded_0,encoded])
	decoded_0 = Dropout(dropout_keep_prob)(decoded_0)
	decoded_1 = Dense(400, activation=activation_function)(decoded_0)
	decoded_1 = Add()([decoded_1,encoded])
	decoded_1 = Dropout(dropout_keep_prob)(decoded_1)
	decoded_2 = Dense(400, activation=activation_function)(decoded_1)
	decoded_2 = Add()([decoded_2,encoded])
	decoded_2 = Dropout(dropout_keep_prob)(decoded_2)
	decoded_3 = Dense(400, activation=activation_function)(decoded_2)
	
	# this model maps an input to its reconstruction
	autoencoder = Model(input_window, decoded_3)
	encoder = Model(input_window, encoded)
	# create the decoder model
	#decoder = Model(encoded_input, decoder)
	autoencoder.compile(optimizer=optimizer, loss=loss_function, metrics = ['accuracy'])
	return autoencoder, encoder#, decoder

def cnn_lstm(params):

	# define the model - # multivariate multi-step encoder-decoder cnn lstm
	# set input and network params
	input_dim = params['data_dim']
	hidden_layers = params['hidden_units']
	dense_layers = params['dense_units']
	timesteps = params['timesteps']

	# convolution parameters
	convolution_filters = params['convolution_filters']	
	convolution_activation_function = params['convolution_activation_function']
	kernel = params['kernel']	
	pooling = params['pooling']

	# optimization parameters
	loss_function = params['loss_function']
	optimizer = params['optimizer']
	activation_function = params['activation_function']
	dropout = params['dropout_keep_prob']

	# input shape for CNN

	input_window = Input(shape=(timesteps,input_dim))

	# define CNN model
	# (batch, steps, channels) => (batch, new_steps, filters)
	# Conv1D => Apply #filters filters with #kernel_size (time window) and later apply activation function
	#Conv1
	cnn = Conv1D(filters=convolution_filters[0], kernel_size=kernel[0], activation=convolution_activation_function, padding='causal', data_format='channels_last', dilation_rate=1)(input_window)
	#MaxPooling 1 (4000 => 2000)
	cnn = MaxPooling1D(pool_size=pooling)(cnn)
	#Conv2
	cnn = Conv1D(filters=convolution_filters[1], kernel_size=kernel[1], activation=convolution_activation_function, padding='causal', data_format='channels_last', dilation_rate=1)(cnn)
	#MaxPooling 2 (2000 => 1000)
	cnn = MaxPooling1D(pool_size=pooling)(cnn)
	#Conv3
	cnn = Conv1D(filters=convolution_filters[2], kernel_size=kernel[2], activation=convolution_activation_function, padding='causal', data_format='channels_last', dilation_rate=1)(cnn)
	#MaxPooling 3 (1000 => 500)
	cnn = MaxPooling1D(pool_size=pooling)(cnn)
	#Conv4
	cnn = Conv1D(filters=convolution_filters[3], kernel_size=kernel[3], activation=convolution_activation_function, padding='causal', data_format='channels_last', dilation_rate=1)(cnn)
	
	# define LSTM model
	# define lstm model
	#2000 => 400 => 400
	lstm = Bidirectional(LSTM(hidden_layers[0], activation=activation_function, dropout=dropout, return_sequences=True))(cnn)
	lstm = Bidirectional(LSTM(hidden_layers[1], activation=activation_function, dropout=dropout))(lstm)
	lstm = Dense(dense_layers[0], activation=activation_function)(lstm)
	
	model = Model(input_window, lstm)
	
	model.compile(loss=loss_function, optimizer=optimizer)
	model.summary()
	
	encoder = Model(input_window, cnn)
	encoder.summary()

	return model, encoder

def cnn_lstm_mlp(params):

	# define the model - # multivariate multi-step encoder-decoder cnn lstm
	# set input and network params
	input_dim = params['data_dim']
	hidden_layers = params['hidden_units']
	dense_layers = params['dense_units']
	timesteps = params['timesteps']

	# convolution parameters
	convolution_filters = params['convolution_filters']	
	convolution_activation_function = params['convolution_activation_function']
	kernel = params['kernel']	
	pooling = params['pooling']

	# optimization parameters
	loss_function = params['loss_function']
	optimizer = params['optimizer']
	activation_function = params['activation_function']
	dropout = params['dropout_keep_prob']

	# input shape for CNN

	input_window = Input(shape=(timesteps,input_dim))

	# define CNN model
	# (batch, steps, channels) => (batch, new_steps, filters)
	# Conv1D => Apply #filters filters with #kernel_size (time window) and later apply activation function
	#Conv1
	cnn = Conv1D(filters=convolution_filters[0], kernel_size=kernel[0], activation=convolution_activation_function, padding='causal', data_format='channels_last', dilation_rate=1)(input_window)
	#MaxPooling 1 (4000 => 2000)
	cnn = MaxPooling1D(pool_size=pooling)(cnn)
	#Conv2
	cnn = Conv1D(filters=convolution_filters[1], kernel_size=kernel[1], activation=convolution_activation_function, padding='causal', data_format='channels_last', dilation_rate=1)(cnn)
	#MaxPooling 2 (2000 => 1000)
	cnn = MaxPooling1D(pool_size=pooling)(cnn)
	#Conv3
	cnn = Conv1D(filters=convolution_filters[2], kernel_size=kernel[2], activation=convolution_activation_function, padding='causal', data_format='channels_last', dilation_rate=1)(cnn)
	#MaxPooling 3 (1000 => 500)
	cnn = MaxPooling1D(pool_size=pooling)(cnn)
	#Conv4
	cnn = Conv1D(filters=convolution_filters[3], kernel_size=kernel[3], activation=convolution_activation_function, padding='valid', data_format='channels_last', dilation_rate=1)(cnn)
	cnn = MaxPooling1D(pool_size=pooling)(cnn)
	cnn = Flatten()(cnn)
	cnn = Dense(dense_layers[0], activation=convolution_activation_function)(cnn)
	
	# define LSTM model
	# define lstm model
	#2000 => 400 => 400
	lstm = Bidirectional(LSTM(hidden_layers[0], activation=activation_function, dropout=dropout, return_sequences=True))(input_window)
	lstm = Bidirectional(LSTM(hidden_layers[1], activation=activation_function, dropout=dropout))(lstm)
	lstm = Dense(dense_layers[0], activation=activation_function)(lstm)

	#Include a last MLP Dense layer for combining the results
	merge = Concatenate(axis=1)([lstm,cnn])
	mlp = Dense(dense_layers[0], activation=activation_function)(merge)
	
	model = Model(input_window, mlp)
	
	model.compile(loss=loss_function, optimizer=optimizer)
	model.summary()
	
	encoder = Model(input_window, cnn)
	encoder.summary()

	return model, encoder

def cnn_mlp(params):

	# set input and network params
	input_dim = params['data_dim']
	hidden_layers = params['hidden_units']
	dense_layers = params['dense_units']
	timesteps = params['timesteps']

	# convolution parameters
	convolution_filters = params['convolution_filters']	
	convolution_activation_function = params['convolution_activation_function']
	kernel = params['kernel']	
	pooling = params['pooling']

	# optimization parameters
	loss_function = params['loss_function']
	optimizer = params['optimizer']
	activation_function = params['activation_function']
	dropout = params['dropout_keep_prob']

	# input shape for CNN

	input_window = Input(shape=(timesteps,input_dim))

	# define CNN model
	# (batch, steps, channels) => (batch, new_steps, filters)
	# Conv1D => Apply #filters filters with #kernel_size (time window) and later apply activation function
	encoded = Conv1D(filters=convolution_filters[0], kernel_size=kernel[0], activation=convolution_activation_function, padding='causal', data_format='channels_last', dilation_rate=1)(input_window)
	encoded = Conv1D(filters=convolution_filters[1], kernel_size=kernel[1], activation=convolution_activation_function, padding='causal', data_format='channels_last', dilation_rate=1)(encoded)
	encoded = Conv1D(filters=convolution_filters[2], kernel_size=kernel[2], activation=convolution_activation_function, padding='causal', data_format='channels_last', dilation_rate=1)(encoded)
	encoded = Conv1D(filters=convolution_filters[3], kernel_size=kernel[3], activation=convolution_activation_function, padding='causal', data_format='channels_last', dilation_rate=1)(encoded)
	encoded = Conv1D(filters=convolution_filters[4], kernel_size=kernel[4], activation=convolution_activation_function, padding='causal', data_format='channels_last', dilation_rate=1)(encoded)
	encoded = Conv1D(filters=convolution_filters[5], kernel_size=kernel[5], activation=convolution_activation_function, padding='causal', data_format='channels_last', dilation_rate=1)(encoded)
	encoded = MaxPooling1D(pool_size=pooling)(encoded)
	encoded = Flatten()(encoded)
	encoded = Dense(dense_layers[2], activation=convolution_activation_function)(encoded)	
	encoded = Dense(dense_layers[2])(encoded)
	#(batch, steps, filters) => (batch, steps, filters)
	#decoded = Flatten()(encoded)

	#4000 => 2000 => 1000 => 400
	# define MLP model
	mlp = Flatten()(input_window) #-- if using more than 1 input_dim
	mlp = Dense(dense_layers[0], activation=activation_function)(mlp)
	#mlp = Dropout(dropout=dropout)(mlp)
	mlp = Dense(dense_layers[1], activation=activation_function)(mlp)
	#mlp = Dropout(dropout=dropout)(mlp)
	mlp = Dense(dense_layers[2], activation=activation_function)(mlp)

	merged = Concatenate(axis=1)([encoded,mlp])
	output_layer = Dense(dense_layers[3], activation=activation_function)(merged)

	model = Model(input_window, output_layer)
	
	model.compile(loss=emg_error, optimizer=optimizer)
	model.summary()
	
	encoder = Model(input_window, encoded)
	encoder.summary()

	return model, encoder

def resnet(params):

	# set input and network params
	input_dim = params['data_dim']
	hidden_layers = params['hidden_units']
	dense_layers = params['dense_units']
	timesteps = params['timesteps']
	prediction_timesteps = params['prediction_timesteps']

	# convolution parameters
	convolution_filters = params['convolution_filters']	
	convolution_activation_function = params['convolution_activation_function']
	kernel = params['kernel']	
	pooling = params['pooling']

	# optimization parameters
	loss_function = params['loss_function']
	optimizer = params['optimizer']
	activation_function = params['activation_function']
	dropout = params['dropout_keep_prob']
	regularization_l1 = params['regularization_l1']
	regularization_l2 = params['regularization_l2']

	regularizer = l1_l2(l1=regularization_l1, l2=regularization_l2)
	
	input_window = Input(shape=(timesteps,input_dim))

	# BLOCK 1 	
	conv_x = Conv1D(filters=convolution_filters[0], activity_regularizer=regularizer, kernel_size=kernel[0], data_format='channels_last', padding='same')(input_window)
	conv_x = BatchNormalization()(conv_x)
	conv_x = Activation(convolution_activation_function)(conv_x)
	
	conv_y = Conv1D(filters=convolution_filters[0], activity_regularizer=regularizer, kernel_size=kernel[1], data_format='channels_last', padding='same')(conv_x)
	conv_y = BatchNormalization()(conv_y)
	conv_y = Activation(convolution_activation_function)(conv_y)
	
	conv_z = Conv1D(filters=convolution_filters[0], activity_regularizer=regularizer, kernel_size=kernel[2], data_format='channels_last', padding='same')(conv_y)
	conv_z = BatchNormalization()(conv_z)
	
	# expand channels for the sum
	shortcut_y = Conv1D(filters=convolution_filters[0], activity_regularizer=regularizer, kernel_size=kernel[3], data_format='channels_last', padding='same')(input_window)
	shortcut_y = BatchNormalization()(shortcut_y)
	
	output_block_1 = Add()([shortcut_y, conv_z])
	output_block_1 = Activation(convolution_activation_function)(output_block_1)
	
	# BLOCK 2 
	conv_x = Conv1D(filters=convolution_filters[1], activity_regularizer=regularizer, kernel_size=kernel[0], data_format='channels_last', padding='same')(output_block_1)
	conv_x = BatchNormalization()(conv_x)
	conv_x = Activation(convolution_activation_function)(conv_x)
	
	conv_y = Conv1D(filters=convolution_filters[1], activity_regularizer=regularizer, kernel_size=kernel[1], data_format='channels_last', padding='same')(conv_x)
	conv_y = BatchNormalization()(conv_y)
	conv_y = Activation(convolution_activation_function)(conv_y)
	
	conv_z = Conv1D(filters=convolution_filters[1], activity_regularizer=regularizer, kernel_size=kernel[2], data_format='channels_last', padding='same')(conv_y)
	conv_z = BatchNormalization()(conv_z)
	
	# expand channels for the sum 
	shortcut_y = Conv1D(filters=convolution_filters[1], activity_regularizer=regularizer, kernel_size=kernel[3], data_format='channels_last', padding='same')(output_block_1)
	shortcut_y = BatchNormalization()(shortcut_y)
	
	output_block_2 = Add()([shortcut_y, conv_z])
	output_block_2 = Activation(convolution_activation_function)(output_block_2)
	
	# BLOCK 3 
	conv_x = Conv1D(filters=convolution_filters[2], activity_regularizer=regularizer, kernel_size=kernel[0], data_format='channels_last', padding='same')(output_block_2)
	conv_x = BatchNormalization()(conv_x)
	conv_x = Activation(convolution_activation_function)(conv_x)
	
	conv_y = Conv1D(filters=convolution_filters[2], activity_regularizer=regularizer, kernel_size=kernel[1], data_format='channels_last', padding='same')(conv_x)
	conv_y = BatchNormalization()(conv_y)
	conv_y = Activation(convolution_activation_function)(conv_y)
	
	conv_z = Conv1D(filters=convolution_filters[2], activity_regularizer=regularizer, kernel_size=kernel[2], data_format='channels_last', padding='same')(conv_y)
	conv_z = BatchNormalization()(conv_z)
	
	# no need to expand channels because they are equal
	shortcut_y = Conv1D(filters=convolution_filters[2], activity_regularizer=regularizer, kernel_size=kernel[3], data_format='channels_last', padding='same')(output_block_2)
	shortcut_y = BatchNormalization()(shortcut_y)
	
	output_block_3 = Add()([shortcut_y, conv_z])
	output_block_3 = Activation(convolution_activation_function)(output_block_3)

	gap_layer = GlobalMaxPooling1D()(output_block_3)
	
	# FINAL 
	output_layer = Dense(prediction_timesteps, activation=activation_function)(gap_layer)

	model = Model(input_window, output_layer)
	
	model.compile(loss=emg_error, optimizer=optimizer, metrics=['mae', 'acc'])
	model.summary()

	return model

def cnn_encoder(params):
	# set input and network params
	input_dim = params['data_dim']
	hidden_layers = params['hidden_units']
	dense_layers = params['dense_units']
	timesteps = params['timesteps']

	# convolution parameters
	convolution_filters = params['convolution_filters']	
	convolution_activation_function = params['convolution_activation_function']
	kernel = params['kernel']	
	pooling = params['pooling']

	# optimization parameters
	loss_function = params['loss_function']
	optimizer = params['optimizer']
	activation_function = params['activation_function']
	dropout = params['dropout_keep_prob']

	# input shape for CNN

	input_window = Input(shape=(timesteps,input_dim))

	# define CNN model
	# (batch, steps, channels) => (batch, new_steps, filters)
	# Conv1D => Apply #filters filters with #kernel_size (time window) and later apply activation function
	encoded = Conv1D(filters=convolution_filters[0], kernel_size=kernel[0], activation=convolution_activation_function, padding='causal', data_format='channels_last', dilation_rate=1)(input_window)
	encoded = MaxPooling1D(pool_size=pooling)(encoded)
	encoded = Conv1D(filters=convolution_filters[1], kernel_size=kernel[1], activation=convolution_activation_function, padding='causal', data_format='channels_last', dilation_rate=1)(encoded)
	encoded = MaxPooling1D(pool_size=pooling)(encoded)
	encoded = Conv1D(filters=convolution_filters[2], kernel_size=kernel[2], activation=convolution_activation_function, padding='causal', data_format='channels_last', dilation_rate=1)(encoded)
	encoded = MaxPooling1D(pool_size=pooling)(encoded)
	encoded = Conv1D(filters=convolution_filters[3], kernel_size=kernel[3], activation=convolution_activation_function, padding='causal', data_format='channels_last', dilation_rate=1)(encoded)
	encoded = MaxPooling1D(pool_size=pooling)(encoded)
	encoded = Conv1D(filters=convolution_filters[4], kernel_size=kernel[4], activation=convolution_activation_function, padding='causal', data_format='channels_last', dilation_rate=1)(encoded)
	encoded = MaxPooling1D(pool_size=pooling)(encoded)
	encoded = Conv1D(filters=convolution_filters[5], kernel_size=kernel[5], activation=convolution_activation_function, padding='causal', data_format='channels_last', dilation_rate=1)(encoded)
	encoded = MaxPooling1D(pool_size=pooling)(encoded)
	encoded = Flatten()(encoded)
	decoded = Dense(dense_layers[0], activation=convolution_activation_function)(encoded)
	decoded = Dense(dense_layers[1], activation=activation_function)(decoded)
	decoded = Dense(dense_layers[2], activation=activation_function)(decoded)
	decoded = Dense(dense_layers[3], activation=activation_function)(decoded)
	model = Model(input_window, decoded)
	
	model.compile(loss=loss_function, optimizer=optimizer)
	model.summary()
	return model

def cnn_autoencoder(params):
	# set input and network params
	input_dim = params['data_dim']
	hidden_layers = params['hidden_units']
	dense_layers = params['dense_units']
	timesteps = params['timesteps']

	# convolution parameters
	convolution_filters = params['convolution_filters']	
	convolution_activation_function = params['convolution_activation_function']
	kernel = params['kernel']	
	pooling = params['pooling']

	# optimization parameters
	loss_function = params['loss_function']
	optimizer = params['optimizer']
	activation_function = params['activation_function']
	dropout = params['dropout_keep_prob']

	# input shape for CNN

	input_window = Input(shape=(timesteps,input_dim))

	# define CNN model
	# (batch, steps, channels) => (batch, new_steps, filters)
	# Conv1D => Apply #filters filters with #kernel_size (time window) and later apply activation function
	encoded = Conv1D(filters=convolution_filters[0], kernel_size=kernel[0], activation=convolution_activation_function, padding='causal', data_format='channels_last', dilation_rate=1)(input_window)
	encoded = MaxPooling1D(pool_size=pooling)(encoded)
	encoded = Conv1D(filters=convolution_filters[1], kernel_size=kernel[1], activation=convolution_activation_function, padding='causal', data_format='channels_last', dilation_rate=1)(encoded)
	encoded = MaxPooling1D(pool_size=pooling)(encoded)
	
	decoded = Conv1D(filters=convolution_filters[1], kernel_size=kernel[1], activation=convolution_activation_function, padding='causal', data_format='channels_last', dilation_rate=1)(encoded)
	decoded = UpSampling1D(size=pooling)(decoded)
	decoded = Conv1D(filters=convolution_filters[0], kernel_size=kernel[0], activation=convolution_activation_function, padding='causal', data_format='channels_last', dilation_rate=1)(decoded)
	decoded = UpSampling1D(size=pooling)(decoded)
	decoded = Conv1D(filters=1, kernel_size=kernel[0], activation=activation_function, padding='causal', data_format='channels_last', dilation_rate=1)(decoded)
	model = Model(input_window, decoded)	
	model.compile(loss=loss_function, optimizer=optimizer)
	model.summary()

	encoder = Model(input_window, encoded)

	return model, encoder

# define the model - # univariate multi-step encoder-decoder convlstm
def convlstm(params):

	# set input and network params
	input_dim = params['data_dim']
	output_dim = params['output_dim']
	hidden_layers = params['hidden_units']
	dense_layers = params['dense_units']
	timesteps = params['timesteps']

	# convolution parameters
	convolution_filters = params['convolution_filters']	
	convolution_activation_function = params['convolution_activation_function']
	kernel = params['kernel']
	conv_rows = params['conv_rows']
	conv_timesteps = params['conv_timesteps']
	n_timesteps = params['n_timesteps']

	# optimization parameters
	loss_function = params['loss_function']
	optimizer = params['optimizer']
	activation_function = params['activation_function']
	dropout = params['dropout_keep_prob']

	# input shape for ConvLSTM
	#samples, time, rows, cols, channels

	
	input_window = Input(shape=(n_timesteps, conv_rows, conv_timesteps, input_dim))
	
	encoded = ConvLSTM2D(filters=convolution_filters, kernel_size=(input_dim,kernel), activation=convolution_activation_function)(input_window)
	encoded = Flatten()(encoded)
	decoded = RepeatVector(output_dim)(encoded)
	decoded = LSTM(hidden_layers, activation=activation_function, return_sequences=True)(decoded)
	decoded = TimeDistributed(Dense(dense_layers, activation=activation_function))(decoded)
	decoded = TimeDistributed(Dense(1))(decoded)

	model = Model(input_window, decoded)
	model.compile(loss=loss_function, optimizer=optimizer)
	
	encoder = Model(input_window, encoded)

	return model, encoder


def mlp(params):
	
	# set input and network params
	input_dim = params['data_dim']
	dense_layers = params['dense_units']
	timesteps = params['timesteps']	
	prediction_timesteps = params['prediction_timesteps']

	# convolution parameters
	convolution_filters = params['convolution_filters']	
	convolution_activation_function = params['convolution_activation_function']
	kernel = params['kernel']
	conv_rows = params['conv_rows']
	conv_timesteps = params['conv_timesteps']

	# optimization parameters for MLP
	loss_function = params['loss_function']
	optimizer = params['optimizer']
	activation_function = params['activation_function']
	dropout_keep_prob = params['dropout_keep_prob']

	model = Sequential()
	#model.add(Conv1D(filters=convolution_filters, kernel_size=kernel, activation=convolution_activation_function, input_shape=(timesteps,input_dim)))
	#model.add(MaxPooling1D(pool_size=conv_rows))
	#model.add(Flatten())
	model.add(Dense(dense_layers[0], activation=activation_function, input_shape=(timesteps,)))
	model.add(Dropout(dropout_keep_prob))
	for i in range(len(dense_layers)-2):	
		model.add(Dense(dense_layers[i+1], activation=activation_function))
		model.add(Dropout(dropout_keep_prob))
	model.add(Dense(prediction_timesteps))
	model.compile(loss=loss_function, optimizer=optimizer,metrics=['accuracy'])

	return model

def mlp_parallel(params):
	
	# set input and network params
	input_dim = params['data_dim']
	dense_layers = params['dense_units']
	timesteps = params['timesteps']	
	prediction_timesteps = params['prediction_timesteps']

	# optimization parameters for MLP
	loss_function = params['loss_function']
	optimizer = params['optimizer']
	activation_function = params['activation_function']
	dropout_keep_prob = params['dropout_keep_prob']

	# input shape for both networks
	input_window = Input(shape=(timesteps,))

	# define MLP model
	mlp_1 = Dense(dense_layers[4], activation=activation_function)(input_window)

	# define MLP model #2
	mlp_2 = Dense(dense_layers[0], activation=activation_function)(input_window)
	mlp_2 = Dense(dense_layers[4], activation=activation_function)(mlp_2)

	# define MLP model #3
	mlp_3 = Dense(dense_layers[0], activation=activation_function)(input_window)
	mlp_3 = Dense(dense_layers[1], activation=activation_function)(mlp_3)
	mlp_3 = Dense(dense_layers[4], activation=activation_function)(mlp_3)

	# define MLP model #4
	mlp_4 = Dense(dense_layers[0], activation=activation_function)(input_window)
	mlp_4 = Dense(dense_layers[1], activation=activation_function)(mlp_4)
	mlp_4 = Dense(dense_layers[2], activation=activation_function)(mlp_4)
	mlp_4 = Dense(dense_layers[4], activation=activation_function)(mlp_4)

	concatenated = [mlp_1,mlp_2,mlp_3,mlp_4]
	merge = Concatenate(axis=1)(concatenated)
	merge = Dense(dense_layers[4], activation=activation_function)(merge)
	merge = Dense(dense_layers[4], activation=activation_function)(merge)
	
	model = Model(input_window, merge)

	model.compile(loss=loss_function, optimizer=optimizer,metrics=['mae', 'acc'])	
	model.summary()

	return model

#Implementing pattern according to "Surface EMG Pattern Recognition Using Long Short-Term Memory Combined with Multilayer Perceptron"
def lstm_mlp(params):
	
	# set input and network params
	input_dim = params['data_dim']
	dense_layers = params['dense_units']
	timesteps = params['timesteps']	
	prediction_timesteps = params['prediction_timesteps']

	# optimization parameters for MLP
	loss_function = params['loss_function']
	optimizer = params['optimizer']
	activation_function = params['activation_function']
	dropout_keep_prob = params['dropout_keep_prob']

	# optimization parameters for LSTM	
	hidden_units = params['hidden_units']

	# input shape for both networks
	input_window = Input(shape=(timesteps,input_dim))

	# define MLP model
	mlp = Flatten()(input_window) #-- if using more than 1 input_dim
	#2000 => 1000 => 400
	mlp = Dense(dense_layers[0], activation=activation_function)(mlp)
	mlp = Dense(dense_layers[1], activation=activation_function)(mlp)

	# define lstm model
	#2000 => 400 => 400
	lstm = Bidirectional(LSTM(hidden_units[0], activation=activation_function, dropout=dropout_keep_prob))(input_window)
	lstm = Dense(hidden_units[1], activation=activation_function)(lstm)

	#400 + 400 = 800 => 400
	concatenated = [mlp,lstm]
	merge = Concatenate()(concatenated)
	merge = Dense(dense_layers[2], activation=activation_function)(merge)
	
	model = Model(input_window, merge)
	model.compile(loss=loss_function, optimizer=optimizer)	

	return model

switcher = {
    "rnn_lstm_envelope": rnn_lstm_envelope,
    "rnn_lstm": rnn_lstm,
    "multi_lstm":multi_lstm,
    "bi_lstm":bi_lstm,
    "rnn_gru": rnn_gru,
    "lstm_autoencoder": lstm_autoencoder,
    "lstm_decoder": lstm_decoder,
    "lstm_encoder_simple":lstm_encoder_simple,
    "mlp_swish": mlp_swish,
    "lstm_mlp": lstm_mlp,
    "mlp_parallel":mlp_parallel,
    "mlp":mlp,
    "convlstm":convlstm,
    "cnn_autoencoder":cnn_autoencoder,
    "cnn_encoder":cnn_encoder,
    "cnn_mlp":cnn_mlp,
    "resnet":resnet,
    "cnn_lstm":cnn_lstm,
    "cnn_lstm_mlp":cnn_lstm_mlp,
    "decoder_mlp_operations":decoder_mlp_operations,
    "simple_mlp":simple_mlp
}

#print(switcher)