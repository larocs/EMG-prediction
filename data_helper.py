import numpy as np
import pandas as pd
from pandas import Series
from sklearn.preprocessing import MinMaxScaler

emg_headers = ['TIMESTAMP','FLEXOR_EMG','FLEXOR_EMG_FILTERED','FLEX_EMG_DIFF','EXNTESOR_EMG','EXTENSOR_EMG_FILTERED','EXTENSOR_EMG_DIFF']
acc_headers = ['TIMESTAMP','FLEXOR_ACC_MAG','FLEXOR_ACC_ANG','EXTENSOR_ACC_MAG','EXTENSOR_ACC_ANG']

#Return pandas data frame with filtered time series
def load_timeseries(filename, series):
	
	#Load time series dataset
	loaded_series = pd.read_csv(filename, sep=',', header=0, index_col=0, squeeze=True)
	
	#Applying filter on the selected series
	selected_series = loaded_series.filter(items=series)
	
	return selected_series

def get_windows(data, window_size):	
		
	# Split data into windows
	raw = []
	for index in range(len(data) - window_size):
		raw.append(data[index: index + window_size])
	
	return raw

def get_sampled_data(X,Y,split_ratio):
	
	# Split the input dataset into train and test
	train_x = X[:int(split_ratio), :, :]
	
	train_y = Y[:int(split_ratio), :, :]
		
	#Shuffle all samples before sampling with X and Y#
	perm = np.arange(train_x.shape[0])
	np.random.shuffle(perm)

	# Shuffle x_train, y_train and scalers
	x_train = train_x[perm,:]
	y_train = train_y[perm,:]

	# x_test and y_test, for testing
	x_test = X[int(split_ratio):, :]
	y_test = Y[int(split_ratio):, :]
	
	return [x_train, y_train, x_test, y_test, perm]

def normalize_windows(window_data):
	"""Normalize data"""

	normalized_data = []
	scalers = []
	for window in window_data:
		# Normalize data
		scaler = MinMaxScaler(feature_range=(0, 1),copy=True)
		scaler.fit(window)
		normalized_window = scaler.transform(window)
		normalized_data.append(normalized_window)
		scalers.append(scaler)
	return normalized_data, scalers

def fft_windows(window_data):
	#Transfor the time-series window to Frequency domain
	fft_data = []
	for window in window_data:
		n = len(window_data)
		fft = np.fft.fft(window) #fft/n
		#fft = fft[range(n/2)] #just take the first half of the fft
		fft_abs = np.abs(fft)		
		fft_angle = np.angle(fft)
		fft_data.append(fft_abs)  #just use the abs for now
	return fft_data

def ifft_windows(window_data):
	#Transfor the fft back to time-series window
	ifft_data = []
	for window in window_data:
		n = len(window_data)
		ifft = np.fft.ifft(window) #ifft / n
		#fft = fft[range(n/2)] #just take the first half of the fft
		ifft_abs = np.abs(ifft)
		ifft_data.append(ifft_abs) #just use the abs for now
	return ifft_data

def normalize_windows_with_scalers(window_data, scalers):
	"""Normalize data"""

	normalized_data = []
	for i in range(len(window_data)):
		# Normalize data
		window = window_data[i]
		scaler = scalers[i]
		reshaped = window.reshape(-1,1)
		normalized_window = scaler.transform(reshaped)
		normalized_data.append(normalized_window)
		
	return normalized_data

def normalize(data):
	"""Normalize data"""
	scaler = MinMaxScaler(feature_range=(0, 1),copy=True)
	scaler.fit(data)
	norm_value = scaler.transform(data)
	return [norm_value, scaler]

def min_max(data, min, max):
	"""Normalize data"""
	scaler = MinMaxScaler(feature_range=(min, max),copy=True)
	scaler.fit(data)
	norm_value = scaler.transform(data)
	return [norm_value, scaler]


def abs_roll(data,window_size):
	data_abs = abs(data)
	avg_window = data_abs.rolling(window=window_size)
	rolling_mean = avg_window.mean()
	rolling_mean_shift = rolling_mean.shift(-window_size)
	rolling_mean_shift = rolling_mean_shift.head(data.shape[0]-window_size)
	return rolling_mean_shift