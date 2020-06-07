import numpy as np
from scipy import signal
from keras import backend as K
import tensorflow as tf

def loss_func(params):
    loss_function = params['loss_function']
    if loss_function == 'mse' or loss_function == 'mean_squared_error':
        return mse
    else:
        if loss_function == 'mae' or loss_function == 'mean_absolute_error':
            return mse
        else:
            if loss_function == 'sm_mae' or loss_function == 'huber_loss':
                delta = params['loss_delta']
                return sm_mae_func(delta)
            else:
                if loss_function == 'quan' or loss_function == 'quantile_loss':
                    theta = params['loss_theta']
                    return sm_mae_func(delta)
                else:
                    if loss_function == 'simple':
                        return simple_error
    return simple_error


def evaluate_emg_error(y_true, y_pred):
    m_factor = 20
    delta = 0.05
    abs_tensor = np.abs(y_true - y_pred)
    mask = abs_tensor > delta # boolean tensor, mask[i] = True iff x[i] > 1
    bigger = abs_tensor[mask]
    m = bigger * m_factor
    loss1 = np.mean(m)
    loss2 = np.mean(abs_tensor**2)
    loss = loss1 + loss2 #adds penalty for not seeking higher values
    return loss

def emg_error(y_true, y_pred):
    m_factor = tf.constant(20.0)
    delta = tf.constant(0.05)
    abs_tensor = K.abs(y_true - y_pred)
    mask = tf.greater(abs_tensor, delta) # boolean tensor, mask[i] = True iff x[i] > 1
    bigger = tf.boolean_mask(abs_tensor, mask)
    m = tf.multiply(bigger, m_factor)
    loss1 = K.mean(m)
    loss2 = K.mean(K.square(abs_tensor))
    loss = loss1 + loss2 #adds penalty for not seeking higher values
    return loss

def emg_on_fft(y_true, y_pred):
    delta = tf.constant(5.0)
    y_true_complex = tf.cast(y_true,dtype=tf.complex64)
    y_pred_complex = tf.cast(y_pred,dtype=tf.complex64)
    fft_true = tf.real(tf.fft(y_true_complex))
    fft_pred = tf.real(tf.fft(y_pred_complex))
    mask = tf.greater(fft_true, delta) # boolean tensor, mask[i] = True iff fft[x] > delta
    fft_true = tf.boolean_mask(fft_true, mask)
    fft_pred = tf.boolean_mask(fft_pred, mask)
    loss = K.mean(K.square(fft_true - fft_pred))
    return loss

def emg_plus_fft(y_true, y_pred):
    return loss_fft(y_true, y_pred) + emg_error(y_true, y_pred)

def loss_fft(y_true, y_pred):
    y_true_complex = tf.cast(y_true,dtype=tf.complex64)
    y_pred_complex = tf.cast(y_pred,dtype=tf.complex64)
    fft_true = K.abs(tf.fft(y_true_complex))
    fft_pred = K.abs(tf.fft(y_pred_complex))
    loss = K.mean(K.square(fft_true - fft_pred))
    return loss

def loss_fft_filter(y_true, y_pred):
    filter_low = 2
    filter_high = 20
    y_true_complex = tf.cast(y_true,dtype=tf.complex64)
    y_pred_complex = tf.cast(y_pred,dtype=tf.complex64)
    fft_true = K.abs(tf.fft(y_true_complex))
    fft_pred = K.abs(tf.fft(y_pred_complex))
    abs_tensor = K.square(fft_true[filter_low:filter_high] - fft_pred[filter_low:filter_high])
    loss = K.mean(abs_tensor)
    return loss

def correlation_loss(y_true, y_pred):
    x = y_true
    y = y_pred
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x-mx, y-my
    r_num = K.sum(tf.multiply(xm,ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)
    return 1 - K.square(r)

def moving_average(x, w):
    return bn.move_mean(x, window=w, min_count=1)

def mean_absolute_percentage_error(y_true, y_pred): 
    delta = y_true - y_pred
    delta = np.abs((delta / y_true)*100.0)
    return np.mean(delta)

def mean_percentage_error(y_true, y_pred): 
    #y_true, y_pred = check_arrays(y_true, y_pred)

    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true): 
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)

    return np.mean((y_true - y_pred) / y_true) * 100

def simple_error(true, pred):
    """
    Mean Square Error (MSE/ L2 Loss)
    true: array of true values    
    pred: array of predicted values
    
    returns: mean square error loss
    """
    
    return np.abs(true - pred)

def mse(true, pred):
    """
    Mean Square Error (MSE/ L2 Loss)
    true: array of true values    
    pred: array of predicted values
    
    returns: mean square error loss
    """
    
    return np.sum((true - pred)**2)

def mae(true, pred):
    """
    Mean Absolute Error (MAE/ L1 loss
    true: array of true values    
    pred: array of predicted values
    
    returns: mean absolute error loss
    """
    
    return np.sum(np.abs(true - pred))

def sm_mse_func(delta):
    """
    Smooth Mean Absolute Error/ Huber Loss
    true: array of true values    
    pred: array of predicted values
    
    returns: smoothed mean absolute error loss
    """
    def sm_mae(true, pred):
        """
        Smooth Mean Absolute Error/ Huber Loss
        true: array of true values    
        pred: array of predicted values
        
        returns: smoothed mean absolute error loss
        """
        loss = np.where(np.abs(true-pred) < delta , 0.5*((true-pred)**2), delta*np.abs(true - pred) - 0.5*(delta**2))
        return np.sum(loss)
    
    return sm_mae

def sm_mae_func(delta):
    """
    Smooth Mean Absolute Error/ Huber Loss
    true: array of true values    
    pred: array of predicted values
    
    returns: smoothed mean absolute error loss
    """
    def sm_mae(true, pred):
        """
        Smooth Mean Absolute Error/ Huber Loss
        true: array of true values    
        pred: array of predicted values
        
        returns: smoothed mean absolute error loss
        """
        loss = np.where(np.abs(true-pred) < delta , 0.5*((true-pred)**2), delta*np.abs(true - pred) - 0.5*(delta**2))
        return np.sum(loss)
    
    return sm_mae

def logcosh(true, pred):
    """
    Log cosh loss
    true: array of true values    
    pred: array of predicted values
    
    returns: smoothed mean absolute error loss
    """
    loss = np.log(np.cosh(pred - true))
    return np.sum(loss)

def quan_func(theta):
    """
    Quantile loss
    true: array of true values    
    pred: array of predicted values
    
    returns: smoothed mean absolute error loss
    """
    def quan(true, pred):
        """
        Quantile loss
        true: array of true values    
        pred: array of predicted values
        
        returns: smoothed mean absolute error loss
        """
        loss = np.where(true >= pred, theta*(np.abs(true-pred)), (1-theta)*(np.abs(true-pred)))
        return np.sum(loss)
    
    return quan

