from tensorflow.keras.layers import Dense, BatchNormalization, Activation, LeakyReLU, UpSampling2D, Conv2D
from SpectralNormalizationKeras import DenseSN, ConvSN2D
import tensorflow as tf

# Weight initilization
def weight_init(stddev = 0.02):
    return tf.random_normal_initializer(mean=0.0, stddev=stddev)
    
# Fully connected layer 
def DL1(y, n = 8, BN = "BN", bias = False, ACT_F="relu", stddev = 0.02):
    
    weights = weight_init(stddev= stddev)

    if BN == "BN":
        y = Dense(n , kernel_initializer=weights, use_bias = bias)(y)
        y = BatchNormalization(momentum=0.8)(y)

    elif BN == "SN":
        y = DenseSN(n , kernel_initializer=weights, use_bias = bias)(y)

    elif BN == "N":
        y = Dense(n , kernel_initializer=weights, use_bias = bias)(y)
        
    if ACT_F == "relu":
        y = Activation(ACT_F)(y)
        
    elif ACT_F == "Lrelu":
        y = LeakyReLU(alpha=0.2)(y)
    
    elif ACT_F == "None":
        y = y
        
    return y


# Convolutional layer for Batchnormalization, SpectralNormalization, Or none + RELU, LRELU, TANH
def CONVL1(y, n = 8, BN ="BN", ks = 6, sd = 1, ups = 1, ACT_F= "relu", bias=False, stddev= 0.02):

    weights = weight_init(stddev= stddev)

    if BN == "BN":
        y = Conv2D(n , kernel_size = (ks), strides=sd ,kernel_initializer=weights, padding='same',use_bias=bias)(y)
        y = BatchNormalization(momentum=0.8)(y)
    
    elif BN == "SN":
        y = ConvSN2D(n , kernel_size = (ks), strides=sd ,kernel_initializer=weights, padding='same', use_bias=bias)(y)

    elif BN == "N":
        y = Conv2D(n , kernel_size = (ks), strides=sd ,kernel_initializer=weights, padding='same', use_bias=bias)(y)

    if ACT_F == "relu":
        y = Activation(ACT_F)(y)
        
    elif ACT_F == "tanh":
        y = Activation(ACT_F)(y)        

    elif ACT_F == "Lrelu":
        y = LeakyReLU(alpha=0.2)(y)

    y = UpSampling2D(size = (ups))(y)

    return y
 
