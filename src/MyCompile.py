from tensorflow.keras.optimizers import RMSprop, Adam
import tensorflow as tf

import Training as TR
import Model as MGD
from MyData import Data_Loader

import numpy as np
import h5py
import matplotlib.pyplot as plt



@tf.RegisterGradient("ResizeBilinearGrad")
def _ResizeBilinearGrad_grad(op, grad):
    up = tf.image.resize(grad,tf.shape(op.inputs[0])[1:-1])
    return up,None


@tf.RegisterGradient("ResizeNearestNeighborGrad")
def ResizeNearestNeighborGrad(op, grad):
    up = tf.image.resize(grad,tf.shape(op.inputs[0])[1:-1])
    return up,None


Data2, info, Dmin, Dmax = Data_Loader()

print(Data2.shape)
print(info.shape)
print(Dmin)
print(Dmax)

rndbacht =  np.random.randint(0, Data2.shape[0],Data2.shape[0])
Data2 = Data2[rndbacht]# np.log(Data2+1e-16)
info = info[rndbacht]

nTrain = np.int(len(Data2) * 0.8)
Data2_Training = Data2[:nTrain]
info_Training = info[:nTrain]


Data2_Valdidate = Data2[nTrain+1:]
info_Valdidate = info[nTrain+1:]


np.save("DATAVALIDATE", Data2_Valdidate[:,:,:,0])
np.save("DATAVALIDATE2", Data2_Valdidate[:4000,:,:,0])

np.save("INFOVALIDATE", info_Valdidate)

print(Data2_Training.shape)
print(Data2_Valdidate.shape)


g_optimizer = Adam(lr =  2e-5 , beta_1 = 0.5, beta_2 = 0.999)
d_optimizer = Adam(lr =  3e-5 , beta_1 = 0.5, beta_2 = 0.999)


MODEL = MGD.MModel()
d_dis = MODEL.build_critic()
d_gen = MODEL.build_generator()


M = TR.CWGAN_GP(discriminator = d_dis, generator = d_gen, latent_dim = 100, 
             discriminator_extra_steps = 1, gp_weight = 10)


M.compile(d_optimizer = d_optimizer, g_optimizer = g_optimizer,
          d_loss_fn = TR.discriminator_loss, g_loss_fn  = TR.generator_loss)


history = M.fit(x = Data2_Training, y = info_Training, shuffle = True, 
        batch_size = 16, epochs = 120,
        validation_data = (Data2_Valdidate, info_Valdidate),
        workers = 5,  callbacks = TR.GANMonitor())


print(history.history['d_loss'])

d_gen.save("generator3")

#history.save("His_Save")
#np.save("./Historyrecord/History_Main", history)
 
