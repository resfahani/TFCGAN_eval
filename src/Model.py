from tensorflow.keras.layers import Input, Reshape, Flatten, concatenate
from tensorflow.keras.models import  Model
from MyLayers import DL1, CONVL1
import tensorflow as tf

class MModel(tf.keras.Model):
    def __init__(self):
        
        self.i_downsample_rows = 16 # 32
        self.i_downsample_cols = 31 # 62
        
        self.img_rows = 128
        self.img_cols = 248
        self.channels = 1
        
        self.latent_dim = 200
        
        self.label  = 3
        self.label1  = 1
        
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        
        self.Gen_BN = "N"
        self.Dis_BN = "SN"
        
        self.Gen_act = "relu"
        self.Dis_act = "Lrelu"
        
    def build_generator(self):
        

        Label1 = Input(shape=(self.label,))

        noiseInput = Input(shape=(self.latent_dim,))

        L1 = DL1(Label1, BN=self.Gen_BN, n = 16, ACT_F=self.Gen_act, bias =True)
        L1 = DL1(L1, BN=self.Gen_BN, n = 32, ACT_F=self.Gen_act, bias =True)
        L1 = DL1(L1, BN=self.Gen_BN, n = 64, ACT_F=self.Gen_act, bias =True)
        L1 = DL1(L1, BN=self.Gen_BN, n = 128,  bias = True)
        
        
        L1 = DL1(L1, BN=self.Gen_BN, n = self.i_downsample_rows * self.i_downsample_cols * 3 , ACT_F=self.Gen_act, bias = True)
        L1 = Reshape((self.i_downsample_rows, self.i_downsample_cols, 3))(L1)
        
        L = DL1(noiseInput, BN="BN",ACT_F=self.Gen_act, n = self.i_downsample_rows * self.i_downsample_cols*1)
        L = Reshape((self.i_downsample_rows, self.i_downsample_cols, 1))(L)
        
        # Concatenate the noise and labels
        L = concatenate([L, L1 ])
        
        L = CONVL1(L, BN=self.Gen_BN, n = 256, ks = 3, sd = 1, ups = 1, ACT_F= self.Gen_act, bias =True)
        L = CONVL1(L, BN=self.Gen_BN, n = 128, ks = 3, sd = 1, ups = 2, ACT_F= self.Gen_act, bias = True)
        L = CONVL1(L, BN=self.Gen_BN, n = 64, ks = 3, sd = 1, ups = 2, ACT_F= self.Gen_act, bias = True)
        L = CONVL1(L, BN=self.Gen_BN, n = 32, ks = 3, sd = 1, ups = 2, ACT_F= self.Gen_act, bias = True)
        L = CONVL1(L, BN=self.Gen_BN, n = 16, ks = 3, sd = 1, ups = 1, ACT_F= self.Gen_act, bias = True)
        L = CONVL1(L, BN=self.Gen_BN, n = 8, ks = 3 , sd = 1, ups = 1, ACT_F= self.Gen_act, bias = True)
        L = CONVL1(L, BN=self.Gen_BN, n = 4, ks = 3, sd = 1, ups = 1, ACT_F= self.Gen_act, bias  = True)
        L = CONVL1(L, BN=self.Gen_BN, n = 1,  ks = 3, sd = 1, ups = 1, ACT_F= "tanh", bias = False)
        
        M = Model([Label1, noiseInput], L)
        M.summary()
        
        return M
    
    
    def build_critic(self):

        Label = Input(shape=(self.label,))
        imgInput = Input(shape=(self.img_shape))
        
        #Mn =  tf.reduce_max(((imgInput+1)/2) ,axis=1)
        #Mn =  tf.reduce_max(Mn ,axis=1)
        #Label2 = concatenate([Label, Mn])
        

        L = DL1(Label ,BN=self.Dis_BN, n = 16, ACT_F=self.Dis_act, bias =True)
        L = DL1(L ,BN=self.Dis_BN, n = 32, ACT_F=self.Dis_act, bias =True)
        L = DL1(L ,BN=self.Dis_BN, n = 64, ACT_F=self.Dis_act, bias =True)
        L = DL1(L ,BN=self.Dis_BN, n = 128, ACT_F=self.Dis_act, bias =True)
        
        L = DL1(L ,BN=self.Dis_BN, n = self.img_rows * self.img_cols * 1, ACT_F=self.Dis_act, bias =True)
        L = Reshape((self.img_rows, self.img_cols, 1))(L)
        
        # Concatenate real image and labels
        L = concatenate([imgInput, L])
        
        L = CONVL1(L, BN=self.Dis_BN, n = 4, ks = 3, sd = 1, ups = 1 , ACT_F= self.Dis_act, bias=True)
        L = CONVL1(L, BN=self.Dis_BN, n = 8, ks = 3, sd = 2, ups = 1 , ACT_F= self.Dis_act, bias=True)
        L = CONVL1(L, BN=self.Dis_BN, n = 16, ks = 3, sd = 1, ups = 1 , ACT_F= self.Dis_act, bias=True)
        L = CONVL1(L, BN=self.Dis_BN, n = 32, ks = 3, sd = 2, ups = 1 , ACT_F= self.Dis_act, bias=True)
        L = CONVL1(L, BN=self.Dis_BN, n = 64, ks = 3, sd = 1, ups = 1 , ACT_F= self.Dis_act, bias=True)
        L = CONVL1(L, BN=self.Dis_BN, n = 128, ks = 3, sd = 2, ups = 1 , ACT_F= self.Dis_act, bias=True)
        L = CONVL1(L, BN=self.Dis_BN, n = 256, ks = 3, sd = 2, ups = 1 , ACT_F= self.Dis_act, bias=True)
        L = CONVL1(L, BN=self.Dis_BN, n = 512, ks = 3, sd = 1, ups = 1 , ACT_F= self.Dis_act, bias=True)
        #L = CONVL1(L, BN=self.Dis_BN, n = 512, ks = 3, sd = 1, ups = 1 , ACT_F= self.Dis_act, bias=True)
        
        L = Flatten()(L)        
        L = DL1(L ,BN="N", n = 1, ACT_F="None", bias =False)
        
        
        M = Model([Label, imgInput], L)
        M.summary()
        
        return M
    
