from tensorflow import keras
import numpy as np
import librosa
#import scipy as sp
from scipy.signal import tukey, butter, filtfilt
from numba import prange
import numba as nb 

rho = 1e-4
eps=1e-5
win_length = 128+64
hop_length = 16
n_fft = 256


################################ Phase retireval ##################################

def PRA_GLA(TF):
    
    mag = np.abs(TF)
    phase = np.random.uniform(0, 2*np.pi,(mag.shape[0], mag.shape[1]))  
    x = librosa.istft(mag * np.exp(phase *1j),  hop_length= hop_length, win_length = win_length, length=4000)
    
    for i in range(10):
        
        TFR = (librosa.stft(x, n_fft=n_fft,  hop_length= hop_length, win_length = win_length))[:128,:248]
        phase = np.angle(TFR)        
        TFR = mag *np.exp(1j* phase)
        x = (librosa.istft(TFR,  hop_length= hop_length, win_length = win_length, length=4000))
        
    #S = taper_filter(S, 0.01, 49, 100)
    return x

def PRA_ADMM(TF):
    
    mag = np.abs(TF)    
    phase =  np.random.uniform(0, 2 *np.pi,(mag.shape[0], mag.shape[1]))
    TFR =  mag * np.exp(1j * phase)     
    x = librosa.istft(TFR, hop_length=hop_length, win_length =win_length,length=4000)
    A = 0
    #x = taper_filter(x, 0.1, 48, 100 )
    for ii in range(10):
        
        X = librosa.stft(x, hop_length=hop_length, win_length=win_length, n_fft = n_fft)[:128,:248]
        H = X + (1/rho)* A
        ph = np.angle(H)
        U = computeProx(abs(H), mag)
        Z = U * np.exp(1j* ph)
        x = librosa.istft(Z-  (1/rho)* A, hop_length=hop_length, win_length=win_length, length=4000)
        Xhat = librosa.stft(x, hop_length=hop_length, win_length=win_length, n_fft = n_fft)[:128,:248]
        A = A + rho * (Xhat - Z)
        
    #x = taper_filter(x, 0.1, 48, 100 )
    return x

def computeProx(Y, R):
    
    v=(rho*Y+2*R)/(rho+2)
    b=1/(R+eps)-rho*Y
    delta = np.square(b)+4*rho
    v = (-b+np.sqrt(delta))/(2*rho)
    
    return v

def my_STFT(x):
    X = librosa.stft(x, hop_length=hop_length, win_length=win_length, n_fft = n_fft)[:128,:248]    
    return X


################################ TFCGAN #########################################

class TFCGAN():
    
    def __init__(self, dirc, scalemin = -10, scalemax = 2.638887, pwr = 1):
        
        self.dirc = dirc # Model directory 
        self.pwr = pwr # Power or absolute 
        self.scalemin = scalemin * self.pwr  # Scaling (clipping the dynamic range)
        self.scalemax = scalemax * self.pwr # Maximum value 
        self. noiseint = 100 # later space
        self.dt = 0.01
        
        self.Model = keras.models.load_model(self.dirc) # Load the model

    # Generate TFR
    def Generator(self, mag, dis, vs,noise, nit=1):
        
        mag = np.ones([nit,1]) * mag
        dis = np.ones([nit,1]) * dis
        vs  = np.ones([nit,1]) * vs / 1000
        
        label = np.concatenate([mag,dis, vs ],axis=1)
        
        cc = self.Model.predict([label,  noise])[:,:,:,0]
        
        cc = (cc+1)/2
        cc = (cc* (self.scalemax-self.scalemin)) + self.scalemin
        cc = (10**cc)**(1/self.pwr)
        
        return cc
    
    # Calculate the TF, Time-history, and FAS
    nb.jit(parallel=True)
    def Maker(self, mag, dis, vs ,  nit = 1, mode = "ADMM"):
        
        noise = np.random.normal(0, 1, (nit, self. noiseint))
        
        S = self.Generator(mag, dis,vs, noise, nit=nit)
        
        x = np.empty((nit,4000))
        x[:] = 0
        
        for i in prange(nit):
            if mode == "ADMM":
                x[i,:] = PRA_ADMM(S[i,:,:])
            elif mode == "GLA":
                x[i,:] = PRA_GLA(S[i,:,:])
                
        freq, xh = self.fft(x)
        tx= np.arange(x.shape[1]) * self.dt
        
        return tx, freq , xh.squeeze(), S, x.squeeze()
    
    def fft(self, S):
        
        if len(S.shape) == 1: S = S[np.newaxis,:]
        
        n = S.shape[1]//2
        lp = np.abs(np.fft.fft(S, axis = 1))[:, :n]
        freq = np.linspace(0, 0.5, n)/self.dt
        
        return freq, lp.T
    
