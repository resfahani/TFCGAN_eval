"""
Created on Thu Nov  3 11:13:02 2022
@author: reza

"""

from tensorflow import keras
import numpy as np
import librosa
#import scipy as sp
from scipy.signal import tukey, butter, filtfilt
from numba import prange
import numba as nb 


### STFT parameters used in pre-processing step
win_length = 128+64 # Window length
hop_length = 16 # Hop length
n_fft = 256 #nfft


################################ Phase retireval ##################################
def PRA_GLA(TF, PRint = 10):
    
    mag = np.abs(TF)
    phase = np.random.uniform(0, 2*np.pi,(mag.shape[0], mag.shape[1]))  
    x = librosa.istft(mag * np.exp(phase *1j),  hop_length= hop_length, win_length = win_length, length=4000)
    
    for i in range(PRint):
        
        TFR = (librosa.stft(x, n_fft=n_fft,  hop_length= hop_length, win_length = win_length))[:128,:248]
        phase = np.angle(TFR)        
        TFR = mag *np.exp(1j* phase)
        x = (librosa.istft(TFR,  hop_length= hop_length, win_length = win_length, length=4000))
        
    #S = taper_filter(S, 0.01, 49, 100)
    return x

def PRA_ADMM(TF,  rho, eps, PRint = 10 , ab = 0):
    # Code modified from https://github.com//phvial/PRBregDiv

    mag = np.abs(TF)    
    phase =  np.random.uniform(0, 0.2,(mag.shape[0], mag.shape[1]))
    TFR =  mag * np.exp(1j * phase)     
    x = librosa.istft(TFR, hop_length=hop_length, win_length =win_length,length=4000)
    A = 0
    x = My_filter(x, 0.05, 48, 100 )
    for ii in range(PRint):
        
        X = librosa.stft(x, hop_length=hop_length, win_length=win_length, n_fft = n_fft)[:128,:248]
        H = X + (1/rho)* A
        ph = np.angle(H)
        U = computeProx(abs(H), mag, rho, eps, ab)
        Z = U * np.exp(1j* ph)
        x = librosa.istft(Z-  (1/rho)* A, hop_length=hop_length, win_length=win_length, length=4000)
        Xhat = librosa.stft(x, hop_length=hop_length, win_length=win_length, n_fft = n_fft)[:128,:248]
        A = A + rho * (Xhat - Z)
        
        x = My_filter(x, 0.05, 48, 100 )
        
    return x

def computeProx(Y, R, rho, eps, ab):
    # Code modified from https://github.com//phvial/PRBregDiv
    eps = np.min(R) +eps
    if ab == 1:
        v=(rho*Y+2*R)/(rho+2)
        
    if ab == 2:
        b=1/(R+eps)-rho*Y
        delta = np.square(b)+4*rho
        v = (-b+np.sqrt(delta))/(2*rho)
     
    #v = 1/rho * ss.lambertw(rho * R * np.exp(rho * Y))
    #delta = 4*rho*R + np.square(1-Y)
    #v = (Y - 1 + np.sqrt(delta))/(2*rho)

    return v

def my_STFT(x):
    
    X = librosa.stft(x, hop_length=hop_length, win_length=win_length, n_fft = n_fft)[:128,:248]    
    return X

def butter_bandpass(lowcut, highcut, fs, order = 4):
    
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    if low <= 0:
        Wn = high
        btype = "lowpass"
    elif high <= 0:
        Wn = low
        btype = "highpass"
    else:
        Wn = [low, high]
        btype = "bandpass"

    b, a = butter(order, Wn, btype=btype)
    
    return b, a


def My_filter(y, fmin, fmax, samp):
    b, a = butter_bandpass(fmin, fmax, samp)
    window_time = tukey(y.shape[-1], 0.1)
    return filtfilt(b, a, y * window_time, axis=-1)


################################ TFCGAN #########################################

class TFCGAN():
    
    def __init__(self, dirc, scalemin = -10, scalemax = 2.638887, pwr = 1, noise= 100, mtype = 1):
        """
        
        Created on Thu Nov  3 11:13:02 2022
        @author: reza
        
        Input parameters description:
        dirc: Trained model directory
        scalemin: Scale factor in Pre-processing step
        scalemax: Scale factor in pre-processing step
        pwr: Power spectrum, 
            1: means absolute value
            2: spectrogram
        noise: Noise vector
        mtype: Type of input label
            0: insert labels (Mw, R, Vs30) as a vector
            1: inset labels (Mw, R, Vs30) separately. 
            
        """
        
        self.dirc = dirc # Model directory 
        self.pwr = pwr # Power or absolute 
        self.scalemin = scalemin * self.pwr  # Scaling (clipping the dynamic range)
        self.scalemax = scalemax * self.pwr # Maximum value 
        self. noiseint = noise # later space
        self.dt = 0.01
        
        self.Model = keras.models.load_model(self.dirc) # Load the model
        self.mtype = mtype
        
    # Generate TFR
    def Generator(self, mag, dis, vs,noise, ngen=1):
        """
        
        Created on Thu Nov  3 11:13:02 2022
        @author: reza
        
        Generate TF representation for one scenario
        
        Input parameters description:
        mag: Magnitude value
        dis: Distance value
        Vs: Vs30 value
        ngen: Number of generatation
        
        Output parameter:
        TF: Descaled Time-frequency representation
        
        """
        
        mag = np.ones([ngen, 1]) * mag
        dis = np.ones([ngen, 1]) * dis
        vs  = np.ones([ngen, 1]) * vs / 1000
        
        label = np.concatenate([mag,dis, vs ],axis=1)

        if self.mtype == 0:
            TF = self.Model.predict([label,  noise])[:,:,:,0]
        elif self.mtype ==1:
            TF = self.Model.predict([label[:,0], label[:,1],label[:,2],  noise])[:,:,:,0]
        
        TF = (TF+1)/2
        TF = (TF* (self.scalemax-self.scalemin)) + self.scalemin
        TF = (10**TF)**(1/self.pwr)
        
        return TF
    
    # Calculate the TF, Time-history, and FAS
    #@nb.jit(parallel=True)
    def Maker(self, mag, dis, vs ,  ngen = 1, PRint = 10, mode = "ADMM", rho = 1e-5, eps = 1e-3, ab = 1):
        """
        
        Genetate accelerogram for one scenario
        
        Input parameters description:
        mag: Magnitude value
        dis: Distance value
        Vs: Vs30 value
        ngen: Number of time-history generatation
        PRint: Number of iteration in Phase retireval
        mode: Type of Phase retireval algorithm
            "GLA": Griffin-Lim Algorithm
            "ADMM": ADMM algorithm for phase retireval based on  Bregman divergences (https://hal.archives-ouvertes.fr/hal-03050635/document)
        
        Output parameter:
        tx: time vector
        freq = frequency veoctor
        xh: Generated time-hisory matrix
        S: Descaled Time-frequency representation matrix 
        
        """

        noise = np.random.normal(0, 1, (ngen, self. noiseint))
        
        S = self.Generator(mag, dis,vs, noise, ngen=ngen)
        
        x = np.empty((ngen, 4000))
        x[:] = 0
        
        for i in prange(ngen):
            if mode == "ADMM":
                x[i,:] = PRA_ADMM(S[i,:,:],rho, eps, PRint, ab)
            elif mode == "GLA":
                x[i,:] = PRA_GLA(S[i,:,:], PRint)
                
        freq, xh = self.fft(x)
        tx= np.arange(x.shape[1]) * self.dt
        
        return tx, freq , xh.squeeze(), S, x.squeeze()
    
    def fft(self, S):
        # Unnormalized fft without any norm specification
        
        if len(S.shape) == 1: S = S[np.newaxis,:]
        
        n = S.shape[1]//2
        lp = np.abs(np.fft.fft(S, norm = "forward", axis = 1))[:, :n]
        freq = np.linspace(0, 0.5, n)/self.dt
        
        return freq, lp.T
    



