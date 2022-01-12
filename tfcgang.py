
from tensorflow import keras
import numpy as np
import librosa
import scipy as sp
import numba as nb
from scipy.signal import tukey, butter, filtfilt

rho = 1e-4
eps=1e-5
win_length = 128+64
hop_length = 16
n_fft = 256

def genT(cc):

    phase =cc * np.random.uniform(0, 2*np.pi,(cc.shape[0], cc.shape[1]))    
    phase = 1 * cc * np.exp(phase *1j)
    mag = np.abs(cc)
    S = (librosa.istft(phase,  hop_length= hop_length, win_length = win_length, length=4000))
    S = taper_filter(S, 0.01, 49, 100 )
    
    for i in range(25):
        
        S = (librosa.stft(S,n_fft=n_fft,  hop_length= hop_length, win_length = win_length))[:128,:248]
        ph = np.angle(S)
        nTF = mag *np.exp(1j* ph)
        S = (librosa.istft(nTF,  hop_length= hop_length, win_length = win_length, length=4000))
        
    S = taper_filter(S, 0.01, 49, 100)
    return S

def computeProx(Y, R):
     
    v=(rho*Y+2*R)/(rho+2)
    b=1/(R+eps)-rho*Y
    delta = np.square(b)+4*rho
    v = (-b+np.sqrt(delta))/(2*rho)
    
    return v


def my_STFT(x):
    
    X = librosa.stft(x, hop_length=hop_length, win_length=win_length, n_fft = n_fft)[:128,:248]
    
    return X


def GETT(TF):
    
    
    mag = np.abs(TF)
    
    phase = mag * np.random.uniform(0, 1 *np.pi,(mag.shape[0], mag.shape[1]))
    
    nTF =  mag * np.exp(1j * phase) #+ 1*np.abs(np.random.randn(mag.shape[0], mag.shape[1]))
    
    x = librosa.istft(nTF, hop_length=hop_length, win_length =win_length,length=4000)
    
    A = 0
    
    #x = taper_filter(x, 0.1, 48, 100 )
    
    for ii in range(25):
        
        X = librosa.stft(x, hop_length=hop_length, win_length=win_length, n_fft = n_fft)[:128,:248]
        H = X + (1/rho)* A
        ph = np.angle(H)
        U = computeProx(abs(H), mag)
        Z = U * np.exp(1j* ph)
        x = librosa.istft(Z-  (1/rho)* A, hop_length=hop_length, win_length=win_length, length=4000)
        Xhat = librosa.stft(x, hop_length=hop_length, win_length=win_length, n_fft = n_fft)[:128,:248]
        A = A + rho * (Xhat - Z)
    x = taper_filter(x, 0.1, 48, 100 )
    return x


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

def taper_filter(arr, fmin, fmax, samp_DAS):
    b, a = butter_bandpass(fmin, fmax, samp_DAS)
    window_time = tukey(arr.shape[-1], 0.1)
    arr_wind = arr * window_time
    arr_wind_filt = filtfilt(b, a, arr_wind, axis=-1)
    return arr_wind_filt



class TFCGAN():
    def __init__(self, addr, scalemin = -10, scalemax = 2.638887, pwr = 1):
        self.addr = addr
        self.scalemin = scalemin
        self.scalemax = scalemax
        self.pwr = pwr
        self. noiseint = 100
        self.Model = keras.models.load_model(self.addr)


    def Generator(self, mag, dis, vs,noise, nit=1):
        
        Sclemin = self.scalemin * self.pwr 
        Sclemax = self.scalemax * self.pwr
        
        mag = np.ones([nit,1]) * mag
        dis = np.ones([nit,1]) * dis
        vs  = np.ones([nit,1]) * vs / 1000

        label = np.concatenate([mag,dis, vs ],axis=1)
        
        cc = (self.Model.predict([label,  noise]))
        
        cc = (cc+1)/2
        cc = (cc* (Sclemax-Sclemin)) + Sclemin
        cc = 10**cc[:,:,:,0]
        cc = cc**(1/self.pwr)
        
        return cc

    

    def Maker(self, mag, dis, vs ,  nit = 1):
        
        noise = np.random.normal(0, 1, (nit, self. noiseint))
        
        S = self.Generator(mag, dis,vs, noise, nit=nit)
        #S[:,:0,:] = 0
        
        x = []
        xh = []
        
        for i in range(nit):
            
            x.append(GETT(S[i,:,:]))
            #x.append(genT(S[i,:,:]))
            
            freq, lp = self.ft(x[i], 0.01)
            xh.append(lp)
            
        return freq , np.asarray(xh).T, S, np.asarray(x).T
    

    def ft(self, S1, dt):
        
        lp = np.abs(np.fft.fft(S1))
        lp = lp[:len(lp)//2]
        freq = np.linspace(0,0.5,len(lp))*1/dt
        
        return freq, np.abs(lp)






