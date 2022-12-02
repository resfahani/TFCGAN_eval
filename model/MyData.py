import numpy as np
import h5py

def Data_Loader():

    file = h5py.File('Meta_DATA_1D_4_400', 'r')
    info = file['M'][:336376,:]
    file.close()

    info[:,:] = np.round(info[:,:],1)
    print(info[1,:])

    d = h5py.File("Data_Base_1D_4_400",'r')
    Data2 = d['M'][:336376, :128,:248]
    d.close()


    indx = np.where(info[:,0] >= 3.8)[0]
    info = info[indx,:]
    Data2 = Data2[indx]

    print(Data2.shape)

    indx = np.where(info[:,4] >= 1)[0]
    info = info[indx,:]
    Data2 = Data2[indx]

    indx = np.where(info[:,4] <= 30)[0]
    info = info[indx,:]
    Data2 = Data2[indx]


    indx = np.where(info[:,1] >= 1)[0]
    info = info[indx,:]
    Data2 = Data2[indx]


    info = np.delete(info, 3, axis = 1)
    info = np.delete(info, 3, axis = 1)
    info = np.delete(info, 3, axis = 1)
    info = np.delete(info, 3, axis = 1)
    info = np.delete(info, 3, axis = 1)
    
    
    ind = info[:,2] != 0
    info = info[ind]
    Data2 = Data2[ind]

    ind = np.isnan(info[:,2])
    info = info[~ind]
    Data2 = Data2[~ind]

    ind = np.isnan(info[:,1])
    info = info[~ind]
    Data2 = Data2[~ind]

    ind = info[:,1] != 0
    info = info[ind]
    Data2 = Data2[ind]

    ind = info[:,0] <= 7.5
    info = info[ind]
    Data2 = Data2[ind]

    ind = info[:,2] <= 1200
    info = info[ind]
    Data2 = Data2[ind]
    
    
    info[:,2] = info[:,2]/1000
    info[:,1] = np.round(info[:,1],1)
    info[:,1] = info[:,1]/1
    info[:,0] = info[:,0]/1
    
    
    Data2[Data2 <= 1e-10] =  1e-10
    Data2 = Data2**1
    Data2 = np.log10(Data2)
    Dmin = np.min(Data2)
    Dmax = np.max(Data2)
    Data2 = (Data2- Dmin)/(Dmax-Dmin)
    Data2 = Data2 * 2 - 1
    Data2 = Data2[:,:, :, np.newaxis]

    
    return Data2, info, Dmin, Dmax




