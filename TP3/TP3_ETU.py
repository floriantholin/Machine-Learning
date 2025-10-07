
import numpy as np
import matplotlib.pyplot as plt
# from fn import norm1, norm2
import os
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal, norm
from PIL import Image

DATA_PATH = 'Data'

image_ext = ['.jpg']
images =  [os.path.join(DATA_PATH, f) for f in os.listdir(DATA_PATH) if f.endswith(".jpg")]

X_train=np.empty((0, 2))
y_train=[]
for name in images[0:-4]:
    I= Image.open(name)
    I = I.convert('YCbCr')  #Y:Luminance, Cb:ChrominanceR, Cr:ChrominanceB
    I = np.array(I)
    I=np.reshape(I[:,:,1:3],[I.shape[0]*I.shape[1],2])
    X_train = np.concatenate((X_train, I), axis=0)
    fichier_png= os.path.splitext(name)[0] + ".png"
    GT = Image.open(fichier_png)
    GT = np.array(GT)
    GT=np.reshape(GT[:,:,1]/255,[GT.shape[0]*GT.shape[1]])
    y_train = np.concatenate((y_train, GT), axis=0)


X_test=np.empty((0, 2))
y_test=[]
for name in images[-4:]:
    I= Image.open(name)
    I = I.convert('YCbCr')
    I = np.array(I)
    I=np.reshape(I[:,:,1:3],[I.shape[0]*I.shape[1],2])
    X_test = np.concatenate((X_test, I), axis=0)
    fichier_png= os.path.splitext(name)[0] + ".png"
    GT = Image.open(fichier_png)
    GT = np.array(GT)
    GT=np.reshape(GT[:,:,1]/255,[GT.shape[0]*GT.shape[1]])
    y_test = np.concatenate((y_test, GT), axis=0)

X_train = X_train.astype('float64')
X_test = X_test.astype('float64')

# X_train, pipo, y_train, pipo = train_test_split(X_train, y_train, train_size=1/1000, random_state=42)
# X_test, pipo, y_test, pipo = train_test_split(X_test, y_test, train_size=1/1000, random_state=42)

# #############################################################################
# I.	Chargement et visualisation des données


# #Pixel peau
# Peau_Train = X_train[np.where(y_train==1),:]
# Peau_Train = np.reshape(Peau_Train,(Peau_Train.shape[1],Peau_Train.shape[2] ))
# #Pixel non peau
# Nonpeau_Train = X_train[np.where(y_train==0),:]
# Nonpeau_Train = np.reshape(Nonpeau_Train,(Nonpeau_Train.shape[1],Nonpeau_Train.shape[2] ))


# plt.plot(Nonpeau_Train[:,0], Nonpeau_Train[:,1], '.b', label='Non peau')
# plt.plot(Peau_Train[:,0], Peau_Train[:,1], '.r', label='Peau')
# plt.legend()
# plt.show()




