#from keras.models import Model, Sequential
#from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from PIL import Image
import numpy as np
#from keras.preprocessing.image import load_img, save_img, img_to_array
#from keras.applications.imagenet_utils import preprocess_input
#from keras.preprocessing import image
import matplotlib.pyplot as plt
#import keras
import os
from glob import glob
from random import randint
import matplotlib.pyplot as plt

import scikitplot as skplt

from sklearn.metrics import roc_curve, auc, roc_auc_score
from scipy.spatial import distance
#HHface = numpy.save('/home/co/Documentos/vgg-face/FACEface.npy')
#HHvgg = numpy.save('/home/co/Documentos/vgg-face/FACEvgg.npy')



vggface_igual = np.load('/home/co/Documentos/vgg-face/vggface2_igual_euclid_PROCESS2.npy')
vggface_outro = np.load('/home/co/Documentos/vgg-face/vggface2_outro_euclid_PROCESS2.npy')
vggface2_igual = np.load('/home/co/Documentos/vgg-face/face-data-padrao/vggface2_gdriv_igual_euclid_PROCESS2.npy')
vggface2_outro = np.load('/home/co/Documentos/vgg-face/face-data-padrao/vggface2_gdriv_outro_euclid_PROCESS2.npy')

def plot_histogram(face_igual, face_outro, vgg_igual, vgg_outro):
    plt.subplot(1, 2, 1)
    plt.hist(face_igual, 50, alpha = 0.8, facecolor='blue')
    plt.hist(face_outro, 50, alpha = 0.8, facecolor='red')
    plt.title('SENET - euclidean')
    plt.xlabel('distancia euclidiana')
    plt.ylabel('QTD ocorrencias')

    plt.subplot(1, 2, 2)
    plt.hist(vgg_igual, 50, alpha = 0.8, facecolor='blue')
    plt.hist(vgg_outro, 50, alpha = 0.8, facecolor='red')
    plt.title('RESNET - euclidean')
    plt.xlabel('distancia euclidiana')
    plt.ylabel('QTD ocorrencias')
    plt.show()

#print(face_igual)

def plot_curve_roc(face_igual, face_outro, color):

    y = np.zeros(face_igual.shape[0] + face_outro.shape[0])
    scores = np.zeros(face_igual.shape[0] + face_outro.shape[0])
    #########################gera um vetor com as classes 1 para pessoas iguais, 0 para diferentes
    for i in range(face_igual.shape[0]):
        y[i] = 1
        scores[i]=face_igual[i]
    ########################coloca os dois vetores (pessoas iguais e diferentes), num mesmo vetor, em sequencia
    for i in range(face_outro.shape[0]):
        scores[face_igual.shape[0]+i] = face_outro[i]





    A,B,C = roc_curve(y,scores) #calcula os parametros para a roc usando os vetores gerados acima
    faceAuc = auc(B, A) #calcula a area auc usando  saida do roc acima

    #plt.subplot(1, 3, 3)
    plt.plot(B, A, linestyle='--', color=color, label='\nauc: {}' .format(faceAuc)) #plota as rocs com as aucs

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    #plt.show()
vermelha = 'red'
verde = 'green'
azul = 'blue'
#plota o histograma

plot_histogram(vggface_igual, vggface_outro, vggface2_igual, vggface2_outro)
plot_curve_roc(vggface_igual, vggface_outro, vermelha)
plot_curve_roc(vggface2_igual, vggface2_outro, verde)
plt.show()
