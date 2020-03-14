# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 10:56:52 2019

@author: mateus
"""

from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from PIL import Image
import numpy as np
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as plt
import keras
import os
from glob import glob
from random import randint
import matplotlib.pyplot as plt
from keras_vggface.vggface import VGGFace

model = VGGFace(model='resnet50',include_top=False)
model.summary()


def preprocess_image(image_end):
    img = load_img(image_end, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def acessa_imgs():
    data = glob('/home/mateus/Documents/visao/face-data/train/*') #acessa a pasta
    imagens = []
    for i in range(len(data)):
        print(os.path.split(data[i])[-1]) #printa ultimo nome do caminho (nome da pasta)
        pasta = glob('/home/mateus/Documents/visao/face-data/train/'+os.path.split(data[i])[-1]+'/*') #pegas o endereço de todas as imagens de cada pasta
        img1 = (pasta[randint(1, (len(pasta)-1))]) #pega o endereço de uma imagem aleatoria na pasta
        img2 = (pasta[randint(1, (len(pasta)-1))])
        #imagens.append((os.path.split(data[i])[-1], img1, img2)) #adiciona o nome da pessoa e o caminho da imagem aleatoria
        imagens.append((img1, img2))
    #print(imagens[1])
    return imagens
    ##################################################usar dicionarios pode ser melhor
            #print('     '+os.path.split(pasta[j])[-1])

def predicts(caminho1):
    image1 = preprocess_image(caminho1)

    img161 = model.predict(image1)
    return img161

'''
def findCosineDistance(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return (1 - (a / (np.sqrt(b) * np.sqrt(c))))
'''
def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance



def verifyFace(img1_representation, img2_representation, vgg161, vgg162): #compara pesos por distancia euclidiana da vggface e vgg16
    #img1_representation = vgg_face_descriptor.predict(preprocess_image(img1))
    #img2_representation = vgg_face_descriptor.predict(preprocess_image(img2))
    #cosine_similarity = findCosineDistance(img1_representation, img2_representation)
    #print("Cosine similarity: ",cosine_similarity)

    euclidean_distance = findEuclideanDistance(img1_representation, img2_representation)
    euclidean_distance_16 = findEuclideanDistance(vgg161, vgg162)
    '''
    if(euclidean_distance < epsilonEuc):
        print("Distancia Euclidiana... Mesma pessoa")
    else:
        print("Distancia Euclidiana... NÃO são a mesma pessoa")
    '''
    #print('vgg face: {}'.format(euclidean_distance))
    #print('vgg 16:   {}'.format(euclidean_distance_16))

    face = (euclidean_distance)
    vgg = (euclidean_distance_16)
    ''' ##### plota as duas imagens comparadas acima
    f = plt.figure()
    f.add_subplot(1,2, 1)
    plt.imshow(image.load_img(caminho1))
    plt.xticks([]); plt.yticks([])
    f.add_subplot(1,2, 2)
    plt.imshow(image.load_img(caminho2))
    plt.xticks([]); plt.yticks([])
    plt.show(block=True)
    print("-----------------------------------------")
    '''
    return face, vgg


'''
end_imgs = acessa_imgs()
vgg_inf = []
face_inf = []
for pessoa in end_imgs:
    #pess1 = pessoa
    face, vgg = dist_final(pessoa[0], pessoa[1])
    face_inf.append(face)
    #face_inf.append(vgg)
    #vgg_inf.append(vgg)
    #print(pessoa[0])

print(face_inf)

HH = np.asarray(face_inf)
#print(face_inf)
#print(vgg_inf)



plt.hist(face_inf, 100, facecolor='green')
plt.savefig('1.png')
plt.show()


vgg_imp = []
face_imp = []
for pessoa in end_imgs:
    for D in end_imgs:
        if D!=pessoa:
            face, vgg = dist_final(D[0], pessoa[1])
            face_imp.append(face)
            vgg_imp.append(vgg)



print(face_imp)
print(vgg_imp)
'''
