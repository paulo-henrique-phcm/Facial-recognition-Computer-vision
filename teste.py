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
from scipy.spatial import distance
import scipy
from keras_vggface import utils
'''
model = Sequential()
model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, (3, 3), activation='relu'))
model.add(MaxPooling2D((2,2), strides=(2,2)))

model.add(Convolution2D(4096, (7, 7), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(4096, (1, 1), activation='relu'))
model.add(Dropout(0.5))
model.add(Convolution2D(2622, (1, 1)))
model.add(Flatten())
model.add(Activation('softmax'))

from keras.models import model_from_json
model.load_weights('/home/co/Documentos/vgg-face/vgg_face_weights.h5')

vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
'''



    ###########

#vgg16 = keras.applications.vgg16.VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)

from keras_vggface.vggface import VGGFace

vggface2 = VGGFace(model='resnet50',include_top=True)
vggface2senet = Model(inputs=vggface2.layers[0].input, outputs=vggface2.layers[-3].output)
vggface2.summary()

'''
vggface2 = VGGFace(model='senet50',include_top=False)
vggface2senet = Model(inputs=vggface2.layers[0].input, outputs=vggface2.layers[-5].output)
vggface2.summary()
'''
############
#from keras_vggface.vggface import VGGFace
#vggface2 = VGGFace(model='resnet50')
#print('Inputs: %s' % model.inputs)
#print('Outputs: %s' % model.outputs)
##################

#model.summary()
#vgg16.summary()
#vggface2.summary()

#load_models()

def preprocess_image(image_end):
    img = load_img(image_end, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = utils.preprocess_input(img, version=2)
    return img

def acessa_imgs():
    data = glob('/home/co/Documentos/vgg-face/face-data-padrao/*') #acessa a pasta
    imagens = []
    for i in range(len(data)):
        #print(os.path.split(data[i])[-1]) #printa ultimo nome do caminho (nome da pasta)
        pasta = glob('/home/co/Documentos/vgg-face/face-data-padrao/'+os.path.split(data[i])[-1]+'/*') #pegas o endereço de todas as imagens de cada pasta
        img1 = (pasta[randint(0, (len(pasta)-1))]) #pega o endereço de uma imagem aleatoria na pasta
        img2 = (pasta[randint(0, (len(pasta)-1))])
        #imagens.append((os.path.split(data[i])[-1], img1, img2)) #adiciona o nome da pessoa e o caminho da imagem aleatoria
        imagens.append((img1, img2))
    #print(imagens[1])
    return imagens
    ##################################################usar dicionarios pode ser melhor
            #print('     '+os.path.split(pasta[j])[-1])

def predicts(caminho1, caminho2):
    image1 = preprocess_image(caminho1)
    image2 = preprocess_image(caminho2)

    ###img1_representation1 = vgg_face_descriptor.predict(image1)
    ###img2_representation2 = vgg_face_descriptor.predict(image2)

    img161 = vggface2senet.predict(image1)
    img162 = vggface2senet.predict(image2)
    return img161, img162

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

def cosine_measure(v1, v2):
    '''
    prod = np.dot(v1, v2)
    len1 = math.sqrt(dot_product(v1, v1))
    len2 = math.sqrt(dot_product(v2, v2))
    return prod / (len1 * len2)
    '''
    #def findCosineDistance(vector_1, vector_2):
    a = np.matmul(np.transpose(v1), v2)
    b = np.matmul(np.transpose(v1), v1)
    c = np.matmul(np.transpose(v2), v2)
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))




def verifyFace(vgg1, vgg2): #compara pesos por distancia euclidiana da vggface e vgg16
    euclidean_distance = findEuclideanDistance(vgg1, vgg2)
    #print(vgg1.shape)
    #cosine_dist = scipy.spatial.distance.cosine(vgg1, vgg2)
    #cosine_dist = distance.cosine(vgg1[0], vgg2[0], w=None)
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
    return euclidean_distance

def dist_final(caminho1, caminho2):
    img1 = preprocess_image(caminho1)
    img2 = preprocess_image(caminho1)

    pred1, pred2 = predicts(caminho1, caminho2)
    distancia = verifyFace(pred1, pred2)

    #print('vgg face: {}'.format(face))
    #print('vgg 16:   {}'.format(vgg))
    return distancia

vgg_igual = [] #distancia para imagens da mesma pessoa em vgg16
#face_igual = [] #distancia para imagens da mesma pessoa em vggFace

vgg_outro = [] #distancia entre pessoa e as outras em vgg16
#face_outro = [] #distancia entre pessoa e as outras em vggFace


for i in range(15):
    end_imgs2 = acessa_imgs()
    print('mesma pessoa ')
    for pessoa in end_imgs2: #para cada pessoa retorna a distancia euclidiana DELA COM ELA MESMA
        vgg = dist_final(pessoa[0], pessoa[1])
        vgg_igual.append(vgg)

end_imgs = acessa_imgs()
for pessoa in end_imgs: #para cada pessoa retorna a distancia DELA PARA TODAS AS OUTRAS PESSOAS
    print('fotos diferentes ')
    for D in end_imgs:
        if D!=pessoa:
            vgg = dist_final(D[1], pessoa[1])
            vgg_outro.append(vgg)
''' #plota
plt.hist(face_igual, facecolor='blue')
plt.hist(face_outro, facecolor='green')
plt.show()

plt.hist(vgg_igual, facecolor='blue')
plt.hist(vgg_outro, facecolor='green')
plt.show()

HHface_igual = np.asarray(face_igual)
HHface_outro = np.asarray(face_outro)
'''
HHvgg_igual = np.asarray(vgg_igual)
HHvgg_outro = np.asarray(vgg_outro)

np.save('/home/co/Documentos/face-data-padrao/vggface2_dataGrande_igual_euclid_PROCESS2.npy', HHvgg_igual)
np.save('/home/co/Documentos/face-data-padrao/vggface2_dataGrande_outro_euclid_PROCESS2.npy', HHvgg_outro)
#print(face_imp)
#print(vgg_imp)
