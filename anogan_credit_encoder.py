from __future__ import print_function
from keras.models import Sequential, Model
from keras.layers import Input, Reshape, Dense, Dropout, MaxPooling2D, Conv2D, Flatten
from keras.layers import Conv2DTranspose, LeakyReLU
from keras.layers.core import Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras import backend as K
from keras import initializers
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import cv2
import math

INPUT_DIM = 28


from keras.utils. generic_utils import Progbar

### combine images for visualization
def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:4]
    image = np.zeros((height*shape[0], width*shape[1], shape[2]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1],:] = img[:, :, :]
    return image

### generator model define
def generator_model():
    inputs = Input((10,))
    fc1 = Dense(input_dim=10, units=64)(inputs)
    fc1 = BatchNormalization()(fc1)
    fc1 = LeakyReLU(0.2)(fc1)

    fc1 = Dense(units=256)(fc1)
    fc1 = BatchNormalization()(fc1)
    fc1 = LeakyReLU(0.2)(fc1)


    fc2 = Dense(units=INPUT_DIM)(fc1)
    #fc2 = BatchNormalization()(fc2)
    outputs = fc2#LeakyReLU(0.2)(fc2)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model


### encoder model define
def encoder_model():
    inputs = Input((INPUT_DIM,))
    fc1 = Dense(input_dim=INPUT_DIM, units=64)(inputs)
    fc1 = BatchNormalization()(fc1)
    fc1 = LeakyReLU(0.2)(fc1)

    fc1 = Dense(units=256)(fc1)
    fc1 = BatchNormalization()(fc1)
    fc1 = LeakyReLU(0.2)(fc1)


    fc2 = Dense(units=10)(fc1)
    #fc2 = BatchNormalization()(fc2)
    outputs = fc2#LeakyReLU(0.2)(fc2)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model


### discriminator model define
def discriminator_model():
    inputs = Input((INPUT_DIM,))
    fc1 = Dense(input_dim=INPUT_DIM, units=16)(inputs)
    #fc1 = BatchNormalization()(fc1)
    fc1 = LeakyReLU(0.2)(fc1)

    fc2 = Dense(1)(fc1)
    outputs = Activation('sigmoid')(fc2)
    
    model = Model(inputs=[inputs], outputs=[outputs])
    return model





### d_on_g model for training generator
def generator_containing_discriminator(g, d):
    d.trainable = False
    ganInput = Input(shape=(10,))
    x = g(ganInput)
    ganOutput = d(x)
    gan_dg = Model(inputs=ganInput, outputs=ganOutput)
    return gan_dg


### g_on_e model for training generator
def generator_containing_encoder(g, e):
    g.trainable = False
    ganInput = Input(shape=(INPUT_DIM,))
    latent = e(ganInput)
    ganOutput = g(latent)
    gan_ge = Model(inputs=ganInput, outputs=ganOutput)
    return gan_ge


### combined traininng 
def combined_model(g, d, e):
    d.trainable = False
    gInput = Input(shape=(10,))
    x = g(gInput)
    gOutput = d(x)

    eInput = Input(shape=(INPUT_DIM,))
    latent = e(eInput)
    eOutput = g(latent)
    
    #dInput = K.concatenate([x, eOutput], axis=-1)
    #dOutput = d(dInput)

    #gan = Model(inputs=[gInput, eInput], outputs=dOutput)

    gan = Model(inputs=[gInput, eInput], outputs=[gOutput,eOutput])
    return gan



def load_model():
    d = discriminator_model()
    g = generator_model()
    e = encoder_model()
    d_optim = RMSprop()
    g_optim = RMSprop(lr=0.0002)
    e_optim = RMSprop()
    g.compile(loss='mse', optimizer=g_optim)
    d.compile(loss='mse', optimizer=d_optim)
    e.compile(loss='mse', optimizer=e_optim)
    d.load_weights('./weights/discriminator.h5')
    g.load_weights('./weights/generator.h5')
    e.load_weights('./weights/encoder.h5')
    return g, d, e

### train generator and discriminator
def train(BATCH_SIZE, X_train, EPOCH=20):
    
    ### model define
    d = discriminator_model()
    e = encoder_model()
    g = generator_model()
    d_on_g = generator_containing_discriminator(g, d)
    g_on_e = generator_containing_encoder(g, e)
    c = combined_model(g,d,e)
    d_optim = RMSprop(lr=0.0004)
    e_optim = RMSprop(lr=0.0004) 
    g_optim = RMSprop(lr=0.0002) 
    c_optim = RMSprop(lr=0.0004)  
    e.compile(loss='mse', optimizer=e_optim)    
    g_on_e.compile(loss='mae', optimizer=e_optim)
    g.trainable = True
    g.compile(loss='mse', optimizer=g_optim)
       
    c.compile(loss=['mse', 'mse'], loss_weights=[0.1, 0.9], optimizer=c_optim)

    d_on_g.compile(loss='mse', optimizer=g_optim)
    d.trainable = True
    d.compile(loss='mse', optimizer=d_optim)

    

    for epoch in range(EPOCH):
        print ("Epoch is", epoch)
        n_iter = int(X_train.shape[0]/BATCH_SIZE)
        progress_bar = Progbar(target=n_iter)
        
        for index in range(n_iter):
            # create random noise -> U(0,1) 10 latent vectors
            noise = np.random.uniform(0, 1, size=(BATCH_SIZE, 10))

            # load real data & generate fake data
            image_batch = X_train[index*BATCH_SIZE:(index+1)*BATCH_SIZE]
            generated_images = g.predict(noise, verbose=0)
            """
            # visualize training results
            if index % 20 == 0:
                image = combine_images(generated_images)
                image = image*127.5+127.5
                cv2.imwrite('./result/'+str(epoch)+"_"+str(index)+".png", image)
            """
            # attach label for training discriminator
            X = np.concatenate((image_batch, generated_images))
            y = np.array([1] * BATCH_SIZE + [0] * BATCH_SIZE)
            
            # training discriminator
            d_loss = d.train_on_batch(X, y)
        
            

            d.trainable = False
            g_loss = c.train_on_batch([noise, image_batch], [np.array([1] * BATCH_SIZE), image_batch])
            d.trainable = True


            """
            # training generator
            d.trainable = False
            g_loss_1 = d_on_g.train_on_batch(noise, np.array([1] * BATCH_SIZE))
            d.trainable = True

            # training encoder
            g.trainable = False
            g_loss_2 = g_on_e.train_on_batch(image_batch, image_batch)
            g.trainable = True
            """
            
            #print(g_loss, d_loss)
            progress_bar.update(index, values=[('g',g_loss[0]), ('g_d',g_loss[1]), ('g_e',g_loss[2]), ('d',d_loss)])
        print ('')

        # save weights for each epoch
        g.save_weights('weights/generator.h5', True)
        d.save_weights('weights/discriminator.h5', True)
        e.save_weights('weights/encoder.h5', True)
    return d, g, e

### generate images
def generate(BATCH_SIZE):
    g = generator_model()
    g.load_weights('weights/generator.h5')
    noise = np.random.uniform(0, 1, (BATCH_SIZE, 10))
    generated_images = g.predict(noise)
    return generated_images

### anomaly loss function 
def sum_of_residual(y_true, y_pred):
    return K.sum(K.abs(y_true - y_pred))

### discriminator intermediate layer feautre extraction
def feature_extractor(d=None):
    if d is None:
        d = discriminator_model()
        d.load_weights('weights/discriminator.h5') 
    intermidiate_model = Model(inputs=d.layers[0].input, outputs=d.layers[-4].output)
    intermidiate_model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    return intermidiate_model

### anomaly detection model define
def anomaly_detector(g=None, d=None):
    if g is None:
        g = generator_model()
        g.load_weights('weights/generator.h5')
    intermidiate_model = feature_extractor(d)
    intermidiate_model.trainable = False
    g = Model(inputs=g.layers[1].input, outputs=g.layers[-1].output)
    g.trainable = False
    # Input layer cann't be trained. Add new layer as same size & same distribution
    aInput = Input(shape=(10,))
    gInput = Dense((10), trainable=True)(aInput)
    gInput = Activation('sigmoid')(gInput)
    
    # G & D feature
    G_out = g(gInput)
    D_out= intermidiate_model(G_out)    
    model = Model(inputs=aInput, outputs=[G_out, D_out])
    model.compile(loss=sum_of_residual, loss_weights= [0.90, 0.10], optimizer='rmsprop')
    
    # batchnorm learning phase fixed (test) : make non trainable
    K.set_learning_phase(0)
    
    return model

### anomaly detection
def compute_anomaly_score(model, x, iterations=500, d=None):
    z = np.random.uniform(0, 1, size=(len(x), 10))
    
    intermidiate_model = feature_extractor(d)
    d_x = intermidiate_model.predict(x)

    #import pdb; pdb.set_trace()
    losses = []
    for k in range(len(x)):
        print('processing {:d} over {:d}'.format(k, len(x)))
        # learning for changing latent
        xk = x[k]
        xk = np.expand_dims(xk,0)
        d_xk = d_x[k]
        d_xk = np.expand_dims(d_xk,0)
        zk = z[k]
        zk = np.expand_dims(zk,0)
        loss = model.fit(zk, [xk, d_xk], batch_size=1, epochs=iterations, verbose=0)
        #similar_data, _ = model.predict(z)
        loss = loss.history['loss'][-1]
        losses.append(loss)    

    return np.array(losses)



def score_from_encoder(x, e=None, g=None):
    if e is None:
        e = encoder_model()
        e.load_weights('weights/encoder.h5')

    if g is None:
        g = generator_model()
        g.load_weights('weights/generator.h5')


    latent = e.predict(x)
    
    xgen = g.predict(latent) 

    losses = np.mean(abs(x-xgen),axis=1)

    return np.array(losses)
