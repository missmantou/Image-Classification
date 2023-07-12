# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 16:16:35 2022

@author: edeny
"""

from keras.applications.resent50 import ResNet50, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from PIL import Image
import numpy as np
Image.MAX_IMAGE_PIXELS = 211680000 # this is to avoid the error message

#Set up the base model
HEIGHT = 300
WIDTH = 300

base_model = ResNet50(weights = 'imagenet',
                      include_top = False,
                      input_shape = (HEIGHT, WIDTH, 3))

#Generate training data
TRAIN_DIR = 'training'
VAL_DIR = 'validation'
HEIGHT = 300
WIDTH = 300
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
        preprocessing_function = preprocess_input,
        rotation_range = 90,
        horizontal_flip = True,
        vertical_flip = True
        )

train_generator = train_datagen.flow_from_directory(TRAIN_DIR,
                                                    target_size = (HEIGHT, WIDTH),
                                                    batch_size = BATCH_SIZE)

#Generate validation data
val_datagen = ImageDataGenerator(
        preprocessing_function = preprocess_input)

val_generator = val_datagen.flow_from_directory(VAL_DIR,
                                                target_size = (HEIGHT, WIDTH),
                                                batch_size = BATCH_SIZE)

#Set up the fine tune model
def build_finetune_model(base_model, dropout, fc_layers, num_classes):
    for layer in base_model.layers:
        layer.trainable = False
        
    x = base_model.output
    x = Flatten()(x)
    for fc in fc_layer:
        #New FC layer, random init
        x = Dense(fc, activation = 'relu')(x)
        x = Dropout(dropout)(x)
    
    # New softmax layer
    predictions = Dense(num_classes, activation = 'softmax')(x)
    
    finetune_model = Model(inputs = base_model.input, outputs = predictions)
    
    return finetune_model

#Set up the class list and FC layers
class_list = ["Declaration", "Form", "Personal_info"]
FC_LAYERS = [1024,1024]
dropout = 0.5

#Instantiate the finetune model with the fc layers
finetune_model = build_finetune_model(base_model,
                                      droupout = dropout,
                                      fc_layer = FC_LAYERS,
                                      num_classes = len(class_list))


# =============================================================================
# #train the model
# =============================================================================

#set the hyperprameters
NUM_EPOCHS = 10
BATCH_SIZE = 32
num_train_images = 1100
num_val_images = 475

adam = Adam(lr = 0.00001)
finetune_model.compile(adam, loss = 'categorical_crossentropy', metrics = ['accuracy'])

filepath = "./checkpoints/" + "ResNet50" + "_model_weights.h5"
checkpoint = ModelCheckpoint(filepath, monitor = ["val_acc"], verbose = 1, mode = 'max')
callbacks_list = [checkpoint]

history = finetune_model.fit_generator(train_generator, epochs = NUM_EPOCHS, workers = 8, 
                                       steps_per_epoch = num_train_images // BATCH_SIZE,
                                       validation_data = val_generator,
                                       validation_steps = num_val_images // BATCH_SIZE,
                                       shuffle = True, callbacks = callbacks_list)

# =============================================================================
# End of Training
# =============================================================================

# Plot the train/validation acc/loss vs epoch
import matplotlib.pyplot as plt

# Plot the training and validation loss + accuracy
def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(epochs, acc, label = "train_acc")
    plt.plot(epochs, val_acc, label = "val_acc")
    plt.plot(epochs, loss, label = "train_loss")
    plt.plot(epochs, val_loss, label = "val_loss")
    plt.title("Training/Validation Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc = "lower left")
    
    plt.savefig('acc&loss_vs_epochs.png')
    
plot_training(history)

#print(history.history.keys())

# =============================================================================
# Test on test set
# =============================================================================

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from os import listdir
from os.path import isfile, join

finetune_model.load_weights("H:\WIP\Image_Classification\scripts\checkpoints\ResNet50_model_weights.h5")
adam = Adam (lr = 0.00001)
finetune_model.compile(adam, loss = 'categorical_crossentropy', metrics= ['accuracy'])

def model_predict(filepath):
    # load an image from file
    image = load_img (filepath, target_size = (300, 300))
    #print (image.format)
    #print (image.mode)
    #print (image.size)
    #image.show()
    #convert the image pixels to a numpy array
    image = img_to_array(image)
    #reshape data for the model
    image = image.reshape(1, image.shape[0], image.shape[1], image.shape[2])
    #prepare the image for the model
    image = preprocess_input(image)
    #predict the probability aross all output classes
    yhat = finetune_model.predict(image)
    #get the index of the maximum value
    label = np.argmax(yhat)
    return label

mypath = "H:\\WIP\\Image_Classfication\\scripts\\test\\form"
src_path = [join(mypath,f) for f in listdirD(mypath) if isfile(join(mypath, f))]

result =[]
for i in src_path:
    result.append(model_predict(i))        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        