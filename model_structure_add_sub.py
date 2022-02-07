import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate
from tensorflow import keras
import os
import sys


def configure_gpu(memory_limit=1024):
    gpus = tf.config.experimental.list_physical_devices("GPU")
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)

        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
    except RuntimeError as e:
        print(e)

def model_training(train_images, test_images, train_labels, test_labels):

    sp1, sp2 = tf.split(train_images,num_or_size_splits=2, axis=0)
    train1 = tf.concat((sp1, sp2), axis=2).numpy()
    train1 = np.expand_dims(train1,axis =3)
    sp1, sp2 = tf.split(test_images, num_or_size_splits=2, axis=0)
    test1 = tf.concat((sp1, sp2), axis=2).numpy()
    test1 = np.expand_dims(test1,axis =3)
    #making addition and subtraction dataset frm labels
    sp1, sp2 = tf.split(train_labels, num_or_size_splits=2)
    train2 = np.add(sp1,sp2)
    train21 = np.subtract(sp1,sp2)
    sp1, sp2 = tf.split(test_labels, num_or_size_splits=2)
    test2 = np.add(sp1,sp2)
    test21 = np.subtract(sp1,sp2)

    y_out_testa = keras.utils.to_categorical(test2, num_classes=19) 
    y_out_testb = keras.utils.to_categorical(test21, num_classes=19) 

    y_out_a = keras.utils.to_categorical(train2, num_classes=19) 
    y_out_b = keras.utils.to_categorical(train21, num_classes=19) 
    x =tf.keras.Input(shape=(28,56,1)),
    x = keras.layers.Conv2D(32, 3, activation="relu")(input)
    x = keras.layers.Conv2D(64, 3, activation="relu")(x)
    x = keras.layers.MaxPooling2D(3)(x)
    x = keras.layers.Conv2D(64, 3, activation="relu")(x)
    x = keras.layers.Conv2D(32, 3, activation="relu")(x)
    x = keras.layers.GlobalMaxPooling2D()(x)

    out_a = keras.layers.Dense(19, activation='softmax', name='add')(x)
    out_b = keras.layers.Dense(19, activation='softmax',name = 'sub')(x)

    encoder = keras.Model( inputs = input, outputs = [out_a, out_b], name="encoder")
    encoder.compile(
        loss = {
            "add": tf.keras.losses.CategoricalCrossentropy(),
            "sub": tf.keras.losses.CategoricalCrossentropy(),
        },

        metrics = {
            "add": 'accuracy',
            "sub": 'accuracy',
        },

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    )

    encoder.fit(train1, [y_out_a, y_out_b],epochs=10, batch_size=32)

  
    save_dir = "/results_ad/"
    model_name = 'keras_mnist5.h5'
    model_path = os.path.join(save_dir, model_name)
    encoder.save(model_path)
    return model_path

def inference_add_sub(image1,image2,a):

    configure_gpu(516)
    image = np.concatenate ((image1, image2), axis = 1)
    image = np.expand_dims(image,0)
    my_model = '/home/rvenugopal/Downloads/keras_mnist5_add.h5'
    detector = keras.models.load_model(my_model)
    out = detector(image)
    if (a == 'sum'):
        class_names_add = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18']
        out1 = class_names_add[np.argmax(out[0])]
    else:
        class_names_sub = ['0','1','2','3','4','5','6','7','8','9','-9','-8','-7','-6','-5','-4','-3','-2','-1']
        out1 = class_names_sub[np.argmax(out[1])]

if __name__ == '__main__':

    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data() 
    train_images = train_images/255.0
    test_images = test_images/255.0
    test_labels = test_labels.astype("int8")
    train_labels = train_labels.astype("int8")

    #trainig model returning saved model path
    model_path = model_training(train_images, test_images, train_labels, test_labels)

    a = str(sys.argv[1])#sum or subtraction using cla
    out = inference_add_sub (test_images[68],test_images[100],a) 
    print(out)