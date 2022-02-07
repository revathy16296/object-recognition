import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils 
import os


def configure_gpu(memory_limit=1024):
    gpus = tf.config.experimental.list_physical_devices("GPU")
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)

        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit)])
    except RuntimeError as e:
        print(e)

def training_model(train_images, train_labels, test_images, test_labels):
    
    train_images = train_images/255.0
    test_images = test_images/255.0

    x1 = tf.expand_dims(train_images,3)
    y1 =tf.expand_dims(train_labels,1)
    x2 = tf.expand_dims(test_images,3)
    y2 = tf.expand_dims(test_labels,1)

    sp1, sp2 = tf.split(x1,num_or_size_splits=2, axis=0)
    x11 = tf.concat((sp1, sp2), axis=3).numpy()
    sp1, sp2 = tf.split(y1, num_or_size_splits=2, axis=0)
    y11 = tf.concat((sp1, sp2), axis=1).numpy()
    sp1, sp2 = tf.split(x2, num_or_size_splits=2, axis=0)
    x12 = tf.concat((sp1, sp2), axis=3).numpy()
    sp1, sp2 = tf.split(y2, num_or_size_splits=2, axis=0)
    y12 = tf.concat((sp1, sp2), axis=1).numpy()
    y11 = np.sum(y11, axis=1)
    y12 = np.sum(y12, axis=1)
    n_classes = 19
    Y_train = np_utils.to_categorical(y11, n_classes) 
    Y_test = np_utils.to_categorical(y12, n_classes) 


    model = tf.keras.Sequential(
        [
            tf.keras.Input(shape=(28,28,2)),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(n_classes, activation="softmax"),
        ]
    )

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(x11, Y_train, batch_size=32, epochs=5, verbose=2, validation_data=(x12, Y_test))

    save_dir = "/results/"
    model_name = 'keras_mnist5.h5'
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    return model_path

def model_inference(model_path,image1,image2):
    image1 = np.expand_dims(image,2)
    image2 = np.expand_dims(image2,2)
    image = np.concatenate ((image1, image2), axis = 2)
    image = np.expand_dims(image,0)
    my_model = model_path 
    detector = keras.models.load_model(my_model)
    out = detector(image))
    return np.argmax(out[0])

if __name__ == "__main__":
    configure_gpu(516)
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data() 
    model_path = training_model(train_images, train_labels,test_images, test_labels)
    out = model_inference(model_path,test_image[25],test_image[863])
    print(out)
