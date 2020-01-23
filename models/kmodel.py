
# kaggle で test_accuracy 86% を取っていたモデル


import os, sys
sys.path.append(os.pardir)

import time
import numpy as np
np.random.seed(seed=114)

import tensorflow as tf
import keras
from keras import backend as K
config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction=0.5
sess = tf.Session(config=config)
K.set_session(sess)

print("TensorFlow version is ", tf.__version__)
print("Keras version is ", keras.__version__)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
from utils.model_handler import ModelHandler


# define -----
batch_size = 16  # my & model2: 16 / model1: 32  <= x (test 余る)
input_size = 150  # my: 224 / model1: 64 / model2: 150
channel = 3
target_size = (input_size, input_size)
input_shape = (input_size, input_size, channel)
set_epochs = 20  # my: 40 / model1: 10 / model2: 20


def model1():

    # cf: https://www.kaggle.com/sanwal092/intro-to-cnn-using-keras-to-predict-pneumonia
    #   > The testing accuracy is : 86.39071487263763 %
    #   <= dataset はそのままに
    #       batch_size を 32 にして test を行なっている
    #           test data の総数は 234 + 390 = 624 枚
    #           batch_size を 32 に設定すると
    #           624 / 32 = 19.5
    #           32 * 19 = 608
    #           624 - 608 = 16枚余る
    #               <= この 16枚 test していない可能性がある..
    #   > test_accu = cnn.evaluate_generator(test_set,steps=624)
    #       <= 違うか
    #           32 * 624 step test しているので
    #           19 step で 1周ということは
    #           32.84.. 回 test している ??!!

    # let's build the CNN model
    model = Sequential()

    #Convolution
    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
    #Pooling
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # 2nd Convolution
    model.add(Conv2D(32, (3, 3), activation="relu"))
    # 2nd Pooling layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Flatten the layer
    model.add(Flatten())
    # Fully Connected Layers
    model.add(Dense(activation='relu', units=128))
    model.add(Dense(activation='softmax',  # 'sigmoid'
                    units=2))

    # Compile the Neural network
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', # 'binary_crossentropy'
                  metrics = ['accuracy'])

    return model


def model2():

    # cf: https://www.kaggle.com/kosovanolexandr/keras-nn-x-ray-predict-pneumonia-86-54
    #   > acc: 87.34%
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2))  # 1
    model.add(Activation('softmax'))  # sigmoid

    model.compile(loss='categorical_crossentropy',  # 'binary_crossentropy'
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    return model
    
def main():

    cwd = os.getcwd()
    prj_root = os.path.dirname(cwd)

    data_dir = os.path.join(prj_root, "datasets")

    use_da_data = False
    increase_val = False
    print( "\nmode: Use Augmented data: {} | increase validation data: {}".format(use_da_data, increase_val) )

    # First define original train_data only as train_dir
    train_dir = os.path.join(data_dir, "train")
    if (use_da_data == True) and (increase_val == False):
        # with_augmented data (no validation increase)
        train_dir = os.path.join(data_dir, "train_with_aug")
    validation_dir = os.path.join(data_dir, "val")  # original validation data

    # pair of decreaced train_data and increased validation data
    if (increase_val == True):
        train_dir = os.path.join(data_dir, "red_train")
        if (use_da_data == True):
            train_dir = os.path.join(data_dir, "red_train_with_aug")
        validation_dir = os.path.join(data_dir, "validation")

    test_dir = os.path.join(data_dir, "test")

    print("\ntrain_dir: ", train_dir)
    print("validation_dir: ", validation_dir)


    # data load ----------
    data_gen = ImageDataGenerator(rescale=1./255)

    train_generator = data_gen.flow_from_directory(train_dir,
                                                   target_size=target_size,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   class_mode='categorical')
                                                           
    validation_generator = data_gen.flow_from_directory(validation_dir,
                                                        target_size=target_size,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        class_mode='categorical')
                                                           
    test_generator = data_gen.flow_from_directory(test_dir,
                                                  target_size=target_size,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  class_mode='categorical')

    data_checker, label_checker = next(train_generator)

    print("train data shape (in batch): ", data_checker.shape)
    print("train label shape (in batch): ", label_checker.shape)
    # print("validation data shape:", validation_data.shape)
    # print("validation label shape:", validation_label.shape)
    # print("test data shape:", test_data.shape)
    # print("test label shape:", test_label.shape)



    # build model ----------
    # mh = ModelHandler(input_size, channel)
    # model = mh.buildMyModel()

    # model = model1()
    model = model2()

    model.summary()


    # instance EarlyStopping -----
    es = EarlyStopping(monitor='val_loss',
                       # monitor='val_accuracy',
                       patience=5,
                       verbose=1,
                       restore_best_weights=True)


    

    print("\ntraining sequence start .....")
    steps_per_epoch = train_generator.n // batch_size
    validation_steps = validation_generator.n // batch_size
    print(steps_per_epoch, " [steps / epoch]")
    print(validation_steps, " (validation steps)")                

    start = time.time()
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=steps_per_epoch,
                                  epochs=set_epochs,
                                  validation_data=validation_generator,
                                  validation_steps=validation_steps,
                                  callbacks=[es],
                                  verbose=1)
    print( "elapsed time (for train): {} [sec]".format(time.time() - start) )


    print("\nevaluate sequence...")
    test_steps = test_generator.n // batch_size
    eval_res = model.evaluate_generator(test_generator,
                                        steps=test_steps,  # 624 ?!
                                        verbose=1)

    print("result loss: ", eval_res[0])
    print("result score: ", eval_res[1])


    # confusion matrix -----
    print("\nconfusion matrix")
    pred = model.predict_generator(test_generator,
                                   steps=test_steps,
                                   verbose=3)

    test_label = []
    for i in range(test_steps):
        _, tmp_tl = next(test_generator)
        if i == 0:
            test_label = tmp_tl
        else:
            test_label = np.vstack((test_label, tmp_tl))    

    idx_label = np.argmax(test_label, axis=-1)  # one_hot => normal
    idx_pred = np.argmax(pred, axis=-1)  # 各 class の確率 => 最も高い値を持つ class
    
    cm = confusion_matrix(idx_label, idx_pred)

    # Calculate Precision and Recall
    tn, fp, fn, tp = cm.ravel()


    print("  | T  | F ")
    print("--+----+---")
    print("N | {} | {}".format(tn, fn))
    print("--+----+---")
    print("P | {} | {}".format(tp, fp))

    # 適合率 (precision):
    precision = tp/(tp+fp)
    print("Precision of the model is {}".format(precision))

    # 再現率 (recall):
    recall = tp/(tp+fn)
    print("Recall of the model is {}".format(recall))



if __name__ == '__main__':
    main()
