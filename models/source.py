
# simple に転移学習モデルを組んで流すだけ

import os, sys
sys.path.append(os.pardir)

import time

from utils.img_utils import inputDataCreator
from utils.model_handler import ModelHandler

def main():

    cwd = os.getcwd()
    prj_root = os.path.dirname(cwd)

    data_dir = os.path.join(prj_root, "datasets")
    train_dir = os.path.join(data_dir, "train")
    validation_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")


    # data load ----------
    train_data, train_label = inputDataCreator(train_dir,
                                               224,
                                               normalize=True,
                                               one_hot=True,
                                               ch='gray')

    validation_data, validation_label = inputDataCreator(validation_dir,
                                                         224,
                                                         normalize=True,
                                                         one_hot=True,
                                                         ch='gray')

    test_data, test_label = inputDataCreator(test_dir,
                                             224,
                                             normalize=True,
                                             one_hot=True,
                                             ch='gray')

    print("train data shape:", train_data.shape)
    print("train label shape:", train_label.shape)
    print("validation data shape:", validation_data.shape)
    print("validation label shape:", validation_label.shape)
    print("test data shape:", test_data.shape)
    print("test label shape:", test_label.shape)
    

    # build model ----------
    mh = ModelHandler()

    model = mh.buildTlearnModel(base='mnv2')

    model.summary()

    start = time.time()
    history = model.fit(train_data,
                        train_label,
                        batch_size=32,
                        epochs=20,
                        validation_data=(validation_data, validation_label),
                        verbose=1)
    print( "elapsed time (for train): {} [sec]".format(time.time() - start) )


    print("\nevaluate sequence...")

    eval_res = model.evaluate(test_data,
                              test_label,
                              #batch_size=10,
                              verbose=1)

    print("result loss: ", eval_res[0])
    print("result score: ", eval_res[1])



if __name__ == '__main__':
    main()
