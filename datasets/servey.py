
import os

cwd = os.getcwd()


train_dir = os.path.join(cwd, "train")

class_list = os.listdir(train_dir)
ignore_files = ['.DS_Store']
for fname in ignore_files:
    if fname in class_list:
        class_list.remove(fname)
class_list = sorted(class_list)

train_0_dir = os.path.join(train_dir, class_list[0])
print(train_0_dir)
print( len(os.listdir(train_0_dir)) )

train_1_dir = os.path.join(train_dir, class_list[1])
print(train_1_dir)
print( len(os.listdir(train_1_dir)) )

""" train data
train_0: ../datasets/train/NORMAL
    1341
train_1: ../datasets/train/PNEUMONIA
    3875
    # 不均衡すぎる..
"""

validation_dir = os.path.join(cwd, "val")

validation_0_dir = os.path.join(validation_dir, class_list[0])
print(validation_0_dir)
print( len(os.listdir(validation_0_dir)) )

validation_1_dir = os.path.join(validation_dir, class_list[1])
print(validation_1_dir)
print( len(os.listdir(validation_1_dir)) )


""" validaiton data
validation_0: ../datasets/val/NORMAL
    8
validation_1: /datasets/val/PNEUMONIA
    8
    # どういうことなのか..
"""

test_dir = os.path.join(cwd, "test")

test_0_dir = os.path.join(test_dir, class_list[0])
print(test_0_dir)
print( len(os.listdir(test_0_dir)) )

test_1_dir = os.path.join(test_dir, class_list[1])
print(test_1_dir)
print( len(os.listdir(test_1_dir)) )


""" test data
test_0: ../datasets/test/NORMAL
    234
test_1: ../datasets/test/PNEUMONIA
    390
"""

""" まとめ
train     : 1300 * 2 classes (?)
validation: なし (train から split する)
test      : 200 * 2 classes
    と変形するのがいいかもしれない

    名前に系統があるが, 複数の要素が絡んでいるみたいで
    わかりづらいなあ..
        NORMAL の方は (無名|NORMAL2) + IM-人の番号-写真番号.jpeg (?)
            写真番号が加算ではなく、2つついているものもある..
        PNEUMONIA の方は person番号_(bacteria|virus)_番号.jpeg
            となっている。
            bacteria と virus の違いがわからない..
            その上 person の番号が NORMAL の番号と対応するのかも不明..
"""
