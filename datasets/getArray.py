
import os, time
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# define ----------
ignore_files = ['.DS_Store']

cwd = os.getcwd()

train_dir = os.path.join(cwd, "train")

class_list = os.listdir(train_dir)
# sheve ----------
for fname in ignore_files:
    if fname in class_list:
        class_list.remove(fname)
class_list = sorted(class_list)


train_0_dir = os.path.join(train_dir, class_list[0])
train_1_dir = os.path.join(train_dir, class_list[1])

validation_dir = os.path.join(cwd, "val")
validation_0_dir = os.path.join(validation_dir, class_list[0])
validation_1_dir = os.path.join(validation_dir, class_list[1])

test_dir = os.path.join(cwd, "test")
test_0_dir = os.path.join(test_dir, class_list[0])
test_1_dir = os.path.join(test_dir, class_list[1])

train_0_dir_list = sorted(os.listdir(train_0_dir))

# class 0 ----------
print(train_0_dir_list[0])

start = time.time()

img_list = []
img_size = 224
channel = 1
img_shape = (img_size, img_size)
for i in range(len(train_0_dir_list)):
    targf = os.path.join(train_0_dir, train_0_dir_list[i])

    pil_obj = Image.open(targf)
    pil_obj = pil_obj.resize(img_shape)
    #pil_obj.convert("L")
    #print(pil_obj)

    img_arr = np.array(pil_obj)
    #print(img_arr.shape)
    assert img_arr.shape == img_shape
    img_list.append(img_arr)

img_list = np.array(img_list)
print("\nimg_list shape: ", img_list.shape)

print("\nelaped time: {} [s]".format(time.time() - start))


# class 1 ----------
train_1_dir_list = sorted(os.listdir(train_1_dir))
img_1_list = []
for i in range(len(train_1_dir_list)):
    targf = os.path.join(train_1_dir, train_1_dir_list[i])

    pil_obj = Image.open(targf)
    pil_obj = pil_obj.resize(img_shape)
    pil_obj = pil_obj.convert("L")
    #print(pil_obj)

    img_arr = np.array(pil_obj)
    #print(img_arr.shape)
    assert img_arr.shape == img_shape, "Error {} => shape: {} | fname: {}".format(i, img_arr.shape, targf)
    img_1_list.append(img_arr)

img_1_list = np.array(img_1_list)
print("\nimg_list shape: ", img_1_list.shape)

print("\nelaped time1: {} [s]".format(time.time() - start))
"""
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(img_list[i], cmap='gray')
    plt.axis(False)
    plt.title(i)
plt.show()
"""
