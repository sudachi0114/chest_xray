
import os, shutil

cwd = os.getcwd()

train_dir = os.path.join(cwd, "train")
val_dir = os.path.join(cwd, "val")

class_list = os.listdir(train_dir)
ignore_files = ['.DS_Store']
for fname in ignore_files:
    if fname in class_list:
        class_list.remove(fname)
class_list = sorted(class_list)


# train の 500枚を validation に分け与える
# validation_sep = 500

# train の 500枚を validation に分け与える
validation_0_sep = 150
validation_1_sep = 350

# increced validation data -----
validation_dir = os.path.join(cwd, "validation")
os.makedirs(validation_dir, exist_ok=True)

validation_0_dir = os.path.join(validation_dir, class_list[0])
os.makedirs(validation_0_dir, exist_ok=True)

validation_1_dir = os.path.join(validation_dir, class_list[1])
os.makedirs(validation_1_dir, exist_ok=True)

    
# reduced train data -----
red_train_dir = os.path.join(cwd, "red_train")
os.makedirs(validation_dir, exist_ok=True)

red_train_0_dir = os.path.join(red_train_dir, class_list[0])
os.makedirs(red_train_0_dir, exist_ok=True)

red_train_1_dir = os.path.join(red_train_dir, class_list[1])
os.makedirs(red_train_1_dir, exist_ok=True)



def copy(src_dir, file_list, dist_dir, param=None):

    for pic_name in file_list:
        copy_src = os.path.join(src_dir, pic_name)
        if param is not None:
            fname, ext = pic_name.rsplit('.', 1)
            fname = "{}_".format(param) + fname
            pic_name = fname + "." + ext
            copy_dst = os.path.join(dist_dir, pic_name)
        else:
            copy_dst = os.path.join(dist_dir, pic_name)
        shutil.copy(copy_src, copy_dst)



def main():

    # class 0 ==========
    train_0_dir = os.path.join(train_dir, class_list[0])
    print(train_0_dir)
    train_0_list = os.listdir(train_0_dir)
    print("get {} data".format(len(train_0_list)))
    train_0_list = sorted(train_0_list)
    train_0_amount = len(train_0_list)

    #sep = train_0_amount - validation_sep
    sep = train_0_amount - validation_0_sep

    red_train_0_list = train_0_list[:sep]
    inc_validation_0_list = train_0_list[sep:]
    print("reduced train amount: ", len(red_train_0_list))
    print("validation amount: ", len(inc_validation_0_list) + 8)

    # file copy -----
    print("copy.....")
    # train_dir にある red_train_list の画像を red_train_0_dir へ移動
    copy(train_0_dir, red_train_0_list, red_train_0_dir)
    print("    Done.")

    print("copy.....")
    # train_dir にある red_train_list の画像を validaiton_0_dir へ移動
    copy(train_0_dir, inc_validation_0_list, validation_0_dir)
    print("    Done.")


    # class 1 ==========
    train_1_dir = os.path.join(train_dir, class_list[1])
    print(train_1_dir)
    train_1_list = os.listdir(train_1_dir)
    print("get {} data".format(len(train_1_list)))
    train_1_list = sorted(train_1_list)
    train_1_amount = len(train_1_list)

    #sep = train_1_amount - validation_sep
    sep = train_1_amount - validation_1_sep

    red_train_1_list = train_1_list[:sep]
    inc_validation_1_list = train_1_list[sep:]
    print("reduced train amount: ", len(red_train_1_list))
    print("validation amount: ", len(inc_validation_1_list) + 8)

    # file copy -----
    print("copy.....")
    # train_dir にある red_train_list の画像を red_train_1_dir へ移動
    copy(train_1_dir, red_train_1_list, red_train_1_dir)
    print("    Done.")

    print("copy.....")
    # train_dir にある red_train_list の画像を validaiton_1_dir へ移動
    copy(train_1_dir, inc_validation_1_list, validation_1_dir)
    print("    Done.")


    # copy original_val data => validation
    val_0_dir = os.path.join(val_dir, class_list[0])
    val_0_list = os.listdir(val_0_dir)

    print("copy.....")
    copy(val_0_dir, val_0_list, validation_0_dir)
    print("    Done.")    

    
    val_1_dir = os.path.join(val_dir, class_list[1])
    val_1_list = os.listdir(val_1_dir)

    print("copy.....")
    copy(val_1_dir, val_1_list, validation_1_dir)
    print("    Done.")



def check():

    print("reduced train class 0's data amount:")
    print( len( os.listdir(red_train_0_dir) ) )

    print("reduced train class 1's data amount:")
    print( len( os.listdir(red_train_1_dir) ) )

    print("increased validation class 0's data amount:")
    print( len( os.listdir(validation_0_dir) ) )

    print("increased validation class 1's data amount:")
    print( len( os.listdir(validation_1_dir) ) )



if __name__ == "__main__":
    # main()
    check()
