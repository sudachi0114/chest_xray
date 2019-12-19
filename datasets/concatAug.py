
import os, shutil

cwd = os.getcwd()

train_dir = os.path.join(cwd, "train")

class_list = os.listdir(train_dir)
ignore_files = ['.DS_Store']
for fname in ignore_files:
    if fname in class_list:
        class_list.remove(fname)
class_list = sorted(class_list)

# -----
save_loc = os.path.join(cwd, "train_with_aug")
os.makedirs(save_loc, exist_ok=True)

save_0_loc = os.path.join(save_loc, class_list[0])
os.makedirs(save_0_loc, exist_ok=True)

save_1_loc = os.path.join(save_loc, class_list[1])
os.makedirs(save_1_loc, exist_ok=True)


def mv(src_dir, file_list, dist_dir, param=None):

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



train_0_dir = os.path.join(train_dir, class_list[0])
train_0_list = os.listdir(train_0_dir)
print(train_0_dir)
print("get {} data".format(len(train_0_list)))

print("move.....")
mv(train_0_dir, train_0_list, save_0_loc)
print("    Done.")


train_1_dir = os.path.join(train_dir, class_list[1])
train_1_list = os.listdir(train_1_dir)
print(train_1_dir)
print("get {} data".format(len(train_1_list)))

print("move.....")
mv(train_1_dir, train_1_list, save_1_loc)
print("    Done.")


# ----------

for i in range(2):
    print("その {} ----------".format(i))
    auged_dir = os.path.join(cwd, "auged_{}".format(i))

    auged_0_dir = os.path.join(auged_dir, class_list[0])
    auged_0_list = os.listdir(auged_0_dir)
    print(auged_0_dir)
    print("get {} data".format(len(auged_0_list)))

    print("move.....")
    mv(auged_0_dir, auged_0_list, save_0_loc, param=i)
    print("    Done.")

    auged_1_dir = os.path.join(auged_dir, class_list[1])
    auged_1_list = os.listdir(auged_1_dir)
    print(auged_1_dir)
    print("get {} data".format(len(auged_1_list)))

    print("move.....")
    mv(auged_1_dir, auged_1_list, save_1_loc, param=i)
    print("    Done.")



