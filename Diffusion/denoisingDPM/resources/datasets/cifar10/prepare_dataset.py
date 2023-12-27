import pickle
from os import listdir
from os.path import isfile, join
from PIL import Image
from os import mkdir, rename
from os.path import exists

root_folder = "cifar10_dataset"
train_batches = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]

def unpickle(filename):
    print(f"processing file: {filename}")
    basename = filename.split("/")
    basename = basename[len(basename)-1]
    if basename in train_batches:
        is_testdata = False
        is_all_images = True
    elif basename == "test_batch":
        is_testdata = True
        is_all_images = True
    elif basename == "batches.meta":
        is_testdata = False
        is_all_images = False
    else:
        print(f"Skipping file: {basename}")
        return None, None, None
    
    with open(filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')

    return dict, is_all_images, is_testdata

allfiles = [f for f in listdir(root_folder) if isfile(join(root_folder, f))]

training_folder = join(root_folder, "train")
if not exists(training_folder):
    print(f"Creating training folder at: {training_folder}")
    mkdir(training_folder)
testing_folder = join(root_folder, "test")
if not exists(testing_folder):
    print(f"Creating testing folder at: {testing_folder}")
    mkdir(testing_folder)

image_idx = 0
for filename in allfiles:
    data_dict, is_all_images, is_testdata = unpickle(join(root_folder, filename))

    if data_dict is None:
        continue

    print(data_dict.keys())

    if is_all_images and (not is_testdata):
        print("This batch is for TRAINING")
        images = data_dict[b"data"]
        labels = data_dict[b"labels"]
        for i in range(len(labels)):
            img_lbl = labels[i]
            # print(f"Label for this image is: {img_lbl}")
            img = (images[i].reshape((3,32,32))).transpose((1,2,0))
            pil_image = Image.fromarray(img)
            # pil_image.save("image2check.png")
            class_folder = join(training_folder, str(img_lbl))
            if not exists(class_folder):
                print(f"Creating Class folder: {class_folder}")
                mkdir(class_folder)
            
            image_foldername = join(class_folder, f"image_{image_idx}.png")
            image_idx += 1
            pil_image.save(image_foldername)
    elif is_all_images and is_testdata:
        print("This batch is for TESTING")
        images = data_dict[b"data"]
        labels = data_dict[b"labels"]
        for i in range(len(labels)):
            img_lbl = labels[i]
            # print(f"Label for this image is: {img_lbl}")
            img = (images[i].reshape((3,32,32))).transpose((1,2,0))
            pil_image = Image.fromarray(img)
            # pil_image.save("image2check.png")
            class_folder = join(testing_folder, str(img_lbl))
            if not exists(class_folder):
                print(f"Creating Class folder: {class_folder}")
                mkdir(class_folder)
            
            image_foldername = join(class_folder, f"image_{image_idx}.png")
            image_idx += 1
            pil_image.save(image_foldername)
    else:
        lbl_names = data_dict[b'label_names']

# renaming the folders
for idx, lbl_map in enumerate(lbl_names):
    # train
    src = f"{training_folder}/{idx}"
    dst = f"{training_folder}/{lbl_map.decode('utf-8')}"
    rename(src, dst)
    # test
    src = f"{testing_folder}/{idx}"
    dst = f"{testing_folder}/{lbl_map.decode('utf-8')}"
    rename(src, dst)