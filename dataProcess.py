import os
import shutil
import random

def split_dataset(source_dir, dest_dir, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    classes = os.listdir(source_dir)
    for cls in classes:
        cls_dir = os.path.join(source_dir, cls)
        images = os.listdir(cls_dir)
        random.shuffle(images)

        n_total = len(images)
        n_train = int(train_ratio * n_total)
        n_val = int(val_ratio * n_total)

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train+n_val]
        test_imgs = images[n_train+n_val:]

        for split_name, split_imgs in zip(['train', 'val', 'test'], [train_imgs, val_imgs, test_imgs]):
            split_dir = os.path.join(dest_dir, split_name, cls)
            os.makedirs(split_dir, exist_ok=True)
            for img in split_imgs:
                shutil.copy(os.path.join(cls_dir, img), os.path.join(split_dir, img))

source = "data/PetImages"
dest = "data/smDatasetSplit"
split_dataset(source, dest)
