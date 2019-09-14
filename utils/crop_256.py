import glob
import warnings
from tqdm import tqdm
import numpy as np
import pandas as pd
import skimage

def crop_and_save(impath):
    img = skimage.io.imread(impath)
    assert img.shape == (256, 1600, 3)
    for i, idx in enumerate(range(0, 1600-256+32, 256-32)):
        cimg = img[:, idx:idx+256, :]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            skimage.io.imsave(impath.replace('.jpg', '_c{}.jpg'.format(i)), cimg)

    labelpath = impath.replace('/train_images/', '/train_masks/').replace('.jpg', '.png')
    label = skimage.io.imread(labelpath)
    assert label.shape == (256, 1600)
    for i, idx in enumerate(range(0, 1600-256+32, 256-32)):
        clabel = label[:, idx:idx+256]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            skimage.io.imsave(labelpath.replace('.png', '_c{}.png'.format(i)), clabel)

def main():
    imlist = glob.glob('./data/train_images/*.jpg')
    for impath in tqdm(imlist):
        crop_and_save(impath)

if __name__ == '__main__':
    main()
