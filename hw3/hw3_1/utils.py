# -*- coding: utf-8 -*-
"""
Created on Sat May 26 17:03:32 2018

@author: zhewei
"""

from skimage import io
from skimage.transform import resize
import numpy as np
import random
import os

imgs_dir = os.path.join('faces', '*.jpg')
def read_imgs(imgs_dir):
    #your path e.g. col_dir = 'cats/*.jpg'
    #creating a collection with the available images
    print("Loading images ...")
    save_file = 'resized_imgs.npy'
    if os.path.exists(save_file):
        imgs = np.load(save_file)
    else:
        imgs = io.imread_collection(imgs_dir)
        imgs = np.array(imgs).astype('float32')
        np.save(save_file, imgs)
    print("Done !")
    return imgs

class Dataset(object):
    def __init__(self, data, width=64, height=64, max_value=255, channels=3):
        # Record image specs
        self.IMAGE_WIDTH = width
        self.IMAGE_HEIGHT = height
        self.IMAGE_MAX_VALUE = float(max_value)
        self.CHANNELS = channels
        self.shape = len(data), self.IMAGE_WIDTH, self.IMAGE_HEIGHT, self.CHANNELS

        # Resize if images are of different size
        if data.shape[1] != self.IMAGE_HEIGHT or data.shape[2] != self.IMAGE_WIDTH:
            data = self.image_resize(data, self.IMAGE_HEIGHT, self.IMAGE_WIDTH)

        # Store away shuffled data
        index = list(range(len(data)))
        random.shuffle(index)
        self.data = data[index]
        
    def image_resize(self, dataset, newHeight, newWidth):
        """Resizing an image if necessary"""
        print("Resizing images ...")
        images_resized = []
        for image in dataset:
            temp = resize(image / self.IMAGE_MAX_VALUE, [64, 64], mode='reflect')
            images_resized.append(temp)
        print("Done !")
        
        save_file = 'resized_imgs.npy'
        if not os.path.exists(save_file):
            print("Saving as npy file ...")
            np.save(save_file, images_resized)
        return np.array(images_resized).astype('float32')

    def get_batches(self, batch_size):
        """Pulling batches of images and their labels"""
        current_index = 0
        # Checking there are still batches to deliver
        while current_index < len(self.data):
            if current_index + batch_size > len(self.data):
                batch_size = len(self.data) - current_index
            data_batch = self.data[current_index:current_index + batch_size]
            #print("current_index: {}, batch_size: {}".format(current_index, batch_size))
            current_index += batch_size
            yield (data_batch - 0.5)

    
if __name__ == '__main__':
    pass
