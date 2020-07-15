import sys
from os.path import basename, dirname
import pickle



def load_batch(filenamepath):
    import os
    import gzip
    import numpy as np

    filename = basename(filenamepath)
    path = dirname(filenamepath)
    if filename.find('batch') == -1:
        return "", ""
    # if filename.find('labels') != -1:
    #     labelsfilename = filename
    #     imagesfilename = labelsfilename.replace('idx1', 'idx3')
    #     imagesfilename = imagesfilename.replace('labels', 'images')
    # elif filename.find('images') != -1:
    #     imagesfilename = filename
    #     labelsfilename = imagesfilename.replace('idx3', 'idx1')
    #     labelsfilename = labelsfilename.replace('images', 'labels')
    # else:
    #     return "", ""

    try:
        with open(filenamepath, 'rb') as imgpath:
            filedict = pickle.load(imgpath, encoding='bytes')
            labels = filedict[b'labels']
            images = filedict[b'data']
            num_of_labels = len(labels)
            num_of_images = len(images)
            if num_of_images != num_of_labels:
                return "", ""
            # first n are red, n-2n green, 2n-3n blue
            # converting to rgb
            rgblist = list()
            for img in images:
                sizeofimg = int(len(img) / 3)
                imgtransl=np.frombuffer(img, dtype=np.uint8, offset=0).reshape((3,sizeofimg))
                plainimgtransl=imgtransl.ravel(order='F')
                rgblist.append(plainimgtransl.tolist())

            return rgblist, labels
    except:
        print("Unexpected error loading mnist file:", sys.exc_info()[0])
        return None, None
