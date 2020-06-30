from os.path import basename, dirname


def load_mnist(filenamepath):
    import os
    import gzip
    import numpy as np

    filename = basename(filenamepath)
    path = dirname(filenamepath)
    if filename.find('idx') == -1:
        return "", ""
    if filename.find('labels') != -1:
        labelsfilename = filename
        imagesfilename = labelsfilename.replace('idx1', 'idx3')
        imagesfilename = imagesfilename.replace('labels', 'images')
    elif filename.find('images') != -1:
        imagesfilename = filename
        labelsfilename = imagesfilename.replace('idx3', 'idx1')
        labelsfilename = labelsfilename.replace('images', 'labels')
    else:
        return "", ""

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, labelsfilename)

    images_path = os.path.join(path, imagesfilename)

    try:
        with gzip.open(labels_path, 'rb') as lbpath:
            lbpath.read(4)
            num_of_labels = int.from_bytes(lbpath.read(4), byteorder="big")

            labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=0)
            if len(labels) != num_of_labels:
                return None, None

        with gzip.open(images_path, 'rb') as imgpath:
            imgpath.read(4)
            num_of_images = int.from_bytes(imgpath.read(4), byteorder="big")
            if num_of_images != num_of_labels:
                return "", ""
            num_of_rows = int.from_bytes(imgpath.read(4), byteorder="big")
            num_of_colums = int.from_bytes(imgpath.read(4), byteorder="big")
            images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                                   offset=0).reshape(num_of_labels, num_of_rows * num_of_colums)
            imagesarr = list()
            for myarray in images:
                myarrayitem=list()
                for myitem in myarray:
                    myarrayitem.append(myitem)
                imagesarr.append(myarrayitem)

        return imagesarr, labels
    except:
        return None, None
