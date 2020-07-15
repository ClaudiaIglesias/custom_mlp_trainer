

# Load a CSV file
import sys
from csv import reader
from math import sqrt
from os.path import basename

from batch_reader import load_batch
from mnist_reader import load_mnist
import re

from datetime import datetime


def load_csv(filename):
    dataset = list()
    try:
        with open(filename, 'r') as file:
            csv_reader = reader(file)
            for row in csv_reader:
                if not row:
                    continue
                dataset.append(row)
        return dataset
    except:
        return None


def datasetoutputfirsttolastcolumn(mydataset):
    for row in mydataset:
        value = row.pop(0)
        row.append(value)
    return mydataset


# String to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# string to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = sorted(set(class_values))
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


# max and min value of each column
def dataset_minmax(dataset):
    minmax = list()
    stats = [[min(column), max(column)] for column in zip(*dataset)]
    return stats

def optional_output_column(mydataset, columnselected):
    last = -1
    first = 0
    if columnselected == first:
        column = last
    else:
        column = first
    outputs = ([row[column] for row in mydataset])
    setoutputs = set(outputs)
    n_outputs = len(setoutputs)
    if n_outputs == 1 or n_outputs > 20:
        return False
    else:
        return True

def guess_output_column(mydataset):
    last = -1
    first = 0
    outputslast = ([row[last] for row in mydataset])
    setoutputslast = set(outputslast)
    n_outputslast = len(setoutputslast)
    outputsfirst = ([row[0] for row in mydataset])
    setoutputsfirst = set(outputsfirst)
    n_outputsfirst = len(setoutputsfirst)
    # more than 1 output
    if (n_outputslast == 1):
        return first
    if (n_outputsfirst == 1):
        return last

    # no float as output
    for row in setoutputslast:
        floatvalue = float(row.strip())
        intvalue = float(int(row.strip()))
        # if floatvalue is not an integer
        if (floatvalue != intvalue):
            return first
    for row in setoutputsfirst:
        floatvalue = float(row.strip())
        intvalue = float(int(floatvalue))
        # if floatvalue is not an integer
        if (floatvalue != intvalue):
            return last
    return last

def guess_is_type_image(mydataset):
    inputfirst = len(mydataset[0])
    if (inputfirst > 20):
        sidesquare = sqrt(inputfirst - 1)
        intsidesquare = float(int(sidesquare))
        # if sidesquare is an integer
        if (sidesquare == intsidesquare):
            return 'luminance'
        sidesquare = sqrt((inputfirst - 1)/3)
        intsidesquare = float(int(sidesquare))
        # if sidesquare is an integer
        if (sidesquare == intsidesquare):
            return 'rgb'
    return False

# range 0-1
def normalize_dataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row) - 1):
            if minmax[i][1] != minmax[i][0]:
                row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
    
def showdataset(filenamepath):
    filename = basename(filenamepath)
    if re.compile('\.*csv').search(filename) is None and (filename.find('idx') != -1 or filename.find('batch') != -1):
        if filename.find('idx') != -1:
            images, labels = load_mnist(filenamepath)
            if images is None or labels is None:
                return None, None, None, None
            # to float
            i = 0
            for floatrow in images:
                floatrow.append(labels[i])
                i += 1
            return False, True, guess_is_type_image(images), images
        else:
            images, labels = load_batch(filenamepath)
            if images is None or labels is None:
                return None, None, None, None

            i = 0
            for floatrow in images:
                floatrow.append(labels[i])
                i += 1
            return False, True, guess_is_type_image(images), images

    else:
        mydataset = load_csv(filenamepath)
        if mydataset is None:
            return None, None, None, None
        outputcolumn = guess_output_column(mydataset)
        outputoptional = optional_output_column(mydataset, outputcolumn)
        isimage = guess_is_type_image(mydataset)
        if isimage == False:
            imageoptional = False
        else:
            imageoptional = True

        if outputcolumn == 0:
            mydataset = datasetoutputfirsttolastcolumn(mydataset)
        return outputoptional, imageoptional, isimage, mydataset

def readdataset(filenamepath, myevent, myapp, forceisimage=None, forceoutputcolumn=None):

    myevent.evreaddataset(5, None, None, myapp)

    filename = basename(filenamepath)
    if re.compile('\.*csv').search(filename) is None and filename.find('idx') != -1:
        images, labels = load_mnist(filenamepath)
        if images is None or labels is None:
            return None, None
        # to float
        myevent.evreaddataset(10, None, None, myapp)
        floatimages=list()
        for row in images:
            floatimages.append([float(col) for col in row])
        # normalize
        myevent.evreaddataset(30, None, floatimages, myapp)
        minmax = dataset_minmax(floatimages)
        i = 0
        for floatrow in floatimages:
            floatrow.append(labels[i])
            i += 1
        myevent.evreaddataset(50,  None, floatimages, myapp)
        normalize_dataset(floatimages, minmax)
        if forceisimage is None:
            isimage = True
        else:
            isimage = forceisimage
        myevent.evreaddataset(100, isimage, floatimages, myapp)
        return isimage, floatimages
    else:
        mydataset = load_csv(filenamepath)
        if mydataset is None:
            return None, None
        if forceoutputcolumn is None:
            outputcolumn = guess_output_column(mydataset)
        else:
            outputcolumn = forceoutputcolumn
        if forceisimage is None:
            isimage = guess_is_type_image(mydataset)
        else:
            isimage = forceisimage
        if outputcolumn == 0:
            mydataset = datasetoutputfirsttolastcolumn(mydataset)

        myevent.evreaddataset(10, isimage, mydataset, myapp)
        for i in range(len(mydataset[0]) - 1):
            str_column_to_float(mydataset, i)

        # convert class column to integers
        myevent.evreaddataset(30, isimage, mydataset, myapp)
        str_column_to_int(mydataset, len(mydataset[0]) - 1)

        # normalize input variables
        myevent.evreaddataset(50, isimage, mydataset, myapp)
        minmax = dataset_minmax(mydataset)

        myevent.evreaddataset(70, isimage, mydataset, myapp)
        normalize_dataset(mydataset, minmax)

        myevent.evreaddataset(100, isimage, mydataset, myapp)
        return isimage, mydataset

# def main():
#     # print command line arguments
#
#     for arg in sys.argv[1:]:
#         isimage, dataset = readdataset(arg)
#         print ("isimage {0},  rows {1}" .format(isimage, len(dataset)))
#
#
# if __name__ == '__main__':
#     main()
