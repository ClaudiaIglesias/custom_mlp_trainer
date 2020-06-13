


from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.checkbox import CheckBox
from kivy.uix.widget import Widget
from kivy.graphics import *
from kivy.config import Config
from kivy.uix.textinput import TextInput
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.lang import Builder
from kivy.uix.slider import Slider
from kivy.uix.image import Image, AsyncImage
import numpy as np
from kivy.graphics.vertex_instructions import *
from random import seed
from random import random
from math import exp
from csv import reader
from random import randrange
from math import exp
from kivy.core.window import Window
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.tabbedpanel import *
from kivy.uix.image import Image
from kivy.uix.progressbar import ProgressBar
from kivy.clock import Clock, mainthread
from kivy.properties import ObjectProperty
from functools import wraps
import threading
import time
from kivy.event import EventDispatcher




numdata = 2
numoutputs = 1
n_outputs = 1
n_inputs = 2
list_hidden = list()
network = 2
n_folds = 5
l_rate = 0.3
n_epoch = 20
a = 0
b = 0
c = 0
d = 0
e = 1
g = 0
i = 0
j = 0
l = 0
valuepb = 0
accuracylabel =''
scores = list()
ti = TextInput(size=(40, 30), size_hint=(None, None))
ti2 = TextInput(size=(40, 30), size_hint=(None, None))
ti3 = TextInput(size=(40, 30), size_hint=(None, None))
progressbar = ProgressBar()

hiddenlayeronelabelone = Label(text='[b]  Hidden\nlayer one', markup=True, font_size=15, color=(1,1,1), size_hint=(None, None))
hiddenlayeronelabeltwo = Label(text='[b]  Hidden\nlayer two', markup=True, size_hint=(None,None), font_size=15 , color=(1,1,1))
hiddenlayeronelabelthree = Label(text='[b]   Hidden\nlayer three', markup=True,size_hint=(None,None), font_size=15, color=(1,1,1))
outputlabel = Label(text='Output', markup=True, font_size=17, size_hint=(None, None), color=(.85, .63, .27))
inputlabel = Label(text='Input', markup=True, font_size=17, size_hint=(None, None), color=(.69, 0, .83))

# Load a CSV file
def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			if not row:
				continue
			dataset.append(row)
	return dataset
# String to float
def str_column_to_float(dataset, column):
	for row in dataset:
		row[column] = float(row[column].strip())
#string to integer
def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup
#max and min value of each column
def dataset_minmax(dataset):
	minmax = list()
	stats = [[min(column), max(column)] for column in zip(*dataset)]
	return stats
#range 0-1
def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)-1):
			row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
#Split the data
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split
#Create the list of weights and biases
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    weightspernode=n_inputs
    for nodesinhidden in n_hidden:
        network.append ( [{'weights': [random() for i in range(weightspernode + 1)]} for i in range(nodesinhidden)] )
        weightspernode=nodesinhidden

    output_layer = [{'weights': [random() for i in range(nodesinhidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network

# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation

# Transfer neuron activation
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))

# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

# Calculate the derivative of an neuron output
def transfer_derivative(output):
    return output * (1.0 - output)

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# Update network weights with error
def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']

# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
    global valuepb
    global n_folds
    step = 100.0/(n_folds*n_epoch)
    for epoch in range(n_epoch):
        sum_error = 0
        valuepb += step
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

#Works with btnGo
def predict(network, row):
    outputs = forward_propagate(network, row)
    print(outputs)
    return outputs.index(max(outputs))

#accuracy
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return (correct / float(len(actual))) * 100.0

#ProgressBar
def layoutpbnext(self):
    global valuepb
    global progressbar
    progressbar.value = valuepb

trainclock = Clock.schedule_interval(layoutpbnext, 1 / 10)



# Backpropagation Algorithm With Stochastic Gradient Descent
def back_propagation(train, test, l_rate, n_epoch, n_hidden):
    global network

    n_inputs = len(train[0]) - 1
    n_outputs = len(set([row[-1] for row in train]))
    network = initialize_network(n_inputs, n_hidden, n_outputs)
    train_network(network, train, l_rate, n_epoch, n_outputs)
    predictions = list()
    for row in test:
        prediction = predict(network, row)
        predictions.append(prediction)
    return(predictions)




class MyApp(App):
    Input = ''

    #When Go is pressed
    def drawingvisualize(self, draw, *largs):
        global network, ti, ti2, ti3
        c = 0
        #Input = [float(ti.text), float(ti2.text), float(ti3.text)]

        # Input = [float(ti.text), float(ti2.text)]
        # predict(network, Input)
        #
        numoutputs = n_outputs
        numdata = n_inputs
        middlex = draw.width / 2
        middley = draw.height / 2 + 40
        space = numdata * 35 + (numdata - 1) * 25
        spaceoutput = numoutputs * 35 + (numoutputs - 1) * 25

        for x in range(numoutputs):
            value = network[-1][c]['weights'][-1]
            valuefloat = network[1][c]['output']
            print(value)
            with draw.canvas:
                Color(0.6/4 + abs(valuefloat)/1.33,1/4 + abs(valuefloat)/1.33,(0.8+((0.2/(numoutputs))*c))/4 + abs(valuefloat)/1.33)
                RoundedRectangle(size=(35, 35), pos=(middlex + 230, c * 50 + (middley - spaceoutput / 2)))
                Color(0, 0, 0)
                RoundedRectangle(size=(30, 30), pos=(middlex + 232.5, c * 50 + (middley+2.5 - spaceoutput / 2)), radius=[(6,6),(6,6),(6,6),(6,6)])
                biases = "{:.2f}".format(value)
                outputtext = "{:.2f}".format(valuefloat)
                draw.add_widget(Label(text='[b]Probability:    [/b]' + outputtext, markup=True, size_hint=(None, None),
                                      font_size=12, pos=(middlex + 290, c * 50 + (middley - 30 - spaceoutput / 2)),
                                      color=(1, 1, 0.83)))
                #draw.nodeslabel = Label(text=biases, font_size=10, pos=(middlex + 196, c * 50 + (middley-34 - spaceoutput / 2)),color=(1, 1, 1, .9))
            c = c + 1

        Color(1,1,1)
        inputtext = Label(text='[b]' + ti.text, markup = True, size=(30, 30), size_hint= (None,None), pos=(middlex, (middley - space / 2)), color=(1,1,1,1))
        inputtext2 = Label(text='[b]' + ti2.text, markup = True, size=(30, 30), size_hint= (None,None), pos=(middlex-250, 50 + (middley - 30 - space / 2)), color=(1,1,1, 1))
        #inputtext3 = Label(text='[b]' + ti3.text, markup = True, size=(30, 30), size_hint= (None,None), pos=(middlex-250, 50 * 2 + (middley - 30 - space / 2)), color=(1,1,1, 1))
        draw.add_widget(inputtext)
        draw.add_widget(inputtext2)
        #draw.add_widget(inputtext3)
        draw.remove_widget(ti)
        draw.remove_widget(ti2)
        #draw.remove_widget(ti3)

    # When Train is pressed
    def drawinglblblackkv(self, draw, titlelayout, *largs):
        global a, b, list_hidden, network, n_outputs, n_inputs, accuracylabel, progressbar
        global hiddenlayeronelabelthree, hiddenlayeronelabelone, hiddenlayeronelabeltwo, outputlabel, inputlabel
        a = 0
        b = 0
        c = 0
        d = 0
        e = 0
        g = 0
        i = 0
        j = 0
        l = 0

        draw.canvas.clear()

        numoutputs = n_outputs
        numdata = n_inputs
        space = numdata * 35 + (numdata - 1) * 25
        spaceone = list_hidden[0] * 35 + (list_hidden[0] - 1) * 25
        spaceoutput = numoutputs * 35 + (numoutputs - 1) * 25
        middlex = draw.width/2
        middley = draw.height/2 + 40
        middletitlex = titlelayout.width / 2
        middletitley = titlelayout.height / 2

        outputlabel.pos = (middletitlex + 202, middletitley - 60)
        inputlabel.pos = (middletitlex - 268, middletitley - 60)

        titlelayout.remove_widget(hiddenlayeronelabelone)
        titlelayout.remove_widget(hiddenlayeronelabeltwo)
        titlelayout.remove_widget(hiddenlayeronelabelthree)
        titlelayout.remove_widget(outputlabel)
        titlelayout.remove_widget(inputlabel)
        titlelayout.add_widget(outputlabel)
        titlelayout.add_widget(inputlabel)

        ti.pos = (90,(middley+5 - space / 2))
        draw.remove_widget(ti)
        ti2.pos = (90, 50 + (middley+5 - space / 2))
        draw.remove_widget(ti2)
        ti3.pos = (90, 2 * 50 + (middley+5 - space / 2))
        draw.remove_widget(ti3)

        if len(list_hidden) == 1:
            # hiddenlayerone
            valuefloat = 0
            linecoloroutput = 0
            linecolorinput = 0
            hiddenlayeronelabelone.pos = (middletitlex - 28, middletitley - 50)
            titlelayout.add_widget(hiddenlayeronelabelone)
            for x in range(list_hidden[0]):
                if valuefloat < abs(network[0][a]['weights'][2]):
                    valuefloat = abs(network[0][a]['weights'][2])
                e = 0
                i = 0
                for y in range(numdata):
                    if linecolorinput < abs(network[0][a]['weights'][i]):
                        linecolorinput = abs(network[0][a]['weights'][i])
                    i = i + 1
                for z in range(numoutputs):
                    if linecoloroutput < abs(network[1][e]['weights'][a]):
                        linecoloroutput = abs(network[1][e]['weights'][a])
                    e = e + 1
                a = a + 1
            a = 0
            for x in range(list_hidden[0]):
                with draw.canvas:
                    value = network[0][a]['weights'][-1]
                    colorlayerone = abs(network[0][a]['weights'][-1]) / valuefloat
                    #Color(0.59, 0.65+colorlayerone/7.69, 0.93, 0.5+colorlayerone/2)
                    #Color(0.6- colorlayerone/10, 0.78 + colorlayerone/33.33, 0.93- colorlayerone/3.85, colorlayerone/3.33 + 0.75)
                    #Color(0.49, .90 - ((0.14 / (list_hidden[0] - 1)) * a), 0.67 + ((0.06 / (list_hidden[0] - 1)) * a))
                    Color(1,1,1, 0.25 + colorlayerone/1.33)
                    RoundedRectangle(size=(35, 35), pos=(middlex, a * 50 + (middley - spaceone / 2)))
                    Color(0,0,0,1)
                    RoundedRectangle(size=(30, 30), pos=(middlex+2.5, a * 50 + (middley + 2.5 - spaceone / 2)), radius=[(6,6),(6,6),(6,6),(6,6)])
                    biases = "{:.2f}".format(value)
                    draw.nodeslabel = Label(text=biases, size_hint=(None,None), font_size=9, pos=(middlex-31.5, a * 50 + (middley-31.5 - spaceone / 2)),
                                            color=(colorlayerone+0.25, colorlayerone+0.25, colorlayerone+0.25, 1))
                    draw.add_widget(draw.nodeslabel)
                i = 0
                e = 0
                for y in range(numdata):
                    color = abs(network[0][a]['weights'][i]) / linecolorinput
                    with draw.canvas:
                        Color(color, color, color)
                        Line(points=[middlex - 205, i * 50 + (middley-10 - space / 2) + 30, middlex, a * 50 + (middley+20 - spaceone / 2) ])
                        i = i + 1
                for z in range(numoutputs):
                    color = abs(network[1][e]['weights'][a]) / linecoloroutput
                    with draw.canvas:
                        Color(color, color, color)
                        Line(points=[middlex+35, a * 50 + (middley  - spaceone / 2)+ 20, middlex+230,
                                     e * 50 + (middley+ 20 - spaceoutput / 2) ])
                    e = e + 1
                a = a + 1

        if len(list_hidden) == 2:
            # hiddenlayerone
            spacetwo = list_hidden[1] * 35 + (list_hidden[1] - 1) * 25
            valuefloat = 0
            linecoloroutput = 0
            linecolorinput = 0
            hiddenlayeronelabelone.pos = (middletitlex - 108, middletitley - 50)
            titlelayout.add_widget(hiddenlayeronelabelone)
            for x in range(list_hidden[0]):
                if valuefloat < abs(network[0][a]['weights'][-1]):
                    valuefloat = abs(network[0][a]['weights'][-1])
                e = 0
                i = 0
                for y in range(numdata):
                    if linecolorinput < abs(network[0][a]['weights'][i]):
                        linecolorinput = abs(network[0][a]['weights'][i])
                    i = i + 1
                a = a + 1
            a = 0
            for x in range(list_hidden[0]):
                with draw.canvas:
                    value = network[0][a]['weights'][-1]
                    colorlayerone = abs(network[0][a]['weights'][-1]) / valuefloat
                    Color(1, 1, 1, 0.25 + colorlayerone / 1.33)
                    RoundedRectangle(size=(35, 35), pos=(middlex-80, a * 50 + (middley - spaceone / 2)))
                    Color(0,0,0,1)
                    RoundedRectangle(size=(30, 30), pos=(middlex + 2.5 - 80, a * 50 + (middley + 2.5 - spaceone / 2)),
                                     radius=[(6, 6), (6, 6), (6, 6), (6, 6)])
                    biases = "{:.2f}".format(value)
                    draw.nodeslabel = Label(text=biases,size_hint=(None,None),font_size=10, pos=(middlex-111.5, a * 50 + (middley-31.5 - spaceone / 2)),
                                            color=(1, 1, 1, .9))
                    draw.add_widget(draw.nodeslabel)
                i = 0
                e = 0
                for y in range(numdata):
                    color = abs(network[0][a]['weights'][i]) / linecolorinput
                    with draw.canvas:
                        Color(color, color, color)
                        Line(points=[middlex - 205, i * 50 + (middley-10 - space / 2) + 30, middlex-80, a * 50 + (middley + 20 - spaceone / 2) ])
                        i = i + 1
                a = a + 1

            # hiddenlayertwo
            valuefloat = 0
            linecoloroutput = 0
            linecolorinput = 0
            d = 0
            a = 0
            hiddenlayeronelabeltwo.pos = (middletitlex + 52, middletitley - 50)
            titlelayout.add_widget(hiddenlayeronelabeltwo)
            for x in range(list_hidden[1]):
                if valuefloat < abs(network[1][d]['weights'][-1]):
                    valuefloat = abs(network[1][d]['weights'][-1])
                e = 0
                j = 0
                for y in range(list_hidden[0]):
                    if linecolorinput < abs(network[1][d]['weights'][j]):
                        linecolorinput = abs(network[1][d]['weights'][j])
                    j = j + 1
                for z in range(numoutputs):
                    if linecoloroutput < abs(network[2][e]['weights'][d]):
                        linecoloroutput = abs(network[2][e]['weights'][d])
                    e = e + 1
                d = d + 1
            d = 0
            for x in range(list_hidden[1]):
                with draw.canvas:
                    value = network[1][d]['weights'][-1]
                    colorlayerone = abs(network[1][d]['weights'][-1]) / valuefloat
                    Color(1, 1, 1, 0.25 + colorlayerone / 1.33)
                    RoundedRectangle(size=(35, 35), pos=(middlex + 80, d * 50 + (middley - spacetwo / 2)))
                    Color(0,0,0,1)
                    RoundedRectangle(size=(30, 30), pos=(middlex + 82.5, d * 50 + (middley + 2.5 - spacetwo / 2)),
                                     radius=[(6, 6), (6, 6), (6, 6), (6, 6)])
                    biases = "{:.2f}".format(value)
                    draw.nodeslabel = Label(text=biases, font_size=10,size_hint=(None,None), pos=(middlex +49, d * 50 + (middley-31.5 - spacetwo / 2)),
                                            color=(1, 1, 1, .9))
                    draw.add_widget(draw.nodeslabel)
                j = 0
                e = 0
                for y in range(list_hidden[0]):
                    color = abs(network[1][d]['weights'][j]) / linecolorinput
                    with draw.canvas:
                        Color(color, color, color)
                        Line(points=[middlex -45, j * 50 + (middley+20 - spaceone / 2), middlex+80, d * 50 + (middley+20 - spacetwo / 2)])
                        j = j + 1
                for z in range(numoutputs):
                    color = abs(network[2][e]['weights'][d]) / linecoloroutput
                    with draw.canvas:
                        Color(color, color, color)
                        Line(points=[middlex+115, d * 50 + (middley+20 - spacetwo / 2), middlex + 230, e * 50 + (middley +20 - spaceoutput / 2)])
                    e = e + 1
                d = d + 1

        if len(list_hidden) == 3:
            # hiddenlayerone
            spacethree = list_hidden[2] * 35 + (list_hidden[2] - 1) * 25
            spacetwo = list_hidden[1] * 35 + (list_hidden[1] - 1) * 25
            valuefloat = 0
            linecolorinput = 0
            hiddenlayeronelabelone.pos = (middletitlex - 158, middletitley - 50)
            titlelayout.add_widget(hiddenlayeronelabelone)
            for x in range(list_hidden[0]):
                if valuefloat < abs(network[0][a]['weights'][-1]):
                    valuefloat = abs(network[0][a]['weights'][-1])
                e = 0
                i = 0
                for y in range(numdata):
                    if linecolorinput < abs(network[0][a]['weights'][i]):
                        linecolorinput = abs(network[0][a]['weights'][i])
                    i = i + 1
                a = a + 1
            a = 0
            for x in range(list_hidden[0]):
                with draw.canvas:
                    value = network[0][a]['weights'][-1]
                    colorlayerone = abs(network[0][a]['weights'][-1]) / valuefloat
                    Color(1, 1, 1, 0.25 + colorlayerone / 1.33)
                    RoundedRectangle(size=(35, 35), pos=(middlex - 130, a * 50 + (middley - spaceone / 2)))
                    Color(0,0,0,1)
                    RoundedRectangle(size=(30, 30), pos=(middlex + 2.5 - 130, a * 50 + (middley + 2.5 - spaceone / 2)),
                                     radius=[(6, 6), (6, 6), (6, 6), (6, 6)])
                    biases = "{:.2f}".format(value)
                    draw.nodeslabel = Label(text=biases,size_hint=(None,None), font_size=10, pos=(middlex-161.5, a * 50 + (middley-31.5 - spaceone / 2)),
                                            color=(1, 1, 1, .9))
                    draw.add_widget(draw.nodeslabel)
                i = 0
                e = 0
                for y in range(numdata):
                    color = abs(network[0][a]['weights'][i]) / linecolorinput
                    with draw.canvas:
                        Color(color, color, color)
                        Line(points=[middlex - 205, i * 50 + (middley-10 - space / 2) + 30, middlex-130,
                                     a * 50 + (middley+ 20 - spaceone / 2) ])
                        i = i + 1
                a = a + 1

            # hiddenlayertwo
            valuefloat = 0
            linecolorinput = 0
            d = 0
            a = 0
            hiddenlayeronelabeltwo.pos = (middletitlex - 28, middletitley - 50)
            titlelayout.add_widget(hiddenlayeronelabeltwo)
            for x in range(list_hidden[1]):
                if valuefloat < abs(network[1][d]['weights'][-1]):
                    valuefloat = abs(network[1][d]['weights'][-1])
                e = 0
                j = 0
                for y in range(list_hidden[0]):
                    if linecolorinput < abs(network[1][d]['weights'][j]):
                        linecolorinput = abs(network[1][d]['weights'][j])
                    j = j + 1
                d = d + 1
            d = 0
            for x in range(list_hidden[1]):
                with draw.canvas:
                    value = network[1][d]['weights'][-1]
                    colorlayerone = abs(network[1][d]['weights'][-1]) / valuefloat
                    Color(1, 1, 1, 0.25 + colorlayerone / 1.33)
                    RoundedRectangle(size=(35, 35), pos=(middlex, d * 50 + (middley - spacetwo / 2)))
                    Color(0,0,0,1)
                    RoundedRectangle(size=(30, 30), pos=(middlex + 2.5, d * 50 + (middley + 2.5 - spacetwo / 2)),
                                     radius=[(6, 6), (6, 6), (6, 6), (6, 6)])
                    biases = "{:.2f}".format(value)
                    draw.nodeslabel = Label(text=biases, font_size=10,size_hint=(None,None), pos=(middlex-31.5, d * 50 + (middley-31.5 - spacetwo / 2)),
                                            color=(1, 1, 1, .9))
                    draw.add_widget(draw.nodeslabel)
                j = 0
                e = 0
                for y in range(list_hidden[0]):
                    color = abs(network[1][d]['weights'][j]) / linecolorinput
                    with draw.canvas:
                        Color(color, color, color)
                        Line(points=[middlex-95, j * 50 + (middley+20 - spaceone / 2), middlex, d * 50 + (middley+20 - spacetwo / 2)])
                        j = j + 1
                d = d + 1

            #hiddenlayerthree
            valuefloat = 0
            linecoloroutput = 0
            linecolorinput = 0
            d = 0
            a = 0
            hiddenlayeronelabelthree.pos = (middletitlex + 101, middletitley - 50)
            titlelayout.add_widget(hiddenlayeronelabelthree)
            for x in range(list_hidden[2]):
                if valuefloat < abs(network[2][d]['weights'][-1]):
                    valuefloat = abs(network[2][d]['weights'][-1])

                e = 0
                j = 0
                for y in range(list_hidden[1]):
                    if linecolorinput < abs(network[2][d]['weights'][j]):
                        linecolorinput = abs(network[2][d]['weights'][j])
                    j = j + 1
                for z in range(numoutputs):
                    if linecoloroutput < abs(network[3][e]['weights'][d]):
                        linecoloroutput = abs(network[3][e]['weights'][d])
                    e = e + 1
                d = d + 1
            d = 0
            for x in range(list_hidden[2]):
                with draw.canvas:
                    value = network[2][d]['weights'][-1]
                    colorlayerone = abs(network[2][d]['weights'][-1]) / valuefloat
                    Color(1, 1, 1, 0.25 + colorlayerone / 1.33)
                    RoundedRectangle(size=(35, 35), pos=(middlex+130, d * 50 + (middley - spacethree / 2)))
                    Color(0,0,0,1)
                    RoundedRectangle(size=(30, 30), pos=(middlex + 132.5, d * 50 + (middley + 2.5 - spacethree / 2)),
                                     radius=[(6, 6), (6, 6), (6, 6), (6, 6)])
                    biases = "{:.2f}".format(value)
                    draw.nodeslabel = Label(text=biases, font_size=10,size_hint=(None,None), pos=(middlex+99, d * 50 + (middley-31.5 - spacethree / 2)),
                                            color=(1, 1, 1, .9))
                    draw.add_widget(draw.nodeslabel)
                l = 0
                e = 0
                for y in range(list_hidden[1]):
                    color = abs(network[2][d]['weights'][l]) / linecolorinput
                    with draw.canvas:
                        Color(color, color, color)
                        Line(points=[middlex+35, l * 50 + (middley+20 - spacetwo / 2), middlex+130, d * 50 + (middley+20 - spacethree / 2)])
                        l = l + 1
                for z in range(numoutputs):
                    color = abs(network[3][e]['weights'][d]) / linecoloroutput
                    with draw.canvas:
                        Color(color, color, color)
                        Line(points=[middlex+165, d * 50 + (middley+20 - spacethree / 2), middlex+230, e * 50 + (middley+20 - spaceoutput / 2)])
                    e = e + 1
                d = d + 1

        # output layer
        for x in range(numoutputs):
            outputoptions = str(c)
            lbloutputoptions = Label(text='[b]' + outputoptions, markup=True, size_hint=(None,None),
                                     pos=(middlex+230, c * 50 + (middley -30 - spaceoutput / 2)), color=(1, 1, 1, 1))
            draw.add_widget(lbloutputoptions)
            value = network[-1][c]['weights'][-1]
            print(value)
            with draw.canvas:
                Color(.85, .63, .27)
                RoundedRectangle(size=(35, 35), pos=(middlex + 230, c * 50 + (middley - spaceoutput / 2)))
                Color(0, 0, 0)
                RoundedRectangle(size=(30, 30), pos=(middlex + 232.5, c * 50 + (middley+2.5 - spaceoutput / 2)), radius=[(6,6),(6,6),(6,6),(6,6)])
                biases = "{:.2f}".format(value)
                #draw.nodeslabel = Label(text=biases, font_size=10, pos=(middlex + 196, c * 50 + (middley-34 - spaceoutput / 2)),color=(1, 1, 1, .9))
            c = c + 1
        # input layer
        for x in range(numdata):
            with draw.canvas:
                Color(.44, .27, .76)
                RoundedRectangle(size=(35, 35), pos=(middlex - 240, b * 50 + (middley - space / 2)))
                Color(0, 0, 0)
                RoundedRectangle(size=(30, 30), pos=(middlex - 237.5, b * 50 + (middley + 2.5 - space / 2)),
                                 radius=[(6, 6), (6, 6), (6, 6), (6, 6)])
                b = b + 1

        draw.add_widget(ti)
        draw.add_widget(ti2)
        #draw.add_widget(ti3)

        #data
        #draw.add_widget(Label(text='[b] Accuracy:' +' [/b]%.3f%%' % accuracylabel,size_hint=(None,None), markup = True, pos=(middlex-30, 0)))

    #When OK is pressed. Plain normal size
    def drawingblank(self, draw, titlelayout, *largs):
        global a, b, list_hidden, network, n_outputs, n_inputs, accuracylabel, progressbar
        global hiddenlayeronelabelthree, hiddenlayeronelabelone, hiddenlayeronelabeltwo, outputlabel, inputlabel
        a = 0
        b = 0
        c = 0
        d = 0
        e = 0
        g = 0
        i = 0
        j = 0
        l = 0

        draw.canvas.clear()

        numoutputs = n_outputs
        numdata = n_inputs
        space = numdata * 35 + (numdata - 1) * 25
        spaceone = list_hidden[0] * 35 + (list_hidden[0] - 1) * 25
        spaceoutput = numoutputs * 35 + (numoutputs - 1) * 25
        middlex = draw.width/2
        middley = draw.height/2 + 140
        middletitlex = titlelayout.width / 2
        middletitley = titlelayout.height / 2

        outputlabel.pos = (middletitlex + 202, middletitley - 60)
        inputlabel.pos = (middletitlex - 268, middletitley - 60)

        titlelayout.remove_widget(hiddenlayeronelabelone)
        titlelayout.remove_widget(hiddenlayeronelabeltwo)
        titlelayout.remove_widget(hiddenlayeronelabelthree)
        titlelayout.remove_widget(outputlabel)
        titlelayout.remove_widget(inputlabel)
        titlelayout.add_widget(outputlabel)
        titlelayout.add_widget(inputlabel)

        ti.pos = (90,(middley+5 - space / 2))
        draw.remove_widget(ti)
        ti2.pos = (90, 50 + (middley+5 - space / 2))
        draw.remove_widget(ti2)
        ti3.pos = (90, 2 * 50 + (middley+5 - space / 2))
        draw.remove_widget(ti3)

        if len(list_hidden) == 1:
            # hiddenlayerone
            valuefloat = 0
            linecoloroutput = 0
            linecolorinput = 0
            hiddenlayeronelabelone.pos = (middletitlex - 28, middletitley - 50)
            titlelayout.add_widget(hiddenlayeronelabelone)
            a = 0
            for x in range(list_hidden[0]):
                with draw.canvas:
                    Color(1,1,1)
                    RoundedRectangle(size=(35, 35), pos=(middlex, a * 50 + (middley - spaceone / 2)))
                    Color(0,0,0)
                    RoundedRectangle(size=(30, 30), pos=(middlex+2.5, a * 50 + (middley + 2.5 - spaceone / 2)), radius=[(6,6),(6,6),(6,6),(6,6)])
                i = 0
                e = 0
                for y in range(numdata):
                    with draw.canvas:
                        Color(.5,.5,.5)
                        Line(points=[middlex - 205, i * 50 + (middley-10 - space / 2) + 30, middlex, a * 50 + (middley+20 - spaceone / 2) ])
                        i = i + 1
                for z in range(numoutputs):
                    with draw.canvas:
                        Color(.5,.5,.5)
                        Line(points=[middlex+35, a * 50 + (middley  - spaceone / 2)+ 20, middlex+230,
                                     e * 50 + (middley+ 20 - spaceoutput / 2) ])
                    e = e + 1
                a = a + 1

        if len(list_hidden) == 2:
            # hiddenlayerone
            spacetwo = list_hidden[1] * 35 + (list_hidden[1] - 1) * 25
            hiddenlayeronelabelone.pos = (middletitlex - 108, middletitley - 50)
            titlelayout.add_widget(hiddenlayeronelabelone)
            a = 0
            for x in range(list_hidden[0]):
                with draw.canvas:
                    Color(1,1,1)
                    RoundedRectangle(size=(35, 35), pos=(middlex-80, a * 50 + (middley - spaceone / 2)))
                    Color(0,0,0,1)
                    RoundedRectangle(size=(30, 30), pos=(middlex + 2.5 - 80, a * 50 + (middley + 2.5 - spaceone / 2)),
                                     radius=[(6, 6), (6, 6), (6, 6), (6, 6)])
                i = 0
                e = 0
                for y in range(numdata):
                    with draw.canvas:
                        Color(.5,.5,.5)
                        Line(points=[middlex - 205, i * 50 + (middley-10 - space / 2) + 30, middlex-80, a * 50 + (middley + 20 - spaceone / 2) ])
                        i = i + 1
                a = a + 1

            # hiddenlayertwo
            d = 0
            a = 0
            hiddenlayeronelabeltwo.pos = (middletitlex + 52, middletitley - 50)
            titlelayout.add_widget(hiddenlayeronelabeltwo)
            d = 0
            for x in range(list_hidden[1]):
                with draw.canvas:
                    Color(1,1,1)
                    RoundedRectangle(size=(35, 35), pos=(middlex + 80, d * 50 + (middley - spacetwo / 2)))
                    Color(0,0,0,1)
                    RoundedRectangle(size=(30, 30), pos=(middlex + 82.5, d * 50 + (middley + 2.5 - spacetwo / 2)),
                                     radius=[(6, 6), (6, 6), (6, 6), (6, 6)])
                j = 0
                e = 0
                for y in range(list_hidden[0]):
                    with draw.canvas:
                        Color(.5,.5,.5)
                        Line(points=[middlex -45, j * 50 + (middley+20 - spaceone / 2), middlex+80, d * 50 + (middley+20 - spacetwo / 2)])
                        j = j + 1
                for z in range(numoutputs):
                    with draw.canvas:
                        Color(.5,.5,.5)
                        Line(points=[middlex+115, d * 50 + (middley+20 - spacetwo / 2), middlex + 230, e * 50 + (middley +20 - spaceoutput / 2)])
                    e = e + 1
                d = d + 1

        if len(list_hidden) == 3:
            # hiddenlayerone
            spacethree = list_hidden[2] * 35 + (list_hidden[2] - 1) * 25
            spacetwo = list_hidden[1] * 35 + (list_hidden[1] - 1) * 25
            hiddenlayeronelabelone.pos = (middletitlex - 158, middletitley - 50)
            titlelayout.add_widget(hiddenlayeronelabelone)
            a = 0
            for x in range(list_hidden[0]):
                with draw.canvas:
                    Color(1,1,1)
                    RoundedRectangle(size=(35, 35), pos=(middlex - 130, a * 50 + (middley - spaceone / 2)))
                    Color(0,0,0,1)
                    RoundedRectangle(size=(30, 30), pos=(middlex + 2.5 - 130, a * 50 + (middley + 2.5 - spaceone / 2)),
                                     radius=[(6, 6), (6, 6), (6, 6), (6, 6)])
                i = 0
                e = 0
                for y in range(numdata):
                    with draw.canvas:
                        Color(.5,.5,.5)
                        Line(points=[middlex - 205, i * 50 + (middley-10 - space / 2) + 30, middlex-130,
                                     a * 50 + (middley+ 20 - spaceone / 2) ])
                        i = i + 1
                a = a + 1

            # hiddenlayertwo
            valuefloat = 0
            linecolorinput = 0
            d = 0
            a = 0
            hiddenlayeronelabeltwo.pos = (middletitlex - 28, middletitley - 50)
            titlelayout.add_widget(hiddenlayeronelabeltwo)
            d = 0
            for x in range(list_hidden[1]):
                with draw.canvas:
                    Color(1,1,1)
                    RoundedRectangle(size=(35, 35), pos=(middlex, d * 50 + (middley - spacetwo / 2)))
                    Color(0,0,0,1)
                    RoundedRectangle(size=(30, 30), pos=(middlex + 2.5, d * 50 + (middley + 2.5 - spacetwo / 2)),
                                     radius=[(6, 6), (6, 6), (6, 6), (6, 6)])
                j = 0
                e = 0
                for y in range(list_hidden[0]):
                    with draw.canvas:
                        Color(.5,.5,.5)
                        Line(points=[middlex-95, j * 50 + (middley+20 - spaceone / 2), middlex, d * 50 + (middley+20 - spacetwo / 2)])
                        j = j + 1
                d = d + 1

            #hiddenlayerthree
            valuefloat = 0
            linecoloroutput = 0
            linecolorinput = 0
            d = 0
            a = 0
            hiddenlayeronelabelthree.pos = (middletitlex + 101, middletitley - 50)
            titlelayout.add_widget(hiddenlayeronelabelthree)
            d = 0
            for x in range(list_hidden[2]):
                with draw.canvas:
                    Color(1, 1, 1)
                    RoundedRectangle(size=(35, 35), pos=(middlex+130, d * 50 + (middley - spacethree / 2)))
                    Color(0,0,0,1)
                    RoundedRectangle(size=(30, 30), pos=(middlex + 132.5, d * 50 + (middley + 2.5 - spacethree / 2)),
                                     radius=[(6, 6), (6, 6), (6, 6), (6, 6)])
                l = 0
                e = 0
                for y in range(list_hidden[1]):
                    with draw.canvas:
                        Color(.5,.5,.5)
                        Line(points=[middlex+35, l * 50 + (middley+20 - spacetwo / 2), middlex+130, d * 50 + (middley+20 - spacethree / 2)])
                        l = l + 1
                for z in range(numoutputs):
                    with draw.canvas:
                        Color(.5,.5,.5)
                        Line(points=[middlex+165, d * 50 + (middley+20 - spacethree / 2), middlex+230, e * 50 + (middley+20 - spaceoutput / 2)])
                    e = e + 1
                d = d + 1

        # output layer
        for x in range(numoutputs):
            outputoptions = str(c)
            lbloutputoptions = Label(text='[b]' + outputoptions, markup=True, size_hint=(None,None),
                                     pos=(middlex+230, c * 50 + (middley -30 - spaceoutput / 2)), color=(1, 1, 1, 1))
            draw.add_widget(lbloutputoptions)
            with draw.canvas:
                Color(.85, .63, .27)
                RoundedRectangle(size=(35, 35), pos=(middlex + 230, c * 50 + (middley - spaceoutput / 2)))
                Color(0, 0, 0)
                RoundedRectangle(size=(30, 30), pos=(middlex + 232.5, c * 50 + (middley+2.5 - spaceoutput / 2)), radius=[(6,6),(6,6),(6,6),(6,6)])
            c = c + 1

        # input layer
        for x in range(numdata):
            with draw.canvas:
                Color(.69,0,.83)
                RoundedRectangle(size=(35, 35), pos=(middlex - 240, b * 50 + (middley - space / 2)))
                Color(0, 0, 0)
                RoundedRectangle(size=(30, 30), pos=(middlex - 237.5, b * 50 + (middley + 2.5 - space / 2)),
                                 radius=[(6, 6), (6, 6), (6, 6), (6, 6)])
                b = b + 1

        draw.add_widget(ti)
        draw.add_widget(ti2)
        #draw.add_widget(ti3)

    #Plain drawing small size
    def drawingtryout(self, draw, titlelayout, *largs):
        global a, b, list_hidden, network, n_outputs, n_inputs, accuracylabel, progressbar
        global hiddenlayeronelabelthree, hiddenlayeronelabelone, hiddenlayeronelabeltwo, outputlabel, inputlabel
        a = 0
        b = 0
        c = 0
        d = 0
        e = 0
        g = 0
        i = 0
        l = 0


        draw.canvas.clear()

        numoutputs = n_outputs
        numdata = n_inputs
        space = numdata * 20 + (numdata - 1) * 5
        spaceone = list_hidden[0] * 20 + (list_hidden[0] - 1) * 5
        spaceoutput = numoutputs * 20 + (numoutputs - 1) * 5
        middlex = draw.width/2
        middley = draw.height/2 -20
        middletitlex = titlelayout.width / 2
        middletitley = titlelayout.height / 2


        titlelayout.remove_widget(hiddenlayeronelabelone)
        titlelayout.remove_widget(hiddenlayeronelabeltwo)
        titlelayout.remove_widget(hiddenlayeronelabelthree)



        if len(list_hidden) == 1:
            # hiddenlayerone
            #hiddenlayeronelabelone.pos =(middletitlex - 28, middletitley - 50)
            #titlelayout.add_widget(hiddenlayeronelabelone)
            a = 0
            for x in range(list_hidden[0]):
                with draw.canvas:
                    Color(1,1,1)
                    RoundedRectangle(size=(20, 20), pos=(middlex, a * 30 + (middley - spaceone / 2)))
                    Color(0,0,0,1)
                    RoundedRectangle(size=(15, 15), pos=(middlex+2.5, a * 30 + (middley + 2.5 - spaceone / 2)), radius=[(6,6),(6,6),(6,6),(6,6)])
                i = 0
                e = 0
                for y in range(numdata):
                    with draw.canvas:
                        Color(.5,.5,.5)
                        Line(points=[middlex - 220, i * 30 + (middley+10 - space / 2), middlex, a * 30 + (middley+10 - spaceone / 2) ])
                        i = i + 1
                for z in range(numoutputs):
                    with draw.canvas:
                        Color(.5,.5,.5)
                        Line(points=[middlex+20, a * 30 + (middley+10  - spaceone / 2), middlex+230,
                                     e * 30 + (middley+ 10 - spaceoutput / 2) ])
                    e = e + 1
                a = a + 1

        if len(list_hidden) == 2:
            # hiddenlayerone
            spacetwo = list_hidden[1] * 20 + (list_hidden[1] - 1) * 5
            #hiddenlayeronelabelone.pos=(middletitlex-108, middletitley - 50)
            #titlelayout.add_widget(hiddenlayeronelabelone)
            a = 0
            for x in range(list_hidden[0]):
                with draw.canvas:
                    Color(1,1,1)
                    RoundedRectangle(size=(20, 20), pos=(middlex-80, a * 30 + (middley - spaceone / 2)))
                    Color(0,0,0,1)
                    RoundedRectangle(size=(15, 15), pos=(middlex + 2.5 - 80, a * 30 + (middley + 2.5 - spaceone / 2)),
                                     radius=[(6, 6), (6, 6), (6, 6), (6, 6)])
                i = 0
                e = 0
                for y in range(numdata):
                    with draw.canvas:
                        Color(.5,.5,.5)
                        Line(points=[middlex - 220, i * 30 + (middley-20 - space / 2) + 30, middlex-80, a * 30 + (middley + 10 - spaceone / 2) ])
                        i = i + 1
                a = a + 1

            # hiddenlayertwo
            d = 0
            a = 0
            #hiddenlayeronelabeltwo.pos=(middletitlex+52, middletitley-50)
            #titlelayout.add_widget(hiddenlayeronelabeltwo)
            d = 0
            for x in range(list_hidden[1]):
                with draw.canvas:
                    Color(1,1,1)
                    RoundedRectangle(size=(20, 20), pos=(middlex + 80, d * 30 + (middley - spacetwo / 2)))
                    Color(0,0,0,1)
                    RoundedRectangle(size=(15, 15), pos=(middlex + 82.5, d * 30 + (middley + 2.5 - spacetwo / 2)),
                                     radius=[(6, 6), (6, 6), (6, 6), (6, 6)])
                j = 0
                e = 0
                for y in range(list_hidden[0]):
                    with draw.canvas:
                        Color(.5,.5,.5)
                        Line(points=[middlex-60, j * 30 + (middley+10 - spaceone / 2), middlex+80, d * 30 + (middley+10 - spacetwo / 2)])
                        j = j + 1
                for z in range(numoutputs):
                    with draw.canvas:
                        Color(.5,.5,.5)
                        Line(points=[middlex+100, d * 30 + (middley+10 - spacetwo / 2), middlex + 230, e * 30 + (middley +10 - spaceoutput / 2)])
                    e = e + 1
                d = d + 1

        if len(list_hidden) == 3:
            # hiddenlayerone
            spacethree = list_hidden[2] * 20 + (list_hidden[2] - 1) * 5
            spacetwo = list_hidden[1] * 20 + (list_hidden[1] - 1) * 5
            #hiddenlayeronelabelone.pos=(middletitlex-158, middletitley-50)
            #titlelayout.add_widget(hiddenlayeronelabelone)
            a = 0
            for x in range(list_hidden[0]):
                with draw.canvas:
                    Color(1,1,1)
                    RoundedRectangle(size=(20, 20), pos=(middlex - 130, a * 30 + (middley - spaceone / 2)))
                    Color(0,0,0,1)
                    RoundedRectangle(size=(15, 15), pos=(middlex + 2.5 - 130, a * 30 + (middley + 2.5 - spaceone / 2)),
                                     radius=[(6, 6), (6, 6), (6, 6), (6, 6)])
                i = 0
                e = 0
                for y in range(numdata):
                    with draw.canvas:
                        Color(.5,.5,.5)
                        Line(points=[middlex - 220, i * 30 + (middley-20 - space / 2) + 30, middlex-130,
                                     a * 30 + (middley+ 10 - spaceone / 2) ])
                        i = i + 1
                a = a + 1

            # hiddenlayertwo
            d = 0
            a = 0
            #hiddenlayeronelabeltwo.pos=(middletitlex-28, middletitley-50)
            #titlelayout.add_widget(hiddenlayeronelabeltwo)
            d = 0
            for x in range(list_hidden[1]):
                with draw.canvas:
                    Color(1,1,1)
                    RoundedRectangle(size=(20, 20), pos=(middlex, d * 30 + (middley - spacetwo / 2)))
                    Color(0,0,0,1)
                    RoundedRectangle(size=(15, 15), pos=(middlex + 2.5, d * 30 + (middley + 2.5 - spacetwo / 2)),
                                     radius=[(6, 6), (6, 6), (6, 6), (6, 6)])
                j = 0
                e = 0
                for y in range(list_hidden[0]):
                    with draw.canvas:
                        Color(.5,.5,.5)
                        Line(points=[middlex-110, j * 30 + (middley+10 - spaceone / 2), middlex, d * 30 + (middley+10 - spacetwo / 2)])
                        j = j + 1
                d = d + 1

            #hiddenlayerthree
            d = 0
            a = 0
            #hiddenlayeronelabelthree.pos=(middletitlex+101, middletitley -50)
            #titlelayout.add_widget(hiddenlayeronelabelthree)
            d = 0
            for x in range(list_hidden[2]):
                with draw.canvas:
                    Color(1,1,1)
                    RoundedRectangle(size=(20, 20), pos=(middlex+130, d * 30 + (middley - spacethree / 2)))
                    Color(0,0,0,1)
                    RoundedRectangle(size=(15, 15), pos=(middlex + 132.5, d * 30 + (middley + 2.5 - spacethree / 2)),
                                     radius=[(6, 6), (6, 6), (6, 6), (6, 6)])
                l = 0
                e = 0
                for y in range(list_hidden[1]):
                    with draw.canvas:
                        Color(.5,.5,.5)
                        Line(points=[middlex+20, l * 30 + (middley+10 - spacetwo / 2), middlex+130, d * 30 + (middley+10 - spacethree / 2)])
                        l = l + 1
                for z in range(numoutputs):
                    with draw.canvas:
                        Color(.5,.5,.5)
                        Line(points=[middlex+150, d * 30 + (middley+10 - spacethree / 2), middlex+230, e * 30 + (middley+10 - spaceoutput / 2)])
                    e = e + 1
                d = d + 1

        # output layer
        for x in range(numoutputs):
            outputoptions = str(c)
            lbloutputoptions = Label(text='[b]' + outputoptions, markup=True, size_hint=(None,None),
                                     pos=(middlex+215, c * 30 + (middley -40 - spaceoutput / 2)), color=(1, 1, 1, 1))
            draw.add_widget(lbloutputoptions)
            with draw.canvas:
                Color(.85+c/9,0.63+c/9,.27+c/9)
                RoundedRectangle(size=(20, 20), pos=(middlex + 230, c * 30 + (middley - spaceoutput / 2)))
                Color(0, 0, 0)
                RoundedRectangle(size=(15, 15), pos=(middlex + 232.5, c * 30 + (middley+2.5 - spaceoutput / 2)), radius=[(6,6),(6,6),(6,6),(6,6)])
            c = c + 1
        # input layer
        for x in range(numdata):
            with draw.canvas:
                Color(.69,0,.83)
                RoundedRectangle(size=(20, 20), pos=(middlex - 240, b * 30 + (middley - space / 2)))
                Color(0, 0, 0)
                RoundedRectangle(size=(15, 15), pos=(middlex - 237.5, b * 30 + (middley + 2.5 - space / 2)),
                                 radius=[(6, 6), (6, 6), (6, 6), (6, 6)])
                b = b + 1

    #Clear canvas
    def clear(self,draw,titlelayout, *largs):
        draw.canvas.clear()
        titlelayout.remove_widget(hiddenlayeronelabelone)
        titlelayout.remove_widget(hiddenlayeronelabeltwo)
        titlelayout.remove_widget(hiddenlayeronelabelthree)
        titlelayout.remove_widget(outputlabel)
        titlelayout.remove_widget(inputlabel)

    def build(self):
        global layoutneuronmainscreen, layoutlabels, parentlayout
        list_hidden.append(6)
        numlayers = str(len(list_hidden))

        parentlayout = LayoutScreen()
        layoutneuronmainscreen = parentlayout.ids.idLayoutNeuron.ids.idLayoutNeuronMainScreen.ids.idrelativelayoutneuronmainscreen
        layoutlabels = parentlayout.ids.idLayoutNeuron.ids.idLayoutNeuronMainScreen.ids.idLabelsLayoutNeuronMainScreen
        layoutmainscreen = parentlayout.ids.idLayoutNeuron.ids.idLayoutNeuronMainScreen
        layout = parentlayout.ids.idLayoutMenu.ids.idBoxLayoutMenu
        btnlayout = parentlayout.ids.idLayoutMenu

        neuronlayout = BoxLayout(orientation='vertical', size_hint=(.3,1))
        neurontwolayout = BoxLayout(orientation='vertical', size_hint=(.3,1))
        neuronthreelayout = BoxLayout(orientation='vertical', size_hint=(.3,1))
        numlayerslayout = BoxLayout(orientation='vertical', size_hint=(.3,1))
        epochlayout = BoxLayout(orientation ='vertical', pos_hint={'top':1}, size_hint=(.3, .8))
        ratelayout = BoxLayout(orientation ='vertical', pos_hint={'top':1}, size_hint=(.3, .8))
        foldslayout = BoxLayout(orientation ='vertical', pos_hint={'top':1}, size_hint=(.3, .8))
        layoutsettingsvertical = BoxLayout(orientation ='vertical')
        layoutsettingsup = BoxLayout(orientation='horizontal', spacing = 10)
        layoutsettingsdown = BoxLayout(orientation='horizontal',spacing = 10)
        layoutsettingsupleft = BoxLayout(orientation='horizontal',spacing = 10, size_hint=(.7,1))
        layoutsettingsupright = BoxLayout(orientation='horizontal',spacing = 10, size_hint=(.3,1))



        btnPredict = Button(color=(1, 1, 1, 1), text='[b]PREDICT', markup = True, font_size=18, size_hint=(.95, .9), pos_hint={'center_x':.5, 'center_y':.5}, background_normal='ButtonPlusP.png', background_down='ButtonPlusPDown.png')
        btnTrain = Button(text='[b]TRAIN',markup = True, background_normal='ButtonPlusP.png', background_down='ButtonPlusADown.png', size_hint=(.95,.9),
                       pos_hint={'center_x':.5, 'center_y':.5}, color = (1,1,1,1), font_size=18)
        btnError = Button(text='[b]ERROR', markup=True, background_normal='ButtonPlusP.png',
                          background_down='ButtonPlusADown.png', size_hint=(.95, .9),
                          pos_hint={'center_x': .5, 'center_y': .5}, color=(1, 1, 1, 1), font_size=18)
        btnOk = Button(text='[b]OK', markup=True, background_normal='ButtonPlusB.png',
                          background_down='ButtonPlusA.png', size_hint=(.8, .8),
                          pos_hint={'center_x': .5, 'center_y': .5}, color=(0, 0, 0, 1), font_size=18)
        btnOpen = Button(text='[b]OPEN', markup=True, background_normal='ButtonPlusP.png',
                       background_down='ButtonPlusGDown.png', size_hint=(.8, .8),
                       pos_hint={'center_x': .5, 'center_y': .5}, color=(1, 1, 1, 1), font_size=18)
        btnBack = Button(text='[b]BACK', markup=True, background_normal='ButtonPlusP.png',
                       background_down='ButtonPlusADown.png', size_hint=(.95, .9),
                       pos_hint={'center_x': .5, 'center_y': .5}, color=(1, 1, 1, 1), font_size=18)
        btnSettings = Button(text='[b]SETTINGS', markup = True, background_normal='ButtonPlusP.png', background_down='ButtonPlusADown.png')
        btnnewinput=Button(color=(1,1,1,1), text='[b]New Input',markup = True, font_size=18, size_hint=(.95,.9),
                       pos_hint={'center_x':.5, 'center_y':.5},background_normal='ButtonPlusP.png', background_down='ButtonPlusPDown.png')
        btnCancel = Button(color=(1,1,1,1), text='[b]CANCEL',markup = True, font_size=18, size_hint=(.95,.9), pos_hint={'center_x':.5, 'center_y':.5},background_normal='ButtonPlusP.png', background_down='ButtonPlusPDown.png')



        box = CheckBox()
        lblbox = Label(text='Sigmoid:', color=(1, 1, 1, 1), font_size=(14))
        boxshowbiaslabel = CheckBox()

        # Left Neuron Switcher
        btnplusneuron = Button(text = '[b]+',markup = True, size_hint=(.2,.8),pos_hint={'center_x':.5, 'center_y':.5},
                               background_normal='ButtonPlusP.png',
                               background_down='ButtonPlusA.png',
                               background_disabled_normal='ButtonDeactivated.png')
        btnlessneuron = Button(text = '[b]-',markup = True,size_hint=(.2, .8),pos_hint={'center_x':.5, 'center_y':.5},
                               background_normal='ButtonPlusP.png',
                               background_down='ButtonPlusA.png',
                               background_disabled_normal='ButtonDeactivated.png')
        lblhiddenlayer = Label(text=str(list_hidden[0]))
        lblneuron = Label(text='[b]Neurons', markup=True)

        # Second neuron switcher
        btnplusneurontwo = Button(text = '[b]+',markup = True,size_hint=(.2, .8),pos_hint={'center_x':.5, 'center_y':.5},
                                  background_normal='ButtonPlusP.png',
                                  background_down='ButtonPlusR.png',
                                  background_disabled_normal='ButtonDeactivated.png')
        btnlessneurontwo = Button(text = '[b]-',markup = True,size_hint=(.2, .8), pos_hint={'center_x':.5, 'center_y':.5},
                                  background_normal='ButtonPlusP.png',
                                  background_down='ButtonPlusR.png',
                                  background_disabled_normal='ButtonDeactivated.png')
        lblhiddenlayertwo = Label(text='4')
        lblneurontwo = Label(text='[b]Neurons', markup=True)

        # Right/Third neuron switcher
        btnplusneuronthree = Button(text = '[b]+',markup = True,size_hint=(.2, .8),pos_hint= {'center_x':1, 'center_y':.5},
                                    background_normal='ButtonPlusP.png',
                                    background_down='ButtonPlusPDown.png',
                                    background_disabled_normal='ButtonDeactivated.png')
        btnlessneuronthree = Button(text = '[b]-',markup = True,size_hint=(.2, .8),pos_hint={'center_x':1, 'center_y':.5},
                                    background_normal='ButtonPlusP.png',
                                    background_down='ButtonPlusPDown.png',
                                    background_disabled_normal='ButtonDeactivated.png')
        lblhiddenlayerthree = Label(text='4')
        lblneuronthree = Label(text='[b]Neurons', markup=True)

        # Number of layers switcher
        btnlayerplusneuron = Button(text = '[b]+',markup = True,size_hint=(.2, .8),pos_hint={'center_x':1, 'center_y':.5},
                                    background_normal='ButtonPlusP.png',
                                    background_down='ButtonPlusGDown.png',
                                    background_disabled_normal='ButtonDeactivated.png')
        btnlayerlessneuron = Button(text = '[b]-',markup = True,size_hint=(.2, .8),pos_hint={'center_x':1, 'center_y':.5},
                                    background_normal='ButtonPlusP.png',
                                    background_down='ButtonPlusGDown.png',
                                    background_disabled_normal='ButtonDeactivated.png')
        lbllayer = Label(text=numlayers)
        lblnumlayers = Label(text='[b] Hiddenlayers',markup = True, font_size=(11))
        lblblack = Widget(size_hint=(.1,.1))
        lblblacktwo = Widget(size_hint=(.1,.1))
        lblblackthree = Widget(size_hint=(.1,.1))
        canvas= Widget(size_hint=(.1,.1))
        visualize = Label(size_hint=(.1,.1))

        btnplusepoch = Button(text='[b]+', markup=True, size_hint=(.2, .8),
                              pos_hint={'center_x': 1, 'center_y': .5},
                              background_normal='ButtonPlusP.png',
                              background_down='ButtonPlusA.png',
                              background_disabled_normal='ButtonDeactivated.png')
        btnlessepoch = Button(text='[b]-', markup=True, size_hint=(.2, .8),
                                    pos_hint={'center_x': 1, 'center_y': .5},
                                    background_normal='ButtonPlusP.png',
                                    background_down='ButtonPlusA.png',
                                    background_disabled_normal='ButtonDeactivated.png')
        lblepoch = Label(text='[b] RANGE:',markup=True, font_size=(11), pos_hint={'center_x':.5, 'center_y':.5})
        numepoch = Label(text=str(n_epoch))
        btnplusrate = Button(text='[b]+', markup=True, size_hint=(.2, .8),
                              pos_hint={'center_x': 1, 'center_y': .5},
                              background_normal='ButtonPlusP.png',
                              background_down='ButtonPlusR.png',
                              background_disabled_normal='ButtonDeactivated.png')
        btnlessrate = Button(text='[b]-', markup=True, size_hint=(.2, .8),
                              pos_hint={'center_x': 1, 'center_y': .5},
                              background_normal='ButtonPlusP.png',
                              background_down='ButtonPlusR.png',
                              background_disabled_normal='ButtonDeactivated.png')
        lblrate = Label(text='[b]RATE:',markup=True, font_size=(11), pos_hint={'center_x': .5, 'center_y': .5})
        numrate = Label(text=str(l_rate))
        btnplusfolds = Button(text='[b]+', markup=True, size_hint=(.2, .8),
                             pos_hint={'center_x': 1, 'center_y': .5},
                             background_normal='ButtonPlusP.png',
                             background_down='ButtonPlusGDown.png',
                             background_disabled_normal='ButtonDeactivated.png')
        btnlessfolds = Button(text='[b]-', markup=True, size_hint=(.2, .8),
                             pos_hint={'center_x': 1, 'center_y': .5},
                             background_normal='ButtonPlusP.png',
                             background_down='ButtonPlusGDown.png',
                             background_disabled_normal='ButtonDeactivated.png')
        lblfolds = Label(text='[b]FOLDS:',markup=True, font_size=(11), pos_hint={'center_x': .5, 'center_y': .5})
        numfolds = Label(text=str(n_folds))


        layoutsettingsup.add_widget(layoutsettingsupleft)
        layoutsettingsup.add_widget(lblblack)
        layoutsettingsup.add_widget(layoutsettingsupright)
        neuronlayout.add_widget(lblneuron)
        neuronlayout.add_widget(lblhiddenlayer)
        layoutsettingsupleft.add_widget(lblblacktwo)
        layoutsettingsupleft.add_widget(btnlessneuron)
        layoutsettingsupleft.add_widget(neuronlayout)
        layoutsettingsupleft.add_widget(btnplusneuron)
        layoutsettingsupright.add_widget(btnlayerlessneuron)
        layoutsettingsupright.add_widget(numlayerslayout)
        layoutsettingsupright.add_widget(btnlayerplusneuron)
        layoutsettingsupright.add_widget(visualize)
        layoutsettingsdown.add_widget(lblblackthree)
        layoutsettingsdown.add_widget(btnOk)
        layoutsettingsdown.add_widget(btnOpen)
        layoutsettingsdown.add_widget(btnlessepoch)
        layoutsettingsdown.add_widget(epochlayout)
        layoutsettingsdown.add_widget(btnplusepoch)
        layoutsettingsdown.add_widget(btnlessrate)
        layoutsettingsdown.add_widget(ratelayout)
        layoutsettingsdown.add_widget(btnplusrate)
        layoutsettingsdown.add_widget(btnlessfolds)
        layoutsettingsdown.add_widget(foldslayout)
        layoutsettingsdown.add_widget(btnplusfolds)
        numlayerslayout.add_widget(lblnumlayers)
        numlayerslayout.add_widget(lbllayer)
        btnlayerlessneuron.disabled = True
        epochlayout.add_widget(lblepoch)
        epochlayout.add_widget(numepoch)
        ratelayout.add_widget(lblrate)
        ratelayout.add_widget(numrate)
        foldslayout.add_widget(lblfolds)
        foldslayout.add_widget(numfolds)

        neurontwolayout.add_widget(lblneurontwo)
        neurontwolayout.add_widget(lblhiddenlayertwo)
        neuronthreelayout.add_widget(lblneuronthree)
        neuronthreelayout.add_widget(lblhiddenlayerthree)

        def settingslayout(settings):
            btnlayout.height = 300
            layoutmainscreen.height = 350
            layouttrainscreen = parentlayout.ids.idLayoutNeuron.ids.idAnchorLayoutNeuron
            layouttrainscreen.anchor_x = 'center'

            layout.remove_widget(btnTrain)
            layout.remove_widget(btnError)
            layout.remove_widget(btnPredict)
            layout.remove_widget(btnSettings)

            layout.add_widget(layoutsettingsvertical)
            layoutsettingsvertical.add_widget(layoutsettingsup)
            layoutsettingsvertical.add_widget(layoutsettingsdown)

            self.clear(layoutneuronmainscreen, layoutlabels)

        settingslayout(lblblackthree)

        def mainlayout(self):
            layoutmainscreen.height = 580
            btnlayout.height = 50
            layouttrainscreen = parentlayout.ids.idLayoutNeuron.ids.idAnchorLayoutNeuron
            layouttrainscreen.anchor_x = 'center'
            layout.remove_widget(layoutsettingsvertical)
            layoutsettingsvertical.remove_widget(layoutsettingsup)
            layoutsettingsvertical.remove_widget(layoutsettingsdown)
            layout.remove_widget(btnCancel)
            layout.remove_widget(btnnewinput)
            layout.remove_widget(btnBack)

            layout.add_widget(btnSettings)
            layout.add_widget(btnTrain)
            layout.add_widget(btnError)
            layout.add_widget(btnPredict)

            MyApp.drawingblank(btnOk, layoutneuronmainscreen, layoutlabels)

        def predictlayout(self):
            layouttrainscreen = parentlayout.ids.idLayoutNeuron.ids.idAnchorLayoutNeuron
            layouttrainscreen.anchor_x = 'right'
            layout.remove_widget(btnTrain)
            layout.remove_widget(btnError)
            layout.remove_widget(btnPredict)
            layout.remove_widget(btnCancel)

            layout.add_widget(btnBack)
            layout.add_widget(btnnewinput)

        def trainlayout(self):
            layouttrainscreen = parentlayout.ids.idLayoutNeuron.ids.idAnchorLayoutNeuron
            layouttrainscreen.anchor_x = 'right'
            layout.remove_widget(btnTrain)
            layout.remove_widget(btnError)
            layout.remove_widget(btnPredict)
            layout.remove_widget(btnCancel)

            layout.add_widget(btnBack)

        def cancellayout(self):
            layout.remove_widget(btnTrain)
            layout.remove_widget(btnError)
            layout.remove_widget(btnPredict)
            layout.remove_widget(btnBack)
            layout.remove_widget(btnnewinput)
            layout.remove_widget(btnSettings)
            layout.add_widget(btnCancel)

        def btnplusneuronpressed (btnplusneuron):
            btnlessneuron.disabled = False
            if list_hidden[0] >= 10:
                list_hidden[0] = 10
            else:
                list_hidden[0] = list_hidden[0] + 1
                self.drawingtryout(layoutneuronmainscreen, layoutlabels)
                lblhiddenlayer.text = str(list_hidden[0])
                if list_hidden[0] == 10:
                    btnplusneuron.disabled = True
        def btnlessneuronpressed(btnlessneuron):
            btnplusneuron.disabled = False
            if list_hidden[0] <= 1:
                list_hidden[0] = 1
            else:
                list_hidden[0] = list_hidden[0] - 1
                lblhiddenlayer.text = str(list_hidden[0])
                self.drawingtryout(layoutneuronmainscreen, layoutlabels)
                if list_hidden[0] == 1:
                    btnlessneuron.disabled = True

        def btnplusneurontwopressed(btnplusneurontwo):
            btnlessneurontwo.disabled = False

            if list_hidden[1] >= 10:
                list_hidden[1] = 10
            else:
                list_hidden[1] = list_hidden[1] + 1
                lblhiddenlayertwo.text = str(list_hidden[1])
                self.drawingtryout(layoutneuronmainscreen, layoutlabels)
                if list_hidden[1] == 10:
                    btnplusneurontwo.disabled = True
        def btnlessneurontwopressed(btnlessneurontwo):
            btnplusneurontwo.disabled = False
            if list_hidden[1] <= 1:
                list_hidden[1] = 1
            else:
                list_hidden[1] = list_hidden[1] - 1
                lblhiddenlayertwo.text = str(list_hidden[1])
                self.drawingtryout(layoutneuronmainscreen, layoutlabels)
                if list_hidden[1] == 1:
                    btnlessneurontwo.disabled = True

        def btnplusneuronthreepressed(btnplusneuronthree):
            btnlessneuronthree.disabled = False

            if list_hidden[2] >= 10:
                list_hidden[2] = 10
            else:
                list_hidden[2] = list_hidden[2] + 1
                lblhiddenlayerthree.text = str(list_hidden[2])
                self.drawingtryout(layoutneuronmainscreen, layoutlabels)
                if list_hidden[2] == 10:
                    btnplusneuronthree.disabled = True
        def btnlessneuronthreepressed(btnlessneuronthree):
            btnplusneuronthree.disabled = False
            if list_hidden[2] <= 1:
                list_hidden[2] = 1
            else:
                list_hidden[2] = list_hidden[2] - 1
                lblhiddenlayerthree.text = str(list_hidden[2])
                self.drawingtryout(layoutneuronmainscreen, layoutlabels)
                if list_hidden[2] == 1:
                    btnlessneuronthree.disabled = True

        def btnlayerplusneuronpressed(btnlayerplusneuron):
            btnlayerlessneuron.disabled = False
            if len(list_hidden) >= 3:
                pass
            else:
                numlayers = str(len(list_hidden))
                lbllayer.text = numlayers
                if len(list_hidden) == 1:
                    list_hidden.append(4)
                    self.drawingtryout(layoutneuronmainscreen, layoutlabels)

                    layoutsettingsupleft.add_widget(btnlessneurontwo)
                    layoutsettingsupleft.add_widget(neurontwolayout)
                    layoutsettingsupleft.add_widget(btnplusneurontwo)
                    lbllayer.text = str(len(list_hidden))
                    lblhiddenlayertwo.text = str(list_hidden[1])
                    btnlayerplusneuron.disabled = False
                elif len(list_hidden) == 2:
                    list_hidden.append(4)
                    self.drawingtryout(layoutneuronmainscreen, layoutlabels)
                    lblhiddenlayerthree.text = str(list_hidden[2])

                    layoutsettingsupleft.add_widget(btnlessneuronthree)
                    layoutsettingsupleft.add_widget(neuronthreelayout)
                    layoutsettingsupleft.add_widget(btnplusneuronthree)
                    lbllayer.text = str(len(list_hidden))
                    btnlayerplusneuron.disabled = True
        def btnlayerlessneuronpressed(btnlayerlessneuron):
            btnlayerplusneuron.disabled = False
            if len(list_hidden) <= 1:
                pass

            else:
                del list_hidden[-1]
                numlayers = str(len(list_hidden))
                lbllayer.text = numlayers
                self.drawingtryout(layoutneuronmainscreen, layoutlabels)
                if len(list_hidden) == 2:
                    layoutsettingsupleft.remove_widget(btnplusneuronthree)
                    layoutsettingsupleft.remove_widget(neuronthreelayout)
                    layoutsettingsupleft.remove_widget(btnlessneuronthree)
                    btnlayerlessneuron.disabled = False

                elif len(list_hidden) == 1:
                    layoutsettingsupleft.remove_widget(btnplusneurontwo)
                    layoutsettingsupleft.remove_widget(neurontwolayout)
                    layoutsettingsupleft.remove_widget(btnlessneurontwo)

                    btnlayerlessneuron.disabled = True

        def btnplusfoldspressed (btnh):
            global n_folds
            btnlessfolds.disabled = False
            if n_folds >= 10:
                n_folds = 10
            else:
                n_folds += 1
                numfolds.text = str(n_folds)
                if n_folds == 10:
                    btnplusfolds.disabled = True
        def btnlessfoldspressed(btnh):
            btnplusfolds.disabled = False
            global n_folds
            if n_folds <= 1:
                n_folds = 1
            else:
                n_folds = n_folds - 1
                numfolds.text = str(n_folds)
                if n_folds == 1:
                    btnlessfolds.disabled = True

        def btnplusratepressed (btnh):
            global l_rate
            btnlessrate.disabled = False
            if l_rate >= 1.0:
                l_rate = 1.0
            else:
                l_rate = l_rate + 0.1
                numrate.text = str("{:.1f}".format(l_rate))
                if l_rate == 1.0:
                    btnplusrate.disabled = True
        def btnlessratepressed(btnh):
            global l_rate
            btnplusrate.disabled = False
            if l_rate <= 0.1:
                l_rate = 0.1
            else:
                l_rate = l_rate - 0.1
                numrate.text = str("{:.1f}".format(l_rate))
                if l_rate == 0.1:
                    btnlessrate.disabled = True

        def btnplusepochpressed (btnh):
            global n_epoch
            btnlessepoch.disabled = False
            if n_epoch >= 500:
                n_epoch = 500
            else:
                n_epoch += 10
                numepoch.text = str(n_epoch)
                if n_epoch == 500:
                    btnplusepoch.disabled = True
        def btnlessepochpressed(btnh):
            global n_epoch
            btnplusepoch.disabled = False
            if n_epoch <= 10:
                n_epoch = 10
            else:
                n_epoch = n_epoch - 10
                numepoch.text = str(n_epoch)
                if n_epoch == 10:
                    btnlessepoch.disabled = True


        def btngopressed(bntGo):

            layoutneuronmainscreen = parentlayout.ids.idLayoutNeuron.ids.idLayoutNeuronMainScreen.ids.idrelativelayoutneuronmainscreen
            self.drawingvisualize(layoutneuronmainscreen)

            #layoutbtgo = parentlayout.ids.idLayoutNeuron.ids.idLayoutNeuronMainScreen
            #layoutbntgo = parentlayout.ids.idLayoutNeuron.ids.idLayoutNeuronMainScreen.ids.idrelativelayoutneuronmainscreen
            #self.drawingvisualize(layoutbntgo)
            layout.remove_widget(btnSettings)
            layout.remove_widget(btnPredict)
            layout.remove_widget(boxshowbiaslabel)
            layout.add_widget(btnnewinput)
            layout.add_widget(boxshowbiaslabel)

        def btnTrainpressed(btnTrain):
            global progressbar, accuracylabel, scores
            global network, n_inputs, n_outputs, n_folds
            cancellayout('h')
            layoutprogressbar = parentlayout.ids.mainseparatorprogressbar
            layoutprogressbar.remove_widget(progressbar)
            layoutprogressbar = parentlayout.ids.mainseparatorprogressbar
            layoutprogressbar.add_widget(progressbar)
            seed(1)
            '''dataset = [[1, 2, 3, 1],
                       [2, 1, 3, 2],
                       [1, 1, 1, 1],
                       [0, 0, 2, 0],
                       [1, 0, 2, 1],
                       [1, 3, 2, 1],
                       [0, 3, 2, 0],
                       [2, 2, 3, 2],
                       [2, 3, 1, 2],
                       [3, 1, 3, 3],
                       [3, 0, 2, 3],
                       [3, 3, 0, 3],
                       [2, 0, 1, 2],
                       [0, 2, 1, 0],
                       [3, 2, 2, 3],
                       [0, 1, 2, 0]]'''


            filename = 'seeds_dataset.csv'
            dataset = load_csv(filename)
            for i in range(len(dataset[0]) - 1):
                str_column_to_float(dataset, i)
            # convert class column to integers
            str_column_to_int(dataset, len(dataset[0]) - 1)
            # normalize input variables
            minmax = dataset_minmax(dataset)
            normalize_dataset(dataset, minmax)

            n_inputs = len(dataset[0]) - 1
            n_outputs = len(set([row[-1] for row in dataset]))
            network = initialize_network(n_inputs, list_hidden, n_outputs)
            train_network(network, dataset, 0.5, 20, n_outputs)
            # for row in dataset:
            #      prediction = predict(network, row)
            #      accuracy_metric(dataset,prediction)
            for layer in network:
                print(layer)


            # filename = 'seeds_dataset.csv'
            # dataset = load_csv(filename)
            # for i in range(len(dataset[0]) - 1):
            #     str_column_to_float(dataset, i)
            # # convert class column to integers
            # str_column_to_int(dataset, len(dataset[0]) - 1)
            # # normalize input variables
            # minmax = dataset_minmax(dataset)
            # normalize_dataset(dataset, minmax)
            # # evaluate algorithm
            #
            # threading.Thread(target=evaluate_algorithm, args=(dataset, back_propagation, n_folds, l_rate, n_epoch, list_hidden)).start()

            # scores = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, list_hidden)
            # print('Scores: %s' % scores)
            # print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
            # accuracylabel= (sum(scores) / float(len(scores)))


        def btnrefreshpressed(btnRefresh):
            global valuepb
            layout.remove_widget(btnRefresh)
            layout.remove_widget(btnPredict)
            layout.remove_widget(boxshowbiaslabel)
            layout.remove_widget(btnCancel)
            layout.add_widget(btnTrain)
            layout.add_widget(lblblacktwo)
            layout.add_widget(btnlessneuron)
            layout.add_widget(neuronlayout)
            layout.add_widget(btnplusneuron)
            layout.add_widget(lblblack)
            layout.add_widget(btnlayerlessneuron)
            layout.add_widget(numlayerslayout)
            layout.add_widget(btnlayerplusneuron)

            valuepb=0
            progressbar.value = 0

            imagescreen = parentlayout.ids.idLayoutNeuron.ids.idLayoutNeuronMainScreen.ids.idrelativelayoutneuronmainscreen
            backgroundscreen = parentlayout.ids.idLayoutNeuron.ids.idLayoutNeuronMainScreen
            clearbackground = (Image(source='background.png', width=backgroundscreen.width, allow_stretch = True))
            backgroundscreen.add_widget((clearbackground))

        def btncancelpressed():
            pass

        class MyEventDispatcher(EventDispatcher):
            def __init__(self, **kwargs):
                self.register_event_type('on_test')
                super(MyEventDispatcher, self).__init__(**kwargs)
            @mainthread
            def do_something(self, value):
                # when do_something is called, the 'on_test' event will be
                # dispatched with the value
                self.dispatch('on_test', value)

            def on_test(self, *args):
                global trainclock
                print ("I am dispatched", args)
                trainlayout('a')
                trainclock.cancel()
                layoutneuronmainscreen = parentlayout.ids.idLayoutNeuron.ids.idLayoutNeuronMainScreen.ids.idrelativelayoutneuronmainscreen
                layoutlabels = parentlayout.ids.idLayoutNeuron.ids.idLayoutNeuronMainScreen.ids.idLabelsLayoutNeuronMainScreen
                self.drawinglblblackkv(layoutneuronmainscreen, layoutlabels)

        def evaluate_algorithm(dataset, algorithm, n_folds, *args):
            global progressbar
            global scores, trainclock
            folds = cross_validation_split(dataset, n_folds)
            trainclock
            for fold in folds:
                train_set = list(folds)
                train_set.remove(fold)
                train_set = sum(train_set, [])
                test_set = list()
                for row in fold:
                    row_copy = list(row)
                    test_set.append(row_copy)
                    row_copy[-1] = None
                predicted = algorithm(train_set, test_set, *args)
                actual = [row[-1] for row in fold]
                accuracy = accuracy_metric(actual, predicted)
                scores.append(accuracy)
            ev = MyEventDispatcher()
            ev.do_something('hola')


        #box.active = True
        btnTrain.bind(on_press=btnTrainpressed)
        btnPredict.bind(on_press=btngopressed)
        btnplusneuron.bind(on_press=btnplusneuronpressed)
        btnlessneuron.bind(on_press=btnlessneuronpressed)
        btnplusneurontwo.bind(on_press=btnplusneurontwopressed)
        btnlessneurontwo.bind(on_press=btnlessneurontwopressed)
        btnplusneuronthree.bind(on_press=btnplusneuronthreepressed)
        btnlessneuronthree.bind(on_press=btnlessneuronthreepressed)
        btnlayerplusneuron.bind(on_press=btnlayerplusneuronpressed)
        btnlayerlessneuron.bind(on_press=btnlayerlessneuronpressed)
        btnSettings.bind(on_press=settingslayout)
        btnCancel.bind(on_press=btncancelpressed)
        btnOk.bind(on_press=mainlayout)
        btnBack.bind(on_press=mainlayout)
        #boxshowbiaslabel.bind(on_press=showbiaslabels)
        btnplusfolds.bind(on_press=btnplusfoldspressed)
        btnlessfolds.bind(on_press=btnlessfoldspressed)
        btnplusepoch.bind(on_press=btnplusepochpressed)
        btnlessepoch.bind(on_press=btnlessepochpressed)
        btnplusrate.bind(on_press=btnplusratepressed)
        btnlessrate.bind(on_press=btnlessratepressed)

        return parentlayout


class LayoutMenu(BoxLayout):
    pass

class LayoutNeuronMainScreen(BoxLayout):
    pass

class LayoutNeuron(AnchorLayout):
    pass

class LayoutScreen(BoxLayout):
    pass

class MyJoseApp(App):
    def build(self):

        screeninstance = LayoutScreen()
        for key, val in screeninstance.ids.items():
            print("key={0}, val={1}".format(key, val))
        menuboxlayout = screeninstance.ids.idLayoutMenu.ids.idBoxLayoutMenu
        return screeninstance




if __name__ == '__main__':
    MyApp().run()

'''
[{'weights': [0.09749164884035584, 0.8385365317026721, 0.7582938712917109, 0.23408540444090786], 'output': 0.9301660140324418, 'delta': -1.0868317911913742e-05}, 
{'weights': [0.47401163144760056, 0.4442572840305746, 0.6486117117173457, 0.7806034239677734], 'output': 0.925664521650442, 'delta': 5.3529719854897176e-05}, 
{'weights': [0.0016999371281541025, -0.0146219338790387, 0.8194260399511025, 0.3954879550202045], 'output': 0.8828008973000434, 'delta': 0.0001642104020448172}, 
{'weights': [0.74677795664685, -0.026641835249969176, 0.4213901331262488, 0.6903018365982354], 'output': 0.8187095048186223, 'delta': -0.000396122630591004}, 
{'weights': [0.23731249989414874, 0.9364853425123087, 0.8934273444398818, 0.004045930610650503], 'output': 0.9386825190062155, 'delta': -0.0003421270447944276}, 
{'weights': [-0.1575321337575302, 0.4801756307991974, 0.9212963506090881, 0.33230677443455353], 'output': 0.9342502547731141, 'delta': 0.0003955205190389014}]

[{'weights': [-0.02227376319714094, 0.16027938872533423, -0.1762775179002198, -0.034083468868553075, 0.19588599105418197, 0.2756735929922148, -0.047937224164803977], 'output': 0.583404980177467, 'delta': 0.000284483165467167}, 
{'weights': [0.04084529909734957, -0.0014940352433524973, 0.32159303123120075, 0.04274004258939806, -0.18257191187218783, 0.6858102804369285, 0.36047410133442515], 'output': 0.7598441235581141, 'delta': 0.011304977568919652}, 
{'weights': [0.5882148021013437, 0.12478078158317889, 0.943340606671976, 0.7994047621136161, 0.06658428560139971, 0.28189585715699317, 0.6536977004994179], 'output': 0.9580725477069769, 'delta': 0.000162867991615779}, 
{'weights': [0.6924990066003731, 0.9183365888460093, 0.4013364797083388, 0.8135408657943968, 0.652037386102589, 0.28189401990112156, 0.5664977312456262], 'output': 0.9812411657028608, 'delta': -0.00022802318276727519}, 
{'weights': [0.8750992581123791, 0.8325987754705224, 0.5024742126517208, 0.5683064500352958, 0.025104557711551123, 0.2413722845621395, 0.7895482922081741], 'output': 0.9714436306767351, 'delta': 0.0017786094154948846}, 
{'weights': [0.3157171373334583, 0.07170870679879857, 0.4461045706147117, 0.6242430091690476, 0.5830824276971796, 0.2670582177519248, 0.29900673986795867], 'output': 0.9152810341652716, 'delta': -0.006451183335242962}]

[{'weights': [0.00929631656959256, 0.2660402373251731, -0.030272521440975315, -0.1595748176088218, -0.05700199111287531, -0.5316975618935268, -0.4847919489820074], 'output': 0.2034345335421965, 'delta': 0.129082576818418}, 
{'weights': [-0.03742916921306206, 0.16953409254723287, -0.30790350088144025, -0.5265394009088484, -0.7400280822134505, -0.38575812948212673, 0.05097811822435465], 'output': 0.15890272476234263, 'delta': -0.02123777007006785},
 {'weights': [0.09661295371762528, -0.20430364420758038, -0.0025709274065064368, -0.6569895390030855, -0.36572687464820425, 0.1302369308730721, -0.3572102442182519], 'output': 0.22641788388146689, 'delta': -0.039657732159934005}, 
 {'weights': [-0.13565131793378424, -0.4353876035681212, -0.1740081781120843, 0.23130390419804386, -0.7144662055522422, 0.08790565655065458, 0.06468429110223625], 'output': 0.333662677470473, 'delta': -0.07418385541739748}]

'''



