


from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.checkbox import CheckBox
from kivy.uix.widget import Widget
from kivy.graphics import *
from datetime import datetime
from kivy.uix.gridlayout import GridLayout
from kivy.config import Config
from kivy.uix.textinput import TextInput
from kivy.graphics.texture import  *
from array import array
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.lang import Builder
from kivy.uix.slider import Slider
from kivy.uix.image import Image, AsyncImage
import numpy as np
from kivy.graphics.vertex_instructions import *
from random import seed
from random import random
from math import exp, sqrt
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
from kivy.uix.floatlayout import FloatLayout
from functools import wraps
import threading
import time
from kivy.factory import Factory
from kivy.event import EventDispatcher
from kivy.uix.filechooser import *
from kivy.uix.popup import Popup
import os
from os.path import dirname, join, basename, exists, expanduser

from loaddataset import showdataset, readdataset, str_column_to_float

numdata = 2
numoutputs = 1
n_outputs = 1
n_inputs = 2
dataset = 1
list_hidden = list()
listinputs = list()
input = list()
network = []
n_folds = 5
l_rate = 0.3
n_epoch = 20
prediction = False
varerrorlayout = False
mytexturedataset = ''
outputchanged = False
pags = 1
a = 0
b = 0
c = 0
d = 0
e = 1
g = 0
i = 0
j = 0
l = 0
pagenum = 1
valuepb = 0
valuetrain = 0
accuracylabel =''
scores = list()
biaseslabels = list()
trainclock = None
myevent = None
leftpanellist = list()
filename = ''
ti = TextInput(size=(40, 30), size_hint=(None, None))
ti2 = TextInput(size=(40, 30), size_hint=(None, None))
ti3 = TextInput(size=(40, 30), size_hint=(None, None))
progressbar = ProgressBar()
popup = None
currentpage = 0

hiddenlayeronelabelone = Label(text='[b]  Hidden\nlayer one', markup=True, font_size=15, color=(1,1,1), size_hint=(None, None))
hiddenlayeronelabeltwo = Label(text='[b]  Hidden\nlayer two', markup=True, size_hint=(None,None), font_size=15 , color=(1,1,1))
hiddenlayeronelabelthree = Label(text='[b]   Hidden\nlayer three', markup=True,size_hint=(None,None), font_size=15, color=(1,1,1))
outputlabel = Label(text='Output', markup=True, font_size=17, size_hint=(None, None), color=(1,1,1))
inputlabel = Label(text='Input', markup=True, font_size=17, size_hint=(None, None), color=(1,1,1))
lblcurrentpage = Label(text=('[b]Page:[/b]' + ' ' + str("{:.0f}".format(currentpage)) + '/' + '0'), markup=True)
lblfilename = Label(text='[b] Current file: [/b]' + '[i] no file selected', markup=True,pos_hint={'center_x': 0, 'center_y': .5} )
btnplussidetext = Button(text='[b]>', markup=True, size_hint=(.2, .8),
                         pos_hint={'center_x': 1, 'center_y': .5},
                         background_normal='ButtonPlusP.png',
                         background_down='ButtonDeactivated.png',
                         background_disabled_normal='ButtonDeactivated.png', font_size=20)
btnlesssidetext = Button(text='[b]<', markup=True, size_hint=(.2, .8),
                         pos_hint={'center_x': 1, 'center_y': .5},
                         background_normal='ButtonPlusP.png',
                         background_down='ButtonDeactivated.png',
                         background_disabled_normal='ButtonDeactivated.png', font_size=20)





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
def train_network(network, train, l_rate, n_epoch, n_outputs, type):
    global valuepb, myevent, valuetrain
    global n_folds
    step = 100.0/(n_folds*n_epoch)
    steptrain = 100.0/ n_epoch
    for epoch in range(n_epoch):
        sum_error = 0
        valuepb += step
        valuetrain += steptrain
        for row in train:
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
    if type == 'train':
        myevent.trainnetwork('hola')

#Works with btnGo
def predict(network, row):
    outputs = forward_propagate(network, row)
    print(outputs)
    print('..........')
    print(outputs.index(max(outputs)))
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
def layoutpbtrainnext(self):
    global valuetrain
    global progressbar
    progressbar.value = valuetrain


# Backpropagation Algorithm With Stochastic Gradient Descent
def back_propagation(train, test, l_rate, n_epoch, n_hidden):
    global network

    n_inputs = len(train[0]) - 1
    n_outputs = len(set([row[-1] for row in train]))
    network = initialize_network(n_inputs, n_hidden, n_outputs)
    train_network(network, train, l_rate, n_epoch, n_outputs, 'backp')
    predictions = list()
    for row in test:
        prediction = predict(network, row)
        predictions.append(prediction)
    return(predictions)




class MyApp(App):

    #When Predictpredictions is pressed
    def drawingvisualize(self, draw, *largs):
        global network, listinputs, input, numdata
        c = 0
        l = 0

        #draw.canvas.clear()
        #MyApp.drawinglblblackkv('j', layoutneuronmainscreen, layoutlabels)

        numoutputs = n_outputs
        numdata = n_inputs
        middlex = draw.width / 2
        middley = draw.height / 2 + 40
        space = numdata * 35 + (numdata - 1) * 25
        spaceoutput = numoutputs * 35 + (numoutputs - 1) * 25

        l = 0
        for element in listinputs:
            #inputtext = element.text
            #input.append(float(inputtext))
            inputlabel = Label(text='[b]' + element.text, markup=True, size=(30, 30), font_size=17, size_hint=(None, None),
                              pos=(middlex - 270, 50 * l + (middley+2 - space / 2)), color=(1, 1, 1, 1))
            draw.add_widget(inputlabel)
            draw.remove_widget(element)
            l += 1

        '''for x in range(numoutputs):
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
            c = c + 1'''

    # When Train is pressed
    def drawinglblblackkv(self, textlayout, mydata, *largs):
        global a, b, accuracylabel, progressbar, prediction, biaseslabels
        global hiddenlayeronelabelthree, hiddenlayeronelabelone, hiddenlayeronelabeltwo, outputlabel, inputlabel
        global n_epoch, n_folds, l_rate
        a = 0
        b = 0
        c = 0
        d = 0
        e = 0
        g = 0
        i = 0
        j = 0
        l = 0
        prediction = True

        draw = parentlayout.ids.idLayoutNeuron.ids.idLayoutNeuronMainScreen.ids.idrelativelayoutneuronmainscreen
        titlelayout = parentlayout.ids.idLayoutNeuron.ids.idLayoutNeuronMainScreen.ids.idLabelsLayoutNeuronMainScreen

        draw.canvas.clear()


        MyApp.leftlabels(self, textlayout, mydata, *largs)


        numoutputs = mydata.n_outputlbl


        space = numdata * 35 + (numdata - 1) * 25
        spaceone =mydata.list_hidden[0] * 35 + (mydata.list_hidden[0] - 1) * 25
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



        if len(mydata.list_hidden) == 1:
            # hiddenlayerone
            valuefloat = 0
            linecoloroutput = 0
            linecolorinput = 0
            hiddenlayeronelabelone.pos = (middletitlex - 28, middletitley - 50)
            titlelayout.add_widget(hiddenlayeronelabelone)
            for x in range(mydata.list_hidden[0]):
                if valuefloat < abs(network[0][a]['weights'][-1]):
                    valuefloat = abs(network[0][a]['weights'][-1])
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
            for x in range(mydata.list_hidden[0]):
                with draw.canvas:
                    value = mydata.networklist[0][a]['weights'][-1]
                    colorlayerone = abs(value) / valuefloat
                    Color(1,1,1, 0.25 + colorlayerone/1.33)
                    RoundedRectangle(size=(35, 35), pos=(middlex, a * 50 + (middley - spaceone / 2)))
                    Color(0,0,0,1)
                    RoundedRectangle(size=(30, 30), pos=(middlex+2.5, a * 50 + (middley + 2.5 - spaceone / 2)), radius=[(6,6),(6,6),(6,6),(6,6)])
                biases = "{:.2f}".format(value)
                nodeslabel = Label(text=biases, size_hint=(None,None), font_size=9, pos=(middlex-31.5, a * 50 + (middley-31.5 - spaceone / 2)),
                                        color=(1,1,1, 0.25 + colorlayerone/1.33))
                draw.add_widget(nodeslabel)
                biaseslabels.append(nodeslabel)
                i = 0
                e = 0
                if numdata < 10:
                    for y in range(numdata):
                        with draw.canvas:
                            Color(.5, .5, .5)
                            Line(points=[middlex - 205, i * 50 + (middley - 10 - space / 2) + 30, middlex,
                                         a * 50 + (middley + 20 - spaceone / 2)])
                            i = i + 1
                else:
                    for y in range(10):
                        with draw.canvas:
                            Color(.5, .5, .5)
                            Line(points=[middlex - 205, i * 50 + (middley - 10 - 575 / 2) + 30, middlex,
                                         a * 50 + (middley + 20 - spaceone / 2)])
                            i = i + 1
                for z in range(numoutputs):
                    color = abs(mydata.networklist[1][e]['weights'][a]) / linecoloroutput
                    with draw.canvas:
                        Color(color, color, color)
                        Line(points=[middlex+35, a * 50 + (middley  - spaceone / 2)+ 20, middlex+230,
                                     e * 50 + (middley+ 20 - spaceoutput / 2) ])
                    e = e + 1
                a = a + 1

        if len(mydata.list_hidden) == 2:
            # hiddenlayerone
            spacetwo = mydata.list_hidden[1] * 35 + (mydata.list_hidden[1] - 1) * 25
            valuefloat = 0
            linecoloroutput = 0
            linecolorinput = 0
            hiddenlayeronelabelone.pos = (middletitlex - 108, middletitley - 50)
            titlelayout.add_widget(hiddenlayeronelabelone)
            for x in range(mydata.list_hidden[0]):
                if valuefloat < abs(mydata.networklist[0][a]['weights'][-1]):
                    valuefloat = abs(mydata.networklist[0][a]['weights'][-1])
                e = 0
                i = 0
                for y in range(numdata):
                    if linecolorinput < abs(mydata.networklist[0][a]['weights'][i]):
                        linecolorinput = abs(mydata.networklist[0][a]['weights'][i])
                    i = i + 1
                a = a + 1
            a = 0
            for x in range(mydata.list_hidden[0]):
                with draw.canvas:
                    value = mydata.networklist[0][a]['weights'][-1]
                    colorlayerone = abs(mydata.networklist[0][a]['weights'][-1]) / valuefloat
                    Color(1, 1, 1, 0.25 + colorlayerone / 1.33)
                    RoundedRectangle(size=(35, 35), pos=(middlex-80, a * 50 + (middley - spaceone / 2)))
                    Color(0,0,0,1)
                    RoundedRectangle(size=(30, 30), pos=(middlex + 2.5 - 80, a * 50 + (middley + 2.5 - spaceone / 2)),
                                     radius=[(6, 6), (6, 6), (6, 6), (6, 6)])
                    biases = "{:.2f}".format(value)
                    draw.nodeslabel = Label(text=biases,size_hint=(None,None),font_size=10, pos=(middlex-111.5, a * 50 + (middley-31.5 - spaceone / 2)),
                                            color=(1,1,1, 0.25 + colorlayerone/1.33))
                    draw.add_widget(draw.nodeslabel)
                i = 0
                e = 0
                if numdata < 10:
                    for y in range(numdata):
                        with draw.canvas:
                            Color(.5, .5, .5)
                            Line(points=[middlex - 205, i * 50 + (middley - 10 - space / 2) + 30, middlex - 80,
                                         a * 50 + (middley + 20 - spaceone / 2)])
                            i = i + 1
                else:
                    for y in range(10):
                        with draw.canvas:
                            Color(.5, .5, .5)
                            Line(points=[middlex - 205, i * 50 + (middley - 10 - 575 / 2) + 30, middlex - 80,
                                         a * 50 + (middley + 20 - spaceone / 2)])
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
            for x in range(mydata.list_hidden[1]):
                if valuefloat < abs(mydata.networklist[1][d]['weights'][-1]):
                    valuefloat = abs(mydata.networklist[1][d]['weights'][-1])
                e = 0
                j = 0
                for y in range(mydata.list_hidden[0]):
                    if linecolorinput < abs(mydata.networklist[1][d]['weights'][j]):
                        linecolorinput = abs(mydata.networklist[1][d]['weights'][j])
                    j = j + 1
                for z in range(numoutputs):
                    if linecoloroutput < abs(mydata.networklist[2][e]['weights'][d]):
                        linecoloroutput = abs(mydata.networklist[2][e]['weights'][d])
                    e = e + 1
                d = d + 1
            d = 0
            for x in range(mydata.list_hidden[1]):
                with draw.canvas:
                    value = mydata.networklist[1][d]['weights'][-1]
                    colorlayerone = abs(mydata.networklist[1][d]['weights'][-1]) / valuefloat
                    Color(1, 1, 1, 0.25 + colorlayerone / 1.33)
                    RoundedRectangle(size=(35, 35), pos=(middlex + 80, d * 50 + (middley - spacetwo / 2)))
                    Color(0,0,0,1)
                    RoundedRectangle(size=(30, 30), pos=(middlex + 82.5, d * 50 + (middley + 2.5 - spacetwo / 2)),
                                     radius=[(6, 6), (6, 6), (6, 6), (6, 6)])
                    biases = "{:.2f}".format(value)
                    draw.nodeslabel = Label(text=biases, font_size=10,size_hint=(None,None), pos=(middlex +49, d * 50 + (middley-31.5 - spacetwo / 2)),
                                            color=(1,1,1, 0.25 + colorlayerone/1.33))
                    draw.add_widget(draw.nodeslabel)
                j = 0
                e = 0
                for y in range(mydata.list_hidden[0]):
                    color = abs(mydata.networklist[1][d]['weights'][j]) / linecolorinput
                    with draw.canvas:
                        Color(color, color, color)
                        Line(points=[middlex -45, j * 50 + (middley+20 - spaceone / 2), middlex+80, d * 50 + (middley+20 - spacetwo / 2)])
                        j = j + 1
                for z in range(numoutputs):
                    color = abs(mydata.networklist[2][e]['weights'][d]) / linecoloroutput
                    with draw.canvas:
                        Color(color, color, color)
                        Line(points=[middlex+115, d * 50 + (middley+20 - spacetwo / 2), middlex + 230, e * 50 + (middley +20 - spaceoutput / 2)])
                    e = e + 1
                d = d + 1

        if len(mydata.list_hidden) == 3:
            # hiddenlayerone
            spacethree = mydata.list_hidden[2] * 35 + (mydata.list_hidden[2] - 1) * 25
            spacetwo = mydata.list_hidden[1] * 35 + (mydata.list_hidden[1] - 1) * 25
            valuefloat = 0
            linecolorinput = 0
            hiddenlayeronelabelone.pos = (middletitlex - 158, middletitley - 50)
            titlelayout.add_widget(hiddenlayeronelabelone)
            for x in range(mydata.list_hidden[0]):
                if valuefloat < abs(mydata.networklist[0][a]['weights'][-1]):
                    valuefloat = abs(mydata.networklist[0][a]['weights'][-1])
                e = 0
                i = 0
                for y in range(numdata):
                    if linecolorinput < abs(mydata.networklist[0][a]['weights'][i]):
                        linecolorinput = abs(mydata.networklist[0][a]['weights'][i])
                    i = i + 1
                a = a + 1
            a = 0
            for x in range(mydata.list_hidden[0]):
                with draw.canvas:
                    value = mydata.networklist[0][a]['weights'][-1]
                    colorlayerone = abs(mydata.networklist[0][a]['weights'][-1]) / valuefloat
                    Color(1, 1, 1, 0.25 + colorlayerone / 1.33)
                    RoundedRectangle(size=(35, 35), pos=(middlex - 130, a * 50 + (middley - spaceone / 2)))
                    Color(0,0,0,1)
                    RoundedRectangle(size=(30, 30), pos=(middlex + 2.5 - 130, a * 50 + (middley + 2.5 - spaceone / 2)),
                                     radius=[(6, 6), (6, 6), (6, 6), (6, 6)])
                    biases = "{:.2f}".format(value)
                    draw.nodeslabel = Label(text=biases,size_hint=(None,None), font_size=10, pos=(middlex-161.5, a * 50 + (middley-31.5 - spaceone / 2)),
                                            color=(1,1,1, 0.25 + colorlayerone/1.33))
                    draw.add_widget(draw.nodeslabel)
                i = 0
                e = 0
                if numdata < 10:
                    for y in range(numdata):
                        with draw.canvas:
                            Color(.5, .5, .5)
                            Line(points=[middlex - 205, i * 50 + (middley - 10 - space / 2) + 30, middlex - 130,
                                         a * 50 + (middley + 20 - spaceone / 2)])
                            i = i + 1
                else:
                    for y in range(10):
                        with draw.canvas:
                            Color(.5, .5, .5)
                            Line(points=[middlex - 205, i * 50 + (middley - 10 - 575 / 2) + 30, middlex - 130,
                                         a * 50 + (middley + 20 - spaceone / 2)])
                            i = i + 1
                a = a + 1

            # hiddenlayertwo
            valuefloat = 0
            linecolorinput = 0
            d = 0
            a = 0
            hiddenlayeronelabeltwo.pos = (middletitlex - 28, middletitley - 50)
            titlelayout.add_widget(hiddenlayeronelabeltwo)
            for x in range(mydata.list_hidden[1]):
                if valuefloat < abs(mydata.networklist[1][d]['weights'][-1]):
                    valuefloat = abs(mydata.networklist[1][d]['weights'][-1])
                e = 0
                j = 0
                for y in range(mydata.list_hidden[0]):
                    if linecolorinput < abs(mydata.networklist[1][d]['weights'][j]):
                        linecolorinput = abs(mydata.networklist[1][d]['weights'][j])
                    j = j + 1
                d = d + 1
            d = 0
            for x in range(mydata.list_hidden[1]):
                with draw.canvas:
                    value = mydata.networklist[1][d]['weights'][-1]
                    colorlayerone = abs(mydata.networklist[1][d]['weights'][-1]) / valuefloat
                    Color(1, 1, 1, 0.25 + colorlayerone / 1.33)
                    RoundedRectangle(size=(35, 35), pos=(middlex, d * 50 + (middley - spacetwo / 2)))
                    Color(0,0,0,1)
                    RoundedRectangle(size=(30, 30), pos=(middlex + 2.5, d * 50 + (middley + 2.5 - spacetwo / 2)),
                                     radius=[(6, 6), (6, 6), (6, 6), (6, 6)])
                    biases = "{:.2f}".format(value)
                    draw.nodeslabel = Label(text=biases, font_size=10,size_hint=(None,None), pos=(middlex-31.5, d * 50 + (middley-31.5 - spacetwo / 2)),
                                            color=(1,1,1, 0.25 + colorlayerone/1.33))
                    draw.add_widget(draw.nodeslabel)
                j = 0
                e = 0
                for y in range(mydata.list_hidden[0]):
                    color = abs(mydata.networklist[1][d]['weights'][j]) / linecolorinput
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
            for x in range(mydata.list_hidden[2]):
                if valuefloat < abs(mydata.networklist[2][d]['weights'][-1]):
                    valuefloat = abs(mydata.networklist[2][d]['weights'][-1])

                e = 0
                j = 0
                for y in range(mydata.list_hidden[1]):
                    if linecolorinput < abs(mydata.networklist[2][d]['weights'][j]):
                        linecolorinput = abs(mydata.networklist[2][d]['weights'][j])
                    j = j + 1
                for z in range(numoutputs):
                    if linecoloroutput < abs(mydata.networklist[3][e]['weights'][d]):
                        linecoloroutput = abs(mydata.networklist[3][e]['weights'][d])
                    e = e + 1
                d = d + 1
            d = 0
            for x in range(mydata.list_hidden[2]):
                with draw.canvas:
                    value = mydata.networklist[2][d]['weights'][-1]
                    colorlayerone = abs(mydata.networklist[2][d]['weights'][-1]) / valuefloat
                    Color(1, 1, 1, 0.25 + colorlayerone / 1.33)
                    RoundedRectangle(size=(35, 35), pos=(middlex+130, d * 50 + (middley - spacethree / 2)))
                    Color(0,0,0,1)
                    RoundedRectangle(size=(30, 30), pos=(middlex + 132.5, d * 50 + (middley + 2.5 - spacethree / 2)),
                                     radius=[(6, 6), (6, 6), (6, 6), (6, 6)])
                    biases = "{:.2f}".format(value)
                    draw.nodeslabel = Label(text=biases, font_size=10,size_hint=(None,None), pos=(middlex+99, d * 50 + (middley-31.5 - spacethree / 2)),
                                            color=(1,1,1, 0.25 + colorlayerone/1.33))
                    draw.add_widget(draw.nodeslabel)
                l = 0
                e = 0
                for y in range(mydata.list_hidden[1]):
                    color = abs(mydata.networklist[2][d]['weights'][l]) / linecolorinput
                    with draw.canvas:
                        Color(color, color, color)
                        Line(points=[middlex+35, l * 50 + (middley+20 - spacetwo / 2), middlex+130, d * 50 + (middley+20 - spacethree / 2)])
                        l = l + 1
                for z in range(numoutputs):
                    color = abs(mydata.networklist[3][e]['weights'][d]) / linecoloroutput
                    with draw.canvas:
                        Color(color, color, color)
                        Line(points=[middlex+165, d * 50 + (middley+20 - spacethree / 2), middlex+230, e * 50 + (middley+20 - spaceoutput / 2)])
                    e = e + 1
                d = d + 1

        # output layer
        for x in range(numoutputs):
            outputoptions = str(c)
            lbloutputoptions = Label(text='[b]' + outputoptions, markup=True, size_hint=(None,None),
                                     pos=(middlex+230, c * 50 + (middley -30 - spaceoutput / 2)), font_size = 17, color=(1, 1, 1, 1))
            draw.add_widget(lbloutputoptions)
            value = mydata.networklist[-1][c]['weights'][-1]
            with draw.canvas:
                Color(.67,.57,.69)
                RoundedRectangle(size=(35, 35), pos=(middlex + 230, c * 50 + (middley - spaceoutput / 2)))
                Color(0, 0, 0)
                RoundedRectangle(size=(30, 30), pos=(middlex + 232.5, c * 50 + (middley+2.5 - spaceoutput / 2)), radius=[(6,6),(6,6),(6,6),(6,6)])
                biases = "{:.2f}".format(value)
                #draw.nodeslabel = Label(text=biases, font_size=10, pos=(middlex + 196, c * 50 + (middley-34 - spaceoutput / 2)),color=(1, 1, 1, .9))
            c = c + 1
        # input layer
        if numdata < 10:
            for x in range(numdata):
                with draw.canvas:
                    Color(.6, .87, .89)
                    RoundedRectangle(size=(35, 35), pos=(middlex - 240, b * 50 + (middley - space / 2)))
                    Color(0, 0, 0)
                    RoundedRectangle(size=(30, 30), pos=(middlex - 237.5, b * 50 + (middley + 2.5 - space / 2)),
                                     radius=[(6, 6), (6, 6), (6, 6), (6, 6)])
                    b = b + 1
        else:
            for x in range(int(numdata / 2 - 1)):
                with draw.canvas:
                    Color(.6, .87, .89)
                    RoundedRectangle(size=(35, 35), pos=(middlex - 240, b * 50 + (middley - 575 / 2)))
                    Color(0, 0, 0)
                    RoundedRectangle(size=(30, 30), pos=(middlex - 237.5, b * 50 + (middley + 2.5 - 575 / 2)),
                                     radius=[(6, 6), (6, 6), (6, 6), (6, 6)])
                    Color(.6, .87, .89)
                    RoundedRectangle(size=(35, 35), pos=(middlex - 240, (middley + 575 / 2) - b * 50 - 123))
                    Color(0, 0, 0)
                    RoundedRectangle(size=(30, 30), pos=(middlex - 237.5, (middley + 2.5 + 575 / 2) - b * 50 - 123),
                                     radius=[(6, 6), (6, 6), (6, 6), (6, 6)])
                    b = b + 1
                    Color(.4 + b / 9, .67 + b / 9, .69 + b / 9)
                    Ellipse(pos=(middlex - 225, middley - b * 20), size=(5, 5))


    #When btnOK is pressed. Plain normal size
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

                if numdata < 10:
                    for y in range(numdata):
                        with draw.canvas:
                            Color(.5,.5,.5)
                            Line(points=[middlex - 205, i * 50 + (middley-10 - space / 2) + 30, middlex, a * 50 + (middley+20 - spaceone / 2) ])
                            i = i + 1
                else:
                    for y in range(10):
                        with draw.canvas:
                            Color(.5,.5,.5)
                            Line(points=[middlex - 205, i * 50 + (middley-10 - 575 / 2) + 30, middlex, a * 50 + (middley+20 - spaceone / 2) ])
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
                if numdata < 10:
                    for y in range(numdata):
                        with draw.canvas:
                            Color(.5, .5, .5)
                            Line(points=[middlex - 205, i * 50 + (middley-10 - space / 2) + 30, middlex-80, a * 50 + (middley + 20 - spaceone / 2) ])
                            i = i + 1
                else:
                    for y in range(10):
                        with draw.canvas:
                            Color(.5, .5, .5)
                            Line(points=[middlex - 205, i * 50 + (middley-10 - 575 / 2) + 30, middlex-80, a * 50 + (middley + 20 - spaceone / 2) ])
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
                if numdata < 10:
                    for y in range(numdata):
                        with draw.canvas:
                            Color(.5, .5, .5)
                            Line(points=[middlex - 205, i * 50 + (middley-10 - space / 2) + 30, middlex-130,
                                     a * 50 + (middley+ 20 - spaceone / 2)  ])
                            i = i + 1
                else:
                    for y in range(10):
                        with draw.canvas:
                            Color(.5, .5, .5)
                            Line(points=[middlex - 205, i * 50 + (middley-10 - 575 / 2) + 30, middlex-130,
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
                Color(.67,.57,.69)
                RoundedRectangle(size=(35, 35), pos=(middlex + 230, c * 50 + (middley - spaceoutput / 2)))
                Color(0, 0, 0)
                RoundedRectangle(size=(30, 30), pos=(middlex + 232.5, c * 50 + (middley+2.5 - spaceoutput / 2)), radius=[(6,6),(6,6),(6,6),(6,6)])
            c = c + 1

        # input layer
        if numdata < 10:
            for x in range(numdata):
                with draw.canvas:
                    Color(.6,.87,.89)
                    RoundedRectangle(size=(35, 35), pos=(middlex - 240, b * 50 + (middley - space / 2)))
                    Color(0, 0, 0)
                    RoundedRectangle(size=(30, 30), pos=(middlex - 237.5, b * 50 + (middley + 2.5 - space / 2)),
                                     radius=[(6, 6), (6, 6), (6, 6), (6, 6)])
                    b = b + 1
        else:
            for x in range(4):
                with draw.canvas:
                    Color(.6,.87,.89)
                    RoundedRectangle(size=(35, 35), pos=(middlex - 240, b * 50 + (middley - 575 / 2)))
                    Color(0, 0, 0)
                    RoundedRectangle(size=(30, 30), pos=(middlex - 237.5, b * 50 + (middley + 2.5 - 575 / 2)),
                                     radius=[(6, 6), (6, 6), (6, 6), (6, 6)])
                    Color(.6, .87, .89)
                    RoundedRectangle(size=(35, 35), pos=(middlex - 240, (middley + 575 / 2) - b * 50 - 123 ))
                    Color(0, 0, 0)
                    RoundedRectangle(size=(30, 30), pos=(middlex - 237.5, (middley + 2.5 +  575 / 2) - b * 50 - 123),
                                     radius=[(6, 6), (6, 6), (6, 6), (6, 6)])
                    b = b + 1
                    Color(.4+b/9,.67+b/9,.69+b/9)
                    Ellipse(pos=(middlex - 225, middley-b*20),size= (5,5))
                draw.add_widget(Label(text=str(numdata), pos=(middlex - 320, middley-96), size_hint=(None,None)))
                    #Point(points=(200,200),pointsize=2)

    class Lblleft():

        def __init__(self, type, dataerror):
            global dataset, scores, varerrorlayout, n_epoch, n_folds, l_rate
            global list_hidden, network, n_outputs, n_inputs, currentpage, list_hidden

            self.datestr = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            self.networklist = network.copy()
            self.n_outputlbl = n_outputs
            self.n_inputlbl = n_inputs
            self.filename = filename
            self.type = type
            self.n_epoch = n_epoch
            self.n_folds = n_folds
            self.l_rate = l_rate
            self.dataerror = dataerror
            self.list_hidden = list_hidden.copy()
            leftpanellist.append(self)
            currentpage = len(leftpanellist)
            MyApp.establishbtnpages(self)

    #When btnError is pressed
    #The lbls on the left
    def leftlabels(self, textlayout, leftpanelitem, *largs):
        global dataset, scores, varerrorlayout, n_epoch, n_folds, l_rate
        global list_hidden, network, n_outputs, n_inputs, currentpage


        lbldate= Label(text=leftpanelitem.datestr, markup=True, halign="left", size_hint=(1, .1))
        lbldate.bind(size=lbldate.setter('text_size'))
        labelfile = (Label(text='[b] File: [/b][i]' + leftpanelitem.filename, markup=True, halign="left",
                           size_hint=(1, .1)))
        labelfile.bind(size=labelfile.setter('text_size'))
        labeltype = (Label(text='[b] [color=4BDCFF]Test: [/b][/color]' + leftpanelitem.type, markup=True,
                           halign="left",
                           size_hint=(1, .1), color=(1, 1, 1)))
        labeltype.bind(size=labeltype.setter('text_size'))
        labelcycles = (Label(text='[b] [color=D464EB]Cycles: [/b][/color]' + str(leftpanelitem.n_epoch), markup=True,
                             halign="left",
                             size_hint=(1, .1), color=(1, 1, 1)))
        labelcycles.bind(size=labelcycles.setter('text_size'))
        labelrate = (Label(text='[b] [color=D464EB]Rate: [/b][/color]' + str(leftpanelitem.l_rate), markup=True,
                           halign="left",
                           size_hint=(1, .1), color=(1, 1, 1)))
        labelrate.bind(size=labelrate.setter('text_size'))
        labelfolds = (Label(text='[b] [color=D464EB]Folds: [/b][/color]' + str(leftpanelitem.n_folds), markup=True,
                            halign="left",
                            size_hint=(1, .1), color=(1, 1, 1)))
        labelfolds.bind(size=labelfolds.setter('text_size'))
        varerrorlayout = True
        labelaccuracy = (Label(text='[b] Accuracy: [/b]' + str("{:.2f}".format(leftpanelitem.dataerror)) + '%', markup=True,
                                   halign="left", size_hint=(1, .1)))
        labelaccuracy.bind(size=labelaccuracy.setter('text_size'))




        if currentpage == 1:
            btnlesssidetext.disabled = True
        if currentpage != 1:
            btnlesssidetext.disabled = False

        textlayout.clear_widgets()

        textlayout.add_widget(lbldate)
        textlayout.add_widget(labelfile)
        textlayout.add_widget(labeltype)
        textlayout.add_widget(labelcycles)
        textlayout.add_widget(labelrate)
        textlayout.add_widget(labelfolds)
        textlayout.add_widget(labelaccuracy)


        lblcurrentpage.text = ('[b]Page:[/b]' + ' ' + str("{:.0f}".format(currentpage)) + '/' + str("{:.0f}".format(len(leftpanellist))))

    def establishbtnpages(self):
        if currentpage == 1:
            btnlesssidetext.disabled = True
        else:
            btnlesssidetext.disabled = False
        if currentpage == len(leftpanellist):
            btnplussidetext.disabled = True
        else:
            btnplussidetext.disabled = False

    #Change between pages
    def leflabelswitch(self, textlayout, dataerror, type, *largs):
        global currentpage, btnplussidetext, btnlesssidetext
        if type == 'btnplus':
            currentpage += 1
            mydata = leftpanellist[currentpage -1]
            MyApp.drawinglblblackkv(self, textlayout, mydata)
        if type == 'btnless':
            currentpage -= 1
            mydata = leftpanellist[currentpage - 1]
            MyApp.drawinglblblackkv(self, textlayout, mydata)
        MyApp.establishbtnpages(self)

        lblcurrentpage.text = ('[b]Page:[/b]' + ' ' + str("{:.0f}".format(currentpage)) + '/' + str(
            "{:.0f}".format(len(leftpanellist))))

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
                if numdata < 10:
                    for y in range(numdata):
                        with draw.canvas:
                            Color(.5,.5,.5)
                            Line(points=[middlex - 220, i * 30 + (middley+10 - space / 2), middlex, a * 30 + (middley+10 - spaceone / 2) ])
                            i = i + 1
                else:
                    for y in range(10):
                        with draw.canvas:
                            Color(.5,.5,.5)
                            Line(points=[middlex - 220, i * 30 + (middley+10 - 245 / 2), middlex, a * 30 + (middley+10 - spaceone / 2) ])
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
                if numdata < 10:
                    for y in range(numdata):
                        with draw.canvas:
                            Color(.5,.5,.5)
                            Line(points=[middlex - 220, i * 30 + (middley-20 - space / 2) + 30, middlex-80, a * 30 + (middley + 10 - spaceone / 2) ])
                            i = i + 1
                else:
                    for y in range(10):
                        with draw.canvas:
                            Color(.5,.5,.5)
                            Line(points=[middlex - 220, i * 30 + (middley-20 - 245 / 2)+30, middlex-80, a * 30 + (middley+10 - spaceone / 2) ])
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
                if numdata < 10:
                    for y in range(numdata):
                        with draw.canvas:
                            Color(.5,.5,.5)
                            Line(points=[middlex - 220, i * 30 + (middley-20 - space / 2) + 30, middlex-130,
                                     a * 30 + (middley+ 10 - spaceone / 2) ])
                            i = i + 1
                else:
                    for y in range(10):
                        with draw.canvas:
                            Color(.5,.5,.5)
                            Line(points=[middlex - 220, i * 30 + (middley -20 - 245 / 2) + 30, middlex - 130, a * 30 + (middley+10 - spaceone / 2) ])
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
                Color(.67+c/9,.57+c/9,.69+c/9)
                RoundedRectangle(size=(20, 20), pos=(middlex + 230, c * 30 + (middley - spaceoutput / 2)))
                Color(0, 0, 0)
                RoundedRectangle(size=(15, 15), pos=(middlex + 232.5, c * 30 + (middley+2.5 - spaceoutput / 2)), radius=[(6,6),(6,6),(6,6),(6,6)])
            c = c + 1
        # input layer
        if numdata < 10:
            for x in range(numdata):
                with draw.canvas:
                    Color(.6,.87,.89)
                    RoundedRectangle(size=(20, 20), pos=(middlex - 240, b * 30 + (middley - space / 2)))
                    Color(0, 0, 0)
                    RoundedRectangle(size=(15, 15), pos=(middlex - 237.5, b * 30 + (middley + 2.5 - space / 2)),
                                     radius=[(6, 6), (6, 6), (6, 6), (6, 6)])
                    b = b + 1
        else:
            for x in range(4):
                with draw.canvas:
                    Color(.6,.87,.89)
                    RoundedRectangle(size=(20, 20), pos=(middlex - 240, b * 30 + (middley - 245 / 2)))
                    Color(0, 0, 0)
                    RoundedRectangle(size=(15, 15), pos=(middlex - 237.5, b * 30 + (middley + 2.5 - 245 / 2)),
                                     radius=[(6, 6), (6, 6), (6, 6), (6, 6)])
                    Color(.6, .87, .89)
                    RoundedRectangle(size=(20, 20), pos=(middlex - 240, (middley + 245 / 2) - b * 30 +25 ))
                    Color(0, 0, 0)
                    RoundedRectangle(size=(15, 15), pos=(middlex - 237.5, (middley + 2.5 +  245 / 2) - b * 30 +25 ),
                                     radius=[(6, 6), (6, 6), (6, 6), (6, 6)])
                    b = b + 1
                    Color(.4+b/9,.67+b/9,.69+b/9)
                    Ellipse(pos=(middlex - 233, middley-b*10+42),size= (4,4))
                draw.add_widget(Label(text=str(numdata), pos=(middlex - 320, middley-30), size_hint=(None,None)))
                    #Point(points=(200,200),pointsize=2)

    #Adding TextInputs
    def drawinginputs(self,draw,titlelayout, *largs):
        global listinputs, numdata
        middlex = draw.width / 2
        middley = draw.height / 2 + 40
        space = numdata * 35 + (numdata - 1) * 25
        listinputs = list()

        a = 0
        for x in range(numdata):
            x = TextInput(size=(40,30), size_hint=(None, None), pos = (middlex - 290, 50 * a + (middley + 4 - space / 2)))
            draw.add_widget(x)
            a += 1
            listinputs.append(x)

        '''ti.pos = (90, (middley + 5 - space / 2))
        draw.add_widget(ti)
        ti2.pos = (90, 50 + (middley + 5 - space / 2))
        draw.add_widget(ti2)
        ti3.pos = (90, 2 * 50 + (middley + 5 - space / 2))
        draw.add_widget(ti3)'''

    #Clear canvas
    def clear(self,draw,titlelayout, *largs):
        draw.canvas.clear()
        titlelayout.remove_widget(hiddenlayeronelabelone)
        titlelayout.remove_widget(hiddenlayeronelabeltwo)
        titlelayout.remove_widget(hiddenlayeronelabelthree)
        titlelayout.remove_widget(outputlabel)
        titlelayout.remove_widget(inputlabel)

    def build(self):
        global layoutneuronmainscreen, layoutlabels, parentlayout, myevent, lblfilename, btnlesssidetext, btnplussidetext, lblcurrentpage
        class MyEventDispatcher(EventDispatcher):
            def __init__(self, **kwargs):
                self.register_event_type('on_test')
                self.register_event_type('on_trainfinished')
                super(MyEventDispatcher, self).__init__(**kwargs)
            @mainthread
            def evevaluatealgorithm(self, value):
                self.dispatch('on_test', value)

            def on_test(self, *args):
                global trainclock
                trainclock.cancel()
                layoutprogressbar = parentlayout.ids.mainseparatorprogressbar
                layoutprogressbar.remove_widget(progressbar)
                progressbar.value = 0
                print('Scores: %s' % scores)
                print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
                accuracylabel = (sum(scores) / float(len(scores)))
                errorlayout(accuracylabel)

            @mainthread
            def trainnetwork(self, value):
                self.dispatch('on_trainfinished', value)

            def on_trainfinished(self, *args):
                global trainclock
                trainclock.cancel()
                layoutprogressbar = parentlayout.ids.mainseparatorprogressbar
                layoutprogressbar.remove_widget(progressbar)
                progressbar.value = 0
                trainlayout(btnTrain)
                for layer in network:
                    print(layer)

            def drawingmainlayout(self):
                MyApp.drawingblank(btnOk, layoutneuronmainscreen, layoutlabels)

        list_hidden.append(6)
        numlayers = str(len(list_hidden))
        myevent = MyEventDispatcher()

        parentlayout = LayoutScreen()
        layoutneuronmainscreen = parentlayout.ids.idLayoutNeuron.ids.idLayoutNeuronMainScreen.ids.idrelativelayoutneuronmainscreen
        layoutlabels = parentlayout.ids.idLayoutNeuron.ids.idLayoutNeuronMainScreen.ids.idLabelsLayoutNeuronMainScreen
        layoutmainscreen = parentlayout.ids.idLayoutNeuron.ids.idLayoutNeuronMainScreen
        layout = parentlayout.ids.idLayoutMenu.ids.idBoxLayoutMenu
        btnlayout = parentlayout.ids.idLayoutMenu
        layoutinfoleft = parentlayout.ids.idboxlayoutleft
        layoutinforight = parentlayout.ids.idboxlayoutright




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
        layoutinfotop = parentlayout.ids.idlayouttop
        layoutinfomiddle = parentlayout.ids.idlayoutmiddle
        layoutinfodown = parentlayout.ids.idlayoutdown
        layoutbtninfo = parentlayout.ids.idlayoutlbldown
        layoutfilenameright = parentlayout.ids.idlayoutrightdown






        btnPredict = Button(color=(1, 1, 1, 1), text='[b]PREDICT', markup = True,background_disabled_normal='ButtonDeactivated.png', font_size=18, size_hint=(.95, .9), pos_hint={'center_x':.5, 'center_y':.5}, background_normal='ButtonPlusP.png', background_down='ButtonDeactivated.png')
        btnProcessPredict = Button(color=(1, 1, 1, 1), text='[b]PREDICT', markup = True,background_disabled_normal='ButtonDeactivated.png', font_size=18, size_hint=(.95, .9), pos_hint={'center_x':.5, 'center_y':.5}, background_normal='ButtonPlusP.png', background_down='ButtonDeactivated.png')
        btnTrain = Button(text='[b]TRAIN',markup = True, background_normal='ButtonPlusP.png', background_down='ButtonDeactivated.png', size_hint=(.95,.9),
                       pos_hint={'center_x':.5, 'center_y':.5}, color = (1,1,1,1), font_size=18)
        btnError = Button(text='[b]ERROR', markup=True, background_normal='ButtonPlusP.png',
                          background_down='ButtonDeactivated.png', size_hint=(.95, .9),
                          pos_hint={'center_x': .5, 'center_y': .5},background_disabled_normal='ButtonDeactivated.png', color=(1, 1, 1, 1), font_size=18)
        btnOk = Button(text='[b]OK', markup=True, background_normal='ButtonPlusB.png',
                          background_down='ButtonDeactivated.png', size_hint=(.8, .8),
                          pos_hint={'center_x': .5, 'center_y': .5}, color=(0, 0, 0, 1), font_size=18)
        btnOpen = Button(text='[b]OPEN', markup=True, background_normal='ButtonPlusP.png',
                       background_down='ButtonPlusGDown.png', size_hint=(.8, .8),
                       pos_hint={'center_x': .5, 'center_y': .5}, color=(1, 1, 1, 1), font_size=18)
        btnBack = Button(text='[b]', markup=True, background_normal='ButtonBack.png',
                       background_down='ButtonBack.png', size_hint=(None, None), size=(60,60),
                       pos_hint={'center_x': 1, 'center_y': .5}, color=(1, 1, 1, 1),background_disabled_normal='ButtonBackDisabled.png')
        btnSettings = Button(background_normal='ButtonSettings.png', background_down='ButtonSettings.png', size_hint=(None, None), size=(60,60), pos_hint={'center_x':1, 'center_y':.5},background_disabled_normal='ButtonSettingsDisabled.png')
        btnnewinput=Button(color=(1,1,1,1), text='[b]New Input',markup = True, font_size=18, size_hint=(.95,.9),
                       pos_hint={'center_x':.5, 'center_y':.5},background_normal='ButtonPlusP.png', background_down='ButtonDeactivated.png')
        btnCancel = Button(color=(1,1,1,1), text='[b]CANCEL',markup = True, font_size=18, size_hint=(.95,.9), pos_hint={'center_x':.5, 'center_y':.5},background_normal='ButtonPlusP.png', background_down='ButtonDeactivated.png')



        box = CheckBox()
        lblbox = Label(text='Sigmoid:', color=(1, 1, 1, 1), font_size=(14))
        boxshowbiaslabel = CheckBox()

        # Left Neuron Switcher
        btnplusneuron = Button(text = '[b]+',markup = True, size_hint=(.2,.8),pos_hint={'center_x':.5, 'center_y':.5},
                               background_normal='ButtonPlusP.png',
                               background_down='ButtonDeactivated.png',
                               background_disabled_normal='ButtonDeactivated.png', font_size=20)
        btnlessneuron = Button(text = '[b]-',markup = True,size_hint=(.2, .8),pos_hint={'center_x':.5, 'center_y':.5},
                               background_normal='ButtonPlusP.png',
                               background_down='ButtonDeactivated.png',
                               background_disabled_normal='ButtonDeactivated.png', font_size=20)
        lblhiddenlayer = Label(text=str(list_hidden[0]))
        lblneuron = Label(text='[b]Neurons', markup=True)

        # Second neuron switcher
        btnplusneurontwo = Button(text = '[b]+',markup = True,size_hint=(.2, .8),pos_hint={'center_x':.5, 'center_y':.5},
                                  background_normal='ButtonPlusP.png',
                                  background_down='ButtonDeactivated.png',
                                  background_disabled_normal='ButtonDeactivated.png', font_size=20)
        btnlessneurontwo = Button(text = '[b]-',markup = True,size_hint=(.2, .8), pos_hint={'center_x':.5, 'center_y':.5},
                                  background_normal='ButtonPlusP.png',
                                  background_down='ButtonDeactivated.png',
                                  background_disabled_normal='ButtonDeactivated.png', font_size=20)
        lblhiddenlayertwo = Label(text='4')
        lblneurontwo = Label(text='[b]Neurons', markup=True)

        # Right/Third neuron switcher
        btnplusneuronthree = Button(text = '[b]+',markup = True,size_hint=(.2, .8),pos_hint= {'center_x':1, 'center_y':.5},
                                    background_normal='ButtonPlusP.png',
                                    background_down='ButtonDeactivated.png',
                                    background_disabled_normal='ButtonDeactivated.png', font_size=20)
        btnlessneuronthree = Button(text = '[b]-',markup = True,size_hint=(.2, .8),pos_hint={'center_x':1, 'center_y':.5},
                                    background_normal='ButtonPlusP.png',
                                    background_down='ButtonDeactivated.png',
                                    background_disabled_normal='ButtonDeactivated.png', font_size=20)
        lblhiddenlayerthree = Label(text='4')
        lblneuronthree = Label(text='[b]Neurons', markup=True)

        # Number of layers switcher
        btnlayerplusneuron = Button(text = '[b]+',markup = True,size_hint=(.2, .8),pos_hint={'center_x':1, 'center_y':.5},
                                    background_normal='ButtonPlusP.png',
                                    background_down='ButtonDeactivated.png',
                                    background_disabled_normal='ButtonDeactivated.png', font_size=20)
        btnlayerlessneuron = Button(text = '[b]-',markup = True,size_hint=(.2, .8),pos_hint={'center_x':1, 'center_y':.5},
                                    background_normal='ButtonPlusP.png',
                                    background_down='ButtonDeactivated.png',
                                    background_disabled_normal='ButtonDeactivated.png', font_size=20)
        lbllayer = Label(text=numlayers)
        lblnumlayers = Label(text='[b] Hiddenlayers',markup = True, font_size=(11))
        lblblack = Widget(size_hint=(.1,.1))
        lblblacktwo = Widget(size_hint=(.1,.1))
        lblblackthree = Widget(size_hint=(.1,.1))
        canvas= Widget(size_hint=(.02,.1))
        visualize = Label(size_hint=(.1,.1))

        btnplusepoch = Button(text='[b]+', markup=True, size_hint=(.2, .8),
                              pos_hint={'center_x': 1, 'center_y': .5},
                              background_normal='ButtonPlusP.png',
                              background_down='ButtonDeactivated.png',
                              background_disabled_normal='ButtonDeactivated.png', font_size=20)
        btnlessepoch = Button(text='[b]-', markup=True, size_hint=(.2, .8),
                                    pos_hint={'center_x': 1, 'center_y': .5},
                                    background_normal='ButtonPlusP.png',
                                    background_down='ButtonDeactivated.png',
                                    background_disabled_normal='ButtonDeactivated.png', font_size=20)
        lblepoch = Label(text='[b] CYCLES:',markup=True, font_size=(11), pos_hint={'center_x':.5, 'center_y':.5})
        numepoch = Label(text=str(n_epoch))
        btnplusrate = Button(text='[b]+', markup=True, size_hint=(.2, .8),
                              pos_hint={'center_x': 1, 'center_y': .5},
                              background_normal='ButtonPlusP.png',
                              background_down='ButtonDeactivated.png',
                              background_disabled_normal='ButtonDeactivated.png', font_size=20)
        btnlessrate = Button(text='[b]-', markup=True, size_hint=(.2, .8),
                              pos_hint={'center_x': 1, 'center_y': .5},
                              background_normal='ButtonPlusP.png',
                              background_down='ButtonDeactivated.png',
                              background_disabled_normal='ButtonDeactivated.png', font_size=20)
        lblrate = Label(text='[b]RATE:',markup=True, font_size=(11), pos_hint={'center_x': .5, 'center_y': .5})
        numrate = Label(text=str(l_rate))
        btnplusfolds = Button(text='[b]+', markup=True, size_hint=(.2, .8),
                             pos_hint={'center_x': 1, 'center_y': .5},
                             background_normal='ButtonPlusP.png',
                             background_down='ButtonDeactivated.png',
                             background_disabled_normal='ButtonDeactivated.png', font_size=20)
        btnlessfolds = Button(text='[b]-', markup=True, size_hint=(.2, .8),
                             pos_hint={'center_x': 1, 'center_y': .5},
                             background_normal='ButtonPlusP.png',
                             background_down='ButtonDeactivated.png',
                             background_disabled_normal='ButtonDeactivated.png', font_size=20)
        lblfolds = Label(text='[b]FOLDS:',markup=True, font_size=(11), pos_hint={'center_x': .5, 'center_y': .5})
        numfolds = Label(text=str(n_folds))



        btnplussidetext.disabled = True
        btnlesssidetext.disabled = True

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
        layoutsettingsvertical.add_widget(layoutsettingsup)
        layoutsettingsvertical.add_widget(layoutsettingsdown)
        layoutbtninfo.add_widget(btnlesssidetext)
        layoutbtninfo.add_widget(btnplussidetext)
        layoutinfodown.add_widget(Label(
            text='-----------------------------------------------------------------------------------------------------------------------------------------------',
            pos_hint={'center_x': .5, 'center_y': .5}, size_hint=(1, .1), color=(.6, .87, .89, 1)))
        layoutinfodown.add_widget(lblcurrentpage)
        layoutfilenameright.add_widget(lblfilename)


        neurontwolayout.add_widget(lblneurontwo)
        neurontwolayout.add_widget(lblhiddenlayertwo)
        neuronthreelayout.add_widget(lblneuronthree)
        neuronthreelayout.add_widget(lblhiddenlayerthree)

        def settingslayout(settings):
            global prediction, biaseslabels, varerrorlayout
            prediction = False
            varerrorlayout = False
            btnlayout.height = 300
            layoutmainscreen.height = 350
            layoutinfoleft.size_hint_x = 0.2
            layoutinforight.size_hint_x = 0.2
            layouttrainscreen = parentlayout.ids.idLayoutNeuron.ids.idAnchorLayoutNeuron
            for element in biaseslabels:
                layoutneuronmainscreen.remove_widget(element)
            biaseslabels = list()

            layout.remove_widget(btnTrain)
            layout.remove_widget(btnError)
            layout.remove_widget(btnPredict)
            layoutinfotop.remove_widget(btnSettings)
            layoutinfotop.remove_widget(btnBack)
            layout.remove_widget(btnPredict)
            layout.remove_widget(btnProcessPredict)
            layout.remove_widget(btnCancel)
            layout.remove_widget(btnnewinput)
            self.clear(layoutneuronmainscreen, layoutlabels)
            layout.add_widget(layoutsettingsvertical)

        settingslayout(lblblackthree)

        def mainlayout(self):
            global prediction, n_inputs, n_outputs, dataset, numdata, listinputs, filename
            layoutmainscreen.height = 580
            btnlayout.height = 50
            layoutinfoleft.size_hint_x = 0.1
            layoutinforight.size_hint_x = 0.1

            #filename = 'seeds_dataset.csv'
            isimage, dataset = readdataset(filename)
            print(filename)
            print (dataset)
            for i in range(len(dataset[0]) - 1):
                str_column_to_float(dataset, i)
            # convert class column to integers
            str_column_to_int(dataset, len(dataset[0]) - 1)
            # normalize input variables
            minmax = dataset_minmax(dataset)
            normalize_dataset(dataset, minmax)
            n_inputs = len(dataset[0]) - 1
            numdata = n_inputs
            n_outputs = len(set([row[-1] for row in dataset]))

            layouttrainscreen = parentlayout.ids.idLayoutNeuron.ids.idAnchorLayoutNeuron
            layouttrainscreen.anchor_x = 'center'

            layout.remove_widget(layoutsettingsvertical)
            layout.remove_widget(btnCancel)
            layout.remove_widget(btnnewinput)
            layoutinfotop.remove_widget(btnSettings)
            layoutinfotop.remove_widget(btnBack)
            btnSettings.disabled = False
            btnBack.disabled = False
            layout.remove_widget(btnProcessPredict)
            layoutprogressbar = parentlayout.ids.mainseparatorprogressbar
            layoutprogressbar.remove_widget(progressbar)


            layout.add_widget(btnTrain)
            layout.add_widget(btnError)
            layout.add_widget(btnPredict)
            layoutinfotop.add_widget(btnSettings)
            if prediction == False:
                if varerrorlayout == False:
                    myevent.drawingmainlayout()
                    btnPredict.disabled = True
                else:
                    btnPredict.disabled = True
            else:
                btnPredict.disabled = False

        def predictlayout(self):
            layouttrainscreen = parentlayout.ids.idLayoutNeuron.ids.idAnchorLayoutNeuron
            layouttrainscreen.anchor_x = 'center'
            layout.remove_widget(btnTrain)
            layout.remove_widget(btnError)
            layout.remove_widget(btnPredict)
            layout.remove_widget(btnCancel)
            layoutinfotop.remove_widget(btnSettings)
            layoutinfotop.remove_widget(btnBack)
            layout.remove_widget(btnnewinput)
            btnSettings.disabled = False
            btnBack.disabled = False

            layout.add_widget(btnProcessPredict)
            layoutinfotop.add_widget(btnBack)
            layoutinfotop.add_widget(btnSettings)

            MyApp.drawinginputs(self,layoutneuronmainscreen, layoutlabels)

        def predictdonelayout(self):
            global listinputs
            layouttrainscreen = parentlayout.ids.idLayoutNeuron.ids.idAnchorLayoutNeuron
            for element in listinputs:
                inputtext = element.text
                if inputtext == '':
                    pass
                else:
                    input.append(float(inputtext))

            if len(input) < numdata:
                container = BoxLayout(orientation = 'vertical')
                container.add_widget(Label(text='Not all inputs where submited'))
                popupone = Popup(title='ERROR', content=container, size_hint=(.3,.3), separator_color = (.67,.57,.69,1), auto_dismiss = False)
                popupbtn = Button(text='OK', on_press= popupone.dismiss, background_normal='ButtonPlusP.png', background_down='ButtonDeactivated.png', background_disabled_normal='ButtonDeactivated.png')
                container.add_widget(popupbtn)
                popupone.open()
            else:
                layouttrainscreen.anchor_x = 'center'
                layout.remove_widget(btnTrain)
                layout.remove_widget(btnError)
                layout.remove_widget(btnPredict)
                layout.remove_widget(btnCancel)
                layoutinfotop.remove_widget(btnSettings)
                layoutinfotop.remove_widget(btnBack)
                layout.remove_widget(btnProcessPredict)
                btnSettings.disabled = False
                btnBack.disabled = False
                layoutinfotop.add_widget(btnBack)
                layoutinfotop.add_widget(btnSettings)
                layout.add_widget(btnnewinput)
                for row in input:
                    prediction = predict(network, row)
                    #accuracy_metric(dataset,prediction)
                MyApp.drawingvisualize(self, layoutneuronmainscreen)

        def trainlayout(self):
            layouttrainscreen = parentlayout.ids.idLayoutNeuron.ids.idAnchorLayoutNeuron
            layoutinfoleft.size_hint_x = 0.4
            layoutinforight.size_hint_x = 0.0
            layout.remove_widget(btnTrain)
            layout.remove_widget(btnError)
            layout.remove_widget(btnPredict)
            layout.remove_widget(btnCancel)
            layoutinfotop.remove_widget(btnSettings)
            layoutinfotop.remove_widget(btnBack)
            layoutprogressbar = parentlayout.ids.mainseparatorprogressbar
            layoutprogressbar.remove_widget(progressbar)
            progressbar.value=0
            btnSettings.disabled = False
            btnBack.disabled = False

            layoutinfotop.add_widget(btnBack)
            layoutinfotop.add_widget(btnSettings)
            mydata = MyApp.Lblleft(type='train', dataerror=0)
            MyApp.drawinglblblackkv(self, layoutinfomiddle, mydata)


        def cancellayout(self):
            layout.remove_widget(btnTrain)
            layout.remove_widget(btnError)
            layout.remove_widget(btnPredict)
            layout.remove_widget(btnnewinput)
            layoutinfotop.remove_widget(btnBack)
            layoutinfotop.remove_widget(btnSettings)
            btnSettings.disabled = True
            btnBack.disabled = True
            layout.add_widget(btnCancel)
            layoutinfotop.add_widget(btnBack)
            layoutinfotop.add_widget(btnSettings)

        def errorlayout(accuracy):
            layouttrainscreen = parentlayout.ids.idLayoutNeuron.ids.idAnchorLayoutNeuron
            layoutinfoleft.size_hint_x = 0.4
            layoutinforight.size_hint_x = 0.0
            layout.remove_widget(btnTrain)
            layout.remove_widget(btnError)
            layout.remove_widget(btnPredict)
            layout.remove_widget(btnCancel)
            layoutinfotop.remove_widget(btnSettings)
            layoutinfotop.remove_widget(btnBack)
            layoutprogressbar = parentlayout.ids.mainseparatorprogressbar
            layoutprogressbar.remove_widget(progressbar)
            progressbar.value = 0
            btnSettings.disabled = False
            btnBack.disabled = False

            layoutinfotop.add_widget(btnBack)
            layoutinfotop.add_widget(btnSettings)
            mydata = MyApp.Lblleft(type='Find error', dataerror=accuracy)
            MyApp.leftlabels(self, layoutinfomiddle, mydata)

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

        def btnTrainpressed(btnTrain):
            global progressbar, accuracylabel, scores, n_epoch, l_rate, numdata, valuetrain
            global network, n_inputs, n_outputs, n_folds, dataset, list_hidden, trainclock
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

            progressbar.value = 0
            valuetrain = 0
            trainclock = Clock.schedule_interval(layoutpbtrainnext, 1 / 10)
            network = initialize_network(n_inputs, list_hidden, n_outputs)
            threading.Thread(target=train_network, args=(network, dataset, l_rate, n_epoch, n_outputs, 'train')).start()

            #train_network(network, dataset, l_rate, n_epoch, n_outputs)
            # for row in dataset:
            #      prediction = predict(network, row)
            #      accuracy_metric(dataset,prediction)
            cancellayout(self)

        def btnErrorpressed(btnTrain):
            global progressbar, accuracylabel, scores, n_epoch, l_rate, numdata, valuepb
            global network, n_inputs, n_outputs, n_folds, dataset, list_hidden, trainclock
            layoutprogressbar = parentlayout.ids.mainseparatorprogressbar
            layoutprogressbar.add_widget(progressbar)
            progressbar.value = 0
            valuepb = 0

            trainclock = Clock.schedule_interval(layoutpbnext, 1 / 10)

            threading.Thread(target=evaluate_algorithm, args=(dataset, back_propagation, n_folds, l_rate, n_epoch, list_hidden)).start()
            cancellayout(self)

        def filechooserfunction(self):
            global popup
            content = Filechooser()
            popup = Popup(title="Load file", content=content,
                                size_hint=(0.9, 0.9),auto_dismiss= False, separator_color= (.67,.57,.69))
            popup.open()

        def btncancelpressed(self):
            progressbar.value=0


        def evaluate_algorithm(dataset, algorithm, n_folds, *args):
            global progressbar, myevent
            global scores
            folds = cross_validation_split(dataset, n_folds)
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
            myevent.evevaluatealgorithm('hola')


        btnTrain.bind(on_press=btnTrainpressed)
        btnPredict.bind(on_press=predictlayout)
        btnProcessPredict.bind(on_press=predictdonelayout)
        btnnewinput.bind(on_press=predictlayout)
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
        btnError.bind(on_press=btnErrorpressed)
        btnplusfolds.bind(on_press=btnplusfoldspressed)
        btnlessfolds.bind(on_press=btnlessfoldspressed)
        btnplusepoch.bind(on_press=btnplusepochpressed)
        btnlessepoch.bind(on_press=btnlessepochpressed)
        btnplusrate.bind(on_press=btnplusratepressed)
        btnlessrate.bind(on_press=btnlessratepressed)
        btnOpen.bind(on_press=filechooserfunction)
        btnplussidetext.bind(on_press=lambda x:MyApp().leflabelswitch(layoutinfomiddle, 0,'btnplus'))
        btnlesssidetext.bind(on_press=lambda x:MyApp().leflabelswitch(layoutinfomiddle, 0, 'btnless'))


        return parentlayout



class LayoutMenu(BoxLayout):
    pass

class LayoutNeuronMainScreen(BoxLayout):
    pass
class LayoutInfo(BoxLayout):
    pass

class LayoutNeuron(AnchorLayout):
    pass

class LayoutScreen(BoxLayout):
    pass

class Filechooser(BoxLayout):
    def myselected(self, path, filenameselected):
        self.ids.text_input.text = basename(filenameselected)


    def open(self, path, filenameselected):
        global popup, filename, pagenum, mytexturedataset
        class WidgetTexture(Widget):
            def __init__(self, imagebits, **kwargs):
                super(WidgetTexture, self).__init__(**kwargs)

                sidesquare = sqrt(len(imagebits)-1)
                intsidesquare = int(sidesquare)
                self.texture = Texture.create(size=(sidesquare, sidesquare))


                buf = list(imagebits)
                buf.pop(-1)
                arr = list()
                for value in buf:
                    arr.append(int(value))

                bufbytes = array('B', arr)
                self.texture.blit_buffer(bufbytes, colorfmt='luminance', bufferfmt='ubyte')
                self.texture.flip_vertical()


                # that's all ! you can use it in your graphics now :)
                # if self is a widget, you can do this
                with self.canvas:
                    self.rect=Rectangle(texture=self.texture, pos=self.pos, size=(sidesquare, sidesquare))

                self.bind(pos=self.update_rect,
                          size=self.update_rect)

            def update_rect(self, *args):
                self.rect.pos = (self.center_x - self.rect.size[0] / 2, self.center_y - self.rect.size[1] / 2)


        def btnactivation(mytexturedataset):
            global pags
            if pagenum <= 1:
                btnpreviouspage.disabled = True
            else:
                btnpreviouspage.disabled = False
            if pagenum >= int(pags):
                btnnextpage.disabled = True
            else:
                btnnextpage.disabled = False

        def Numberdataset(checkbox, value):
            global pagenum, mytexturedataset, pags
            mainlayout.clear_widgets()
            if checkbox == 'plusbtn':
                pagenum += 1
                btnactivation(mytexturedataset)
            else:
                if checkbox == 'lessbtn':
                    pagenum -= 1
                    btnactivation(mytexturedataset)
                else:
                    pass
            #Show number
            if numberdatabox.active == True:
                pags = len(mytexturedataset) / 14
                check = is_whole(pags)
                lblelementsize.text = '[b]element size: [/b]' + str(len(mytexturedataset[0]))
                if check == True:
                    pags = "{:.0f}".format(pags)
                else:
                    pags = int("{:.0f}".format(pags))
                    pags += 1
                lblpage.text = '[b]Page: [/b]' + str(pagenum) + '/' + str(pags)

                if len(mytexturedataset[0]) > 20:
                    mainlayout.cols = 11
                    linesperpage = 14
                    for rowtexture in range(linesperpage):
                        index = rowtexture + (pagenum - 1) * linesperpage
                        pos = 0
                        if index >= len(mytexturedataset):
                            break
                        firstline = mytexturedataset[rowtexture + (pagenum-1) * linesperpage]
                        for number in firstline:
                            if pos < 5 or pos >= (len(firstline) - 5):

                                datalbl = Label(text=str(number))
                                mainlayout.add_widget(datalbl)
                            elif pos == 5 :
                                datalbl = Label(text='...')
                                mainlayout.add_widget(datalbl)
                            pos += 1
                else:
                    mainlayout.cols = len(mytexturedataset[0])
                    for rowtexture in range(14):
                        for number in mytexturedataset[rowtexture+(pagenum-1)*14]:
                            mystrrow = number + ' '
                            datalbl = Label(text =mystrrow)
                            mainlayout.add_widget(datalbl)


            #Show Images
            else:
                points = sqrt(len(mytexturedataset[0]) - 1)
                cols = int((Window.width - 200) / points)
                mainlayout.cols = int(cols)
                rows = int((Window.height - 200) / points)
                mainlayout.rows = int (rows)
                nitemsperpage = (mainlayout.cols * mainlayout.rows)
                pagsfloat = (len(mytexturedataset) / nitemsperpage)
                pagsaux = float(int(pagsfloat))
                pags = int(pagsfloat)
                lblelementsize.text = '[b]image size: [/b]' + str(int(points)) + ' x ' + str(int(points))
                if pagsaux != pagsfloat:
                    pags += 1

                lblpage.text = '[b]Page: [/b]' + str(pagenum) + '/' + str(pags)
                for rowtexture in range(nitemsperpage): #in range cols * rows
                    index = rowtexture+(pagenum-1)*nitemsperpage
                    if index >= len(mytexturedataset):
                        break
                    imagedata = WidgetTexture(mytexturedataset[index])
                    mainlayout.add_widget(imagedata)
            btnactivation(mytexturedataset)

        def Outputdataset(checkbox, value):
            global mytexturedataset, outputchanged
            if outputfirstbox.active == True and outputchanged == False:
                isimage, mytexturedataset = showdataset(filenameselected)
                Numberdataset(0, 0)
                outputchanged = True
            elif outputlastbox.active == True and outputchanged == True:
                mytexturedataset = datasetoutputlastcolumn(mytexturedataset)
                Numberdataset(0, 0)
                outputchanged = False
            else:
                pass

        def Outputoptions(self):
            containeroutput = BoxLayout(orientation='vertical')
            outputbox = BoxLayout(orientation='horizontal')
            outputlblbox = BoxLayout(orientation='vertical')
            popupoutputs = Popup(title='OUTPUTS:', content=containeroutput, separator_color=(.6,1,1, 1),
                          auto_dismiss=False, size_hint=(.6,.8))
            popupoutputbtn = Button(text='OK', on_press=popupoutputs.dismiss, background_normal='ButtonPlusP.png',
                                    background_down='ButtonDeactivated.png',
                                    background_disabled_normal='ButtonDeactivated.png',
                                    pos_hint={'center_x': .5, 'center_y': .5}, size_hint=(.4, 1))
            outputbox.add_widget(outputlblbox)

            outputslast = ([row[-1] for row in mytexturedataset])
            setoutputslast = sorted(set(outputslast))
            for number in setoutputslast:
                lbloutput = Label(text=str(number))
                outputlblbox.add_widget(lbloutput)
            if imagedatabox.active == True:
                outputimagebox = BoxLayout(orientation='vertical')
                for number in setoutputslast:
                    for row in mytexturedataset:
                        if row[-1] == number:
                            imagedata = WidgetTexture(row)
                            outputimagebox.add_widget(imagedata)
                            break

                outputbox.add_widget(outputimagebox)
            containeroutput.add_widget(outputbox)
            containeroutput.add_widget(popupoutputbtn)

            popupoutputs.open()
        global pags

        filename = os.path.join(path, filenameselected)
        lblfilename.text = ('[b] Current file: [/b]' + '[i]' + filenameselected)
        isimage, mytexturedataset = showdataset(filename)

        if mytexturedataset == None:
            containererror = BoxLayout(orientation='vertical')
            popuperror = Popup(title='DATASET:', content=containererror, separator_color=(.6,1,1, 1),
                              auto_dismiss=False, size_hint=(.5,.5))
            lblerror = Label(text='{} : Wrong file format' .format(filenameselected))
            popuperrorbtn = Button(text='OK', on_press=popuperror.dismiss, background_normal='ButtonPlusP.png',
                              background_down='ButtonDeactivated.png', background_disabled_normal='ButtonDeactivated.png',
                              pos_hint={'center_x': .5, 'center_y': .5}, size_hint=(1,.5))
            containererror.add_widget(lblerror)
            containererror.add_widget(popuperrorbtn)
            popuperror.open()
        else:
            popup.dismiss()
            pagenum = 1
            containerbox = BoxLayout(orientation='vertical', spacing = 10)
            popupshow = Popup(title='DATASET:', content=containerbox, separator_color=(.6,1,1, 1),
                              auto_dismiss=False)
            popupbtn = Button(text='OK', on_press=popupshow.dismiss, background_normal='ButtonPlusP.png',
                              background_down='ButtonDeactivated.png', background_disabled_normal='ButtonDeactivated.png',
                              pos_hint={'center_x': .5, 'center_y': .5}, size_hint=(.4,1))
            mainlayout = GridLayout(cols=1)
            middlelayout = BoxLayout(orientation = 'horizontal')
            btnnextpage = Button(text='[b]>', markup=True, size_hint=(.05, .2),
                             pos_hint={'center_x': 1, 'center_y': .5},
                             background_normal='ButtonPlusP.png',
                             background_down='ButtonDeactivated.png',
                             background_disabled_normal='ButtonDeactivated.png', font_size=20)
            btnpreviouspage = Button(text='[b]<', markup=True, size_hint=(.05, .2),
                                 pos_hint={'center_x': 1, 'center_y': .5},
                                 background_normal='ButtonPlusP.png',
                                 background_down='ButtonDeactivated.png',
                                 background_disabled_normal='ButtonDeactivated.png', font_size=20)
            btnlayout = BoxLayout(orientation='horizontal', size_hint=(1,.1))
            typelayout = GridLayout( cols = 4)
            infolayout = BoxLayout(orientation='horizontal', size_hint=(1,.1))
            imagedatabox = CheckBox(group = 'typeobject', size_hint=(.4,1))
            lblimagedatabox = Label(text='Images dataset',halign="left")
            lblimagedatabox.bind(size=lblimagedatabox.setter('text_size'))
            numberdatabox = CheckBox(group = 'typeobject', size_hint=(.4,1))
            lblnumberdatabox = Label(text='Numbers dataset',halign="left")
            lblnumberdatabox.bind(size=lblnumberdatabox.setter('text_size'))
            outputfirstbox = CheckBox(group='output', size_hint=(.4,1))
            lbloutputfirstbox = Label(text='Outputs on first column',halign="left")
            lbloutputfirstbox.bind(size=lbloutputfirstbox.setter('text_size'))
            outputlastbox = CheckBox(group='output', size_hint=(.4,1))
            lbloutputlastbox = Label(text='Outputs on last column',halign="left")
            lbloutputlastbox.bind(size=lbloutputlastbox.setter('text_size'))

            def is_whole(n):
                return n % 1 == 0

            lblpage = Label(text='', markup = True, pos_hint={'center_x': .5, 'center_y': .5},
                            size_hint=(1, .1), color=(.7,.9,.9))
            outputslast = ([row[-1] for row in mytexturedataset])
            setoutputslast = set(outputslast)

            btnnoutputs = Button(text='[b]nº outputs: [/b]' + str(len(setoutputslast)),markup = True, pos_hint={'center_x': .5, 'center_y': .5},
                            size_hint=(1, .9), background_normal='ButtonPlusP.png',
                             background_down='ButtonDeactivated.png')
            lblnelements = Label(text='[b]nº elements: [/b]' + str(len(outputslast)), markup = True, pos_hint={'center_x': .5, 'center_y': .5},
                            size_hint=(.8, .1))

            lblelementsize = Label(text='', markup = True, pos_hint={'center_x': .5, 'center_y': .5},
                            size_hint=(.8, .1))




            outputlastbox.active = True
            outputchanged = False
            btnpreviouspage.disabled = True
            btnnextpage.disabled = False
            btnnextpage.bind(on_press=lambda x:Numberdataset('plusbtn',self))
            btnpreviouspage.bind(on_press=lambda x: Numberdataset('lessbtn', self))
            outputfirstbox.bind(on_press=lambda x: Outputdataset(0, self))
            numberdatabox.bind(active=Numberdataset)
            btnnoutputs.bind(on_press=Outputoptions)

            typelayout.add_widget(numberdatabox)
            typelayout.add_widget(lblnumberdatabox)
            typelayout.add_widget(outputlastbox)
            typelayout.add_widget(lbloutputlastbox)
            typelayout.add_widget(imagedatabox)
            typelayout.add_widget(lblimagedatabox)
            typelayout.add_widget(outputfirstbox)
            typelayout.add_widget(lbloutputfirstbox)
            btnlayout.add_widget(popupbtn)
            btnlayout.add_widget(typelayout)
            middlelayout.add_widget(btnpreviouspage)
            middlelayout.add_widget(mainlayout)
            middlelayout.add_widget(btnnextpage)
            containerbox.add_widget(middlelayout)
            containerbox.add_widget(infolayout)
            infolayout.add_widget(btnnoutputs)
            infolayout.add_widget(lblnelements)
            infolayout.add_widget(lblelementsize)
            infolayout.add_widget(lblpage)
            containerbox.add_widget(btnlayout)
            popupshow.open()

            if isimage == True:
                imagedatabox.active = True
            else:
                numberdatabox.active = True
            Numberdataset(numberdatabox,numberdatabox)

    def BtnImagesFncn(self,btnimages):
        if btnimages.text == 'Numbers dataset':
            btnimages.text = 'Images dataset'
        else:
            btnimages.text = 'Numbers dataset'
    def BtnOutputFncn(self,btnimages):
        if btnimages.text == 'Output on last column':
            btnimages.text = 'Output on first column'
        else:
            btnimages.text = 'Output on last column'


    def PrepareFile(self, filenameselected,btnimages, btnoutput, popupone, path):







        if btnimages.text == 'Numbers dataset':
            containershow.cols = len(filenameselected[0])
            for row in filenameselected:
                for column in range(len(filenameselected[0])):
                    #print(open(os.path.join(path, filenameselected[0][0])).read())
                    containershow.add_widget(Label(text=file[0]))
            containerbox.add_widget(containershow)
            containerbox.add_widget(popupbtn)
            popupshow.open()
        elif btnimages.text == 'Images dataset':
            isimage, mytexturedataset = showdataset(filenameselected)
            mytexturedataset = datasetoutputfirstcolumn(mytexturedataset)
            containershow = GridLayout(cols=10)
            for rowtexture in range(15):
                imagedata = WidgetTexture(mytexturedataset[rowtexture])
                containershow.add_widget(imagedata)
            containerbox.add_widget(containershow)
            containerbox.add_widget(popupbtn)
            popupshow.open()
            # with imagedata.canvas.after:
            #     Rectangle(size=(imagedata.size), pos=(imagedata.pos))




        #with popupshow.canvas:
            #Rectangle(size=(popupshow.width,popupshow.height))



    def popupdismiss(self):
        global popup
        popup.dismiss()






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



