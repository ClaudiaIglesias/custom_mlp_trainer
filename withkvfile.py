


from kivy.app import App
from kivy.storage.jsonstore import JsonStore
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
from kivy.properties import ObjectProperty, NumericProperty
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
from kivy.animation import Animation


from loaddataset import showdataset, readdataset, str_column_to_float

numoutputs = 1
n_outputs = 1
n_inputs = 2
dataset = 1
list_hidden = list()
listinputs = list()
input = list()
outputimageslist = list()
network = []
n_folds = 5
l_rate = 0.3
n_epoch = 20
prediction = False
varerrorlayout = False


timestart = None
timeleft = ''
lbltimeleft = Label(text='time left:', markup = True)

errorlbl = Label(text = '', markup = True)
labelcycles = (Label(text='[b] [color=D464EB]Cycles: [/b][/color]', markup=True,
                             halign="left",
                             size_hint=(1, .1), color=(1, 1, 1)))
labelfolds = (Label(text='[b] [color=D464EB]Folds: [/b][/color]' , markup=True,
                            halign="left",
                            size_hint=(1, .1), color=(1, 1, 1)))
sum_error = 0

valuepb = 0
currentcycle = 0
currentfold = 0
valuetrain = 0
stoptrainning = False
valuemydata = None
accuracylabel =''
scores = list()
biaseslabels = list()
trainclock = None
leftpanellist = list()
listoflines = list()
filename = ''
ti = TextInput(size=(40, 30), size_hint=(None, None))
ti2 = TextInput(size=(40, 30), size_hint=(None, None))
ti3 = TextInput(size=(40, 30), size_hint=(None, None))
progressbar = ProgressBar()
popupfilechooser = None
currentpage = 0
myevent=None



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
def initialize_network(n_inputs, n_hidden, n_outputs, transfer_f, transfer_derfivative_f):
    network = list()
    weightspernode=n_inputs
    for nodesinhidden in n_hidden:
        layer = list()
        for i in range(nodesinhidden):
            layer.append({'weights': [random() * sqrt(1 / weightspernode) for j in range(weightspernode + 1)],
                 'transfer': transfer_f, 'transfer_derivative': transfer_derfivative_f})
        network.append(layer)
        weightspernode=nodesinhidden

    output_layer = list()
    for i in range(n_outputs):
        output_layer.append({'weights':[ random() * sqrt(1 / nodesinhidden) for j in range(nodesinhidden + 1) ],
            'transfer': transfer_sigmoid, 'transfer_derivative': transfer_sigmoid_derivative})
    network.append(output_layer)
    return network

def initialize_network_no_weights(n_inputs, n_hidden, n_outputs):
    network = list()
    weightspernode=n_inputs
    defaultweight = 1.0
    for nodesinhidden in n_hidden:
        network.append ( [{'weights': [defaultweight for i in range(weightspernode + 1)]} for i in range(nodesinhidden)] )
        weightspernode=nodesinhidden

    output_layer = [{'weights': [defaultweight for i in range(nodesinhidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network
# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation

# Transfer neuron activation
def transfer_rectified(output):
   return output * (output > 0)

def transfer_sigmoid(activation):
    return 1.0 / (1.0 + exp(-activation))


# Calculate the derivative of an neuron output
def transfer_rectified_derivative(output):
    return 1. * (output > 0)


def transfer_sigmoid_derivative(output):
    return output * (1.0 - output)

# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = neuron['transfer'](activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

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
            neuron['delta'] = errors[j] * neuron['transfer_derivative'](neuron['output'])

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
def train_network(network, train, l_rate, n_epoch, n_outputs, type, eventp, myapp):
    global valuepb, valuetrain, valuemydata, timestart, timeleft, currentcycle, currentfold
    global n_folds, sum_error
    step = 100.0/(n_folds*n_epoch*len(train))
    steptrain = 100.0/ n_epoch
    currentfold += 1
    currentcycle = 0
    for epoch in range(n_epoch):
        sum_error = 0
        valuetrain += steptrain
        valuemydata = MyApp.Lblleft(type=type, dataerror=0)
        currentcycle += 1
        if stoptrainning:
            break
        for row in train:
            valuepb += step
            outputs = forward_propagate(network, row)
            expected = [0 for i in range(n_outputs)]
            expected[row[-1]] = 1
            sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
    if type == 'train':
        eventp.trainnetwork(myapp)


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


def auxlayoutpb(valuetrain):
    global valuemydata, printth
    if valuetrain > printth:
        for layer in valuemydata.networklist:
            print(layer)
        printth += 20
    # calculate new maxs
    if valuemydata is None:
        return
    for layer in range(len(valuemydata.networklist)):
        max = 0.0
        for myline in listoflines:
            if myline.layer == layer:
                myline.value = abs(valuemydata.networklist[myline.layer][myline.hidden]['weights'][myline.offset])
                if myline.value > max:
                    max = myline.value
        for myline in listoflines:
            if myline.layer == layer:
                myline.maximum = max

    nchanges = 0
    for myline in listoflines:
        nchanges += myline.change()
    print ('number of changes {} \n' .format(nchanges))
#Error progress
def layoutpbnext(self):
    global valuepb, lbltimeleft, timestart, timeleft, sum_error, currentcycle, currentfold
    global progressbar

    timenow = time.monotonic()
    elapse = timenow - timestart
    if valuepb == 0:
        lbltimeleft.text = 'estimating...'
    else:
        left = (elapse / valuepb) * (100.0 - valuepb)
        if left > 3600:
            hours = divmod(left, 3600)
            minutes = hours[1]/60
            lbltimeleft.text = '[b] Remaining time: [/b]' + str("{:.0f}".format(hours[0])) + ' hours ' + str("{:.0f}".format(minutes) + ' minutes')
        elif left > 60:
            minutes = left / 60
            lbltimeleft.text = '[b] Remaining time: [/b]' + str("{:.0f}".format(minutes)) + ' minutes'
        else:
            lbltimeleft.text = '[b] Remaining time: [/b]' + str("{:.0f}".format(left)) + ' seconds'
    progressbar.value = valuepb
    auxlayoutpb(valuepb)
    errorlbl.text = '[b] [color=D464EB]Error: [/b][/color]' + str("{:.2f}".format(sum_error))
    labelcycles.text = '[b] [color=D464EB]Cycles: [/b][/color]' + str(currentcycle) + '/' + str(n_epoch)
    labelfolds.text =  '[b] [color=D464EB]Folds: [/b][/color]' + str(currentfold) + '/' + str(n_folds)
#Train progress
def layoutpbtrainnext(self):
    global valuetrain, lbltimeleft, timestart, timeleft, labelcycles, labelfolds
    global progressbar, currentcycle, currentfold
    timenow = time.monotonic()
    elapse = timenow - timestart
    if valuetrain == 0:
        lbltimeleft.text = 'estimating...'
    else:
        left = (elapse / valuetrain) * (100.0 - valuetrain)
        if left > 3600:
            hours = divmod(left, 3600)
            minutes = hours[1] / 60
            lbltimeleft.text = '[b] Remaining time: [/b]' + str("{:.0f}".format(hours[0])) + ' hours ' + str("{:.0f}".format(minutes) + ' minutes')
        elif left > 60:
            minutes = left / 60
            lbltimeleft.text = '[b] Remaining time: [/b]' + str("{:.0f}".format(minutes)) + ' minutes'
        else:
            lbltimeleft.text = '[b] Remaining time: [/b]' + str("{:.0f}".format(left)) + ' seconds'
    progressbar.value = valuetrain
    auxlayoutpb(valuetrain)
    labelcycles.text = '[b] [color=D464EB]Cycles: [/b][/color]' + str(currentcycle) + '/' + str(n_epoch)
    labelfolds.text = '[b] [color=D464EB]Folds: [/b][/color]' + str(n_folds)


    #MyApp.drawinglblblackkv(self, self.parentlayout.ids.idlayoutmiddle, valuemydata)



# Backpropagation Algorithm With Stochastic Gradient Descent
def back_propagation(train, test, eventp, myapp, l_rate, n_epoch, n_hidden):
    global network

    n_inputs = len(train[0]) - 1
    n_outputs = len(set([row[-1] for row in train]))
    network = initialize_network(n_inputs, n_hidden, n_outputs, transfer_sigmoid, transfer_sigmoid_derivative)
    train_network(network, train, l_rate, n_epoch, n_outputs, 'backp', eventp, myapp)
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
            myinputlabel = Label(text='[b]' + element.text, markup=True, size=(30, 30), font_size=17, size_hint=(None, None),
                              pos=(middlex - 270, 50 * l + (middley+2 - space / 2)), color=(1, 1, 1, 1))
            draw.add_widget(myinputlabel)
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
    def drawinglblblackkv(self, textlayout, mydata):
        global accuracylabel, progressbar, prediction, biaseslabels
        global n_epoch, n_folds, l_rate
        global listoflines
        global valuemydata
        valuemydata = MyApp.Lblleft(mydata.type, mydata.dataerror)

        listoflines.clear()
        prediction = True

        draw = self.parentlayout.ids.idLayoutNeuron.ids.idLayoutNeuronMainScreen.ids.idrelativelayoutneuronmainscreen
        titlelayout = self.parentlayout.ids.idLayoutNeuron.ids.idLayoutNeuronMainScreen.ids.idLabelsLayoutNeuronMainScreen

        draw.canvas.clear()


        MyApp.leftlabels(self, textlayout, mydata)


        numoutputs = mydata.n_outputlbl

        numdata = mydata.n_inputlbl

        if numdata < 11:
            space = numdata * 35 + (numdata - 1) * 15
        else:
            space = 10 * 35 + (10 - 1) * 15

        if mydata.list_hidden[0] < 21:
            spaceone =mydata.list_hidden[0] * 18 + (mydata.list_hidden[0] - 1) * 7
        else:
            spaceone = 20 * 18 + (20 - 1) * 7
        spaceoutput = numoutputs * 35 + (numoutputs - 1) * 15
        middlex = draw.width/2
        middley = draw.height/2 - 20
        middletitlex = titlelayout.width / 2
        middletitley = titlelayout.height / 2

        self.outputlabel.pos = (middletitlex + 202, middletitley - 60)
        self.inputlabel.pos = (middletitlex - 268, middletitley - 60)

        titlelayout.remove_widget(self.hiddenlayeronelabelone)
        titlelayout.remove_widget(self.hiddenlayeronelabeltwo)
        titlelayout.remove_widget(self.hiddenlayeronelabelthree)
        titlelayout.remove_widget(self.outputlabel)
        titlelayout.remove_widget(self.inputlabel)
        titlelayout.add_widget(self.outputlabel)
        titlelayout.add_widget(self.inputlabel)



        if len(mydata.list_hidden) == 1:
            # hiddenlayerone
            valuefloat = 0
            linecoloroutput = 0
            linecolorinput = 0
            self.hiddenlayeronelabelone.pos = (middletitlex - 28, middletitley - 50)
            titlelayout.add_widget(self.hiddenlayeronelabelone)
            for a in range(mydata.list_hidden[0]):
                if valuefloat < abs(mydata.networklist[0][a]['weights'][-1]):
                    valuefloat = abs(mydata.networklist[0][a]['weights'][-1])

                for i in range(numdata):
                    if linecolorinput < abs(mydata.networklist[0][a]['weights'][i]):
                        linecolorinput = abs(mydata.networklist[0][a]['weights'][i])

            for e in range(numoutputs):
                for a in range(mydata.list_hidden[0]):
                    if linecoloroutput < abs(mydata.networklist[1][e]['weights'][a]):
                        linecoloroutput = abs(mydata.networklist[1][e]['weights'][a])

            if mydata.list_hidden[0] < 21:
                for a in range(mydata.list_hidden[0]):
                    with draw.canvas:
                        value = mydata.networklist[0][a]['weights'][-1]
                        colorlayerone = abs(value) / valuefloat
                        Color(1,1,1, 0.25 + colorlayerone/1.33)
                        RoundedRectangle(size=(18, 18), pos=(middlex, a * 25 + (middley - spaceone / 2)))
                        Color(0,0,0,1)
                        RoundedRectangle(size=(13, 13), pos=(middlex+2.5, a * 25 + (middley + 2.5 - spaceone / 2)), radius=[(6,6),(6,6),(6,6),(6,6)])
                    biases = "{:.2f}".format(value)
                    #nodeslabel = Label(text=biases, size_hint=(None,None), font_size=9, pos=(middlex-31.5, a * 25 + (middley-31.5 - spaceone / 2)),
                                            #color=(1,1,1, 0.25 + colorlayerone/1.33))
                    #draw.add_widget(nodeslabel)
                    #biaseslabels.append(nodeslabel)
                    i = 0
                    e = 0
                    if numdata < 11:
                        for d in range(numdata):
                            mypoints = [middlex - 205, d * 50 + (middley - space / 2) + 17.5, middlex,
                                      a * 25 + (middley + 9 - spaceone / 2)]
                            MyApp.listoflinesappend(self, 0, a, d, linecolorinput, draw.canvas, mydata.networklist, mypoints)


                    else:
                        for d in range(10):
                            mypoints = [middlex - 205, d * 50 + (middley - space / 2) + 17.5, middlex,
                                             a * 25 + (middley + 9 - spaceone / 2)]
                            MyApp.listoflinesappend(self, 0, a, d, linecolorinput, draw.canvas, mydata.networklist, mypoints)

            else:
                for a in range(20):
                    if a < 8 or a > 11:
                        with draw.canvas:
                            value = mydata.networklist[0][a]['weights'][-1]
                            colorlayerone = abs(value) / valuefloat
                            Color(1,1,1, 0.25 + colorlayerone/1.33)
                            RoundedRectangle(size=(18, 18), pos=(middlex, a * 25 + (middley - spaceone / 2)))
                            Color(0,0,0,1)
                            RoundedRectangle(size=(13, 13), pos=(middlex+2.5, a * 25 + (middley + 2.5 - spaceone / 2)), radius=[(6,6),(6,6),(6,6),(6,6)])
                        biases = "{:.2f}".format(value)
                        #nodeslabel = Label(text=biases, size_hint=(None,None), font_size=9, pos=(middlex-31.5, a * 25 + (middley-31.5 - spaceone / 2)),
                                                #color=(1,1,1, 0.25 + colorlayerone/1.33))
                        #draw.add_widget(nodeslabel)
                        #biaseslabels.append(nodeslabel)
                        i = 0
                        e = 0
                    else:
                        with draw.canvas:
                            Ellipse(pos=(middlex + 7, a * 25 + (middley - spaceone / 2) + 9), size=(5, 5))

                    if numdata < 10:
                        for d in range(numdata):
                            mypoints = [middlex - 205, d * 50 + (middley - 10 - space / 2) + 30, middlex,
                                      a * 25 + (middley + 9 - spaceone / 2)]
                            MyApp.listoflinesappend(self, 0, a, d, linecolorinput, draw.canvas, mydata.networklist, mypoints)


                    else:
                        for d in range(10):
                            mypoints = [middlex - 205, d * 50 + (middley - 10 - 575 / 2) + 30, middlex,
                                             a * 25 + (middley + 9 - spaceone / 2)]
                            MyApp.listoflinesappend(self, 0, a, d, linecolorinput, draw.canvas, mydata.networklist, mypoints)
            draw.add_widget(Label(text = str(mydata.list_hidden[0]), pos = (middlex - 40 , middley+220), size_hint=(None,None), font_size=13))

            for e in range(numoutputs):
                if mydata.list_hidden[0] < 21:
                    for a in range(mydata.list_hidden[0]):
                        mypoints =[middlex+18, a * 25 + (middley  - spaceone / 2)+ 9, middlex+230,
                                         e * 50 + (middley+ 20 - spaceoutput / 2) ]
                        MyApp.listoflinesappend(self, 1, e, a, linecoloroutput, draw.canvas, mydata.networklist, mypoints)
                else:
                    for a in range(20):
                        mypoints =[middlex+18, a * 25 + (middley  - spaceone / 2)+ 9, middlex+230,
                                         e * 50 + (middley+ 20 - spaceoutput / 2) ]
                        MyApp.listoflinesappend(self, 1, e, a, linecoloroutput, draw.canvas, mydata.networklist, mypoints)

        if len(mydata.list_hidden) == 2:
            # hiddenlayerone
            if mydata.list_hidden[1] < 21:
                spacetwo = mydata.list_hidden[1] * 18 + (mydata.list_hidden[1] - 1) * 7
            else:
                spacetwo = 20 * 18 + (20 - 1) * 7
            valuefloat = 0
            linecoloroutput = 0
            linecolorinput = 0
            self.hiddenlayeronelabelone.pos = (middletitlex - 108, middletitley - 50)
            titlelayout.add_widget(self.hiddenlayeronelabelone)
            for a in range(mydata.list_hidden[0]):
                if valuefloat < abs(mydata.networklist[0][a]['weights'][-1]):
                    valuefloat = abs(mydata.networklist[0][a]['weights'][-1])

                for i in range(numdata):
                    if linecolorinput < abs(mydata.networklist[0][a]['weights'][i]):
                        linecolorinput = abs(mydata.networklist[0][a]['weights'][i])
            if mydata.list_hidden[0] < 21:
                for a in range(mydata.list_hidden[0]):
                    with draw.canvas:
                        value = mydata.networklist[0][a]['weights'][-1]
                        colorlayerone = abs(mydata.networklist[0][a]['weights'][-1]) / valuefloat
                        Color(1, 1, 1, 0.25 + colorlayerone / 1.33)
                        RoundedRectangle(size=(18, 18), pos=(middlex-80, a * 25 + (middley - spaceone / 2)))
                        Color(0,0,0,1)
                        RoundedRectangle(size=(13, 13), pos=(middlex + 2.5 - 80, a * 25 + (middley + 2.5 - spaceone / 2)),
                                         radius=[(6, 6), (6, 6), (6, 6), (6, 6)])
                        biases = "{:.2f}".format(value)
                        #draw.nodeslabel = Label(text=biases,size_hint=(None,None),font_size=10, pos=(middlex-111.5, a * 50 + (middley-31.5 - spaceone / 2)),
                                                #color=(1,1,1, 0.25 + colorlayerone/1.33))
                        #draw.add_widget(draw.nodeslabel)

                    if numdata < 10:
                        for i in range(numdata):
                            mypoints = [middlex - 205, i * 50 + (middley - space / 2) + 17.5, middlex - 80,
                                             a * 25 + (middley + 9 - spaceone / 2)]
                            MyApp.listoflinesappend(self, 0, a, i, linecolorinput, draw.canvas, mydata.networklist,
                                                    mypoints)

                    else:
                        for i in range(10):
                            mypoints = [middlex - 205, i * 50 + (middley - space / 2) + 17.5, middlex - 80,
                                             a * 25 + (middley + 9 - spaceone / 2)]
                            MyApp.listoflinesappend(self, 0, a, i, linecolorinput, draw.canvas, mydata.networklist, mypoints)
            else:
                for a in range(20):
                    if a < 8 or a > 11:
                        with draw.canvas:
                            value = mydata.networklist[0][a]['weights'][-1]
                            colorlayerone = abs(mydata.networklist[0][a]['weights'][-1]) / valuefloat
                            Color(1, 1, 1, 0.25 + colorlayerone / 1.33)
                            RoundedRectangle(size=(18, 18), pos=(middlex - 80, a * 25 + (middley - spaceone / 2)))
                            Color(0, 0, 0, 1)
                            RoundedRectangle(size=(13, 13),
                                             pos=(middlex + 2.5 - 80, a * 25 + (middley + 2.5 - spaceone / 2)),
                                             radius=[(6, 6), (6, 6), (6, 6), (6, 6)])
                            biases = "{:.2f}".format(value)
                            # draw.nodeslabel = Label(text=biases,size_hint=(None,None),font_size=10, pos=(middlex-111.5, a * 50 + (middley-31.5 - spaceone / 2)),
                            # color=(1,1,1, 0.25 + colorlayerone/1.33))
                            # draw.add_widget(draw.nodeslabel)
                    else:
                        with draw.canvas:
                            Ellipse(pos=(middlex - 73, a * 25 + (middley - spaceone / 2) + 9), size=(5, 5))


                    if numdata < 10:
                        for i in range(numdata):
                            mypoints = [middlex - 205, i * 50 + (middley - space / 2) + 17.5, middlex - 80,
                                        a * 25 + (middley + 9 - spaceone / 2)]
                            MyApp.listoflinesappend(self, 0, a, i, linecolorinput, draw.canvas, mydata.networklist,
                                                    mypoints)

                    else:
                        for i in range(10):
                            mypoints = [middlex - 205, i * 50 + (middley - space / 2) + 17.5, middlex - 80,
                                        a * 25 + (middley + 9 - spaceone / 2)]
                            MyApp.listoflinesappend(self, 0, a, i, linecolorinput, draw.canvas, mydata.networklist,
                                                    mypoints)
            draw.add_widget(Label(text=str(mydata.list_hidden[0]), pos=(middlex - 121, middley + 220), size_hint=(None, None),font_size=13))
            # hiddenlayertwo
            valuefloat = 0
            linecoloroutput = 0
            linecolorinput = 0

            self.hiddenlayeronelabeltwo.pos = (middletitlex + 52, middletitley - 50)
            titlelayout.add_widget(self.hiddenlayeronelabeltwo)
            for d in range(mydata.list_hidden[1]):
                if valuefloat < abs(mydata.networklist[1][d]['weights'][-1]):
                    valuefloat = abs(mydata.networklist[1][d]['weights'][-1])
                    for j in range(mydata.list_hidden[0]):
                        if linecolorinput < abs(mydata.networklist[1][d]['weights'][j]):
                            linecolorinput = abs(mydata.networklist[1][d]['weights'][j])
                        j = j + 1
            for e in range(numoutputs):
                for d in range(mydata.list_hidden[1]):
                    if linecoloroutput < abs(mydata.networklist[2][e]['weights'][d]):
                        linecoloroutput = abs(mydata.networklist[2][e]['weights'][d])

            if mydata.list_hidden[1] < 21:
                for d in range(mydata.list_hidden[1]):
                    with draw.canvas:
                        value = mydata.networklist[1][d]['weights'][-1]
                        colorlayerone = abs(mydata.networklist[1][d]['weights'][-1]) / valuefloat
                        Color(1, 1, 1, 0.25 + colorlayerone / 1.33)
                        RoundedRectangle(size=(18, 18), pos=(middlex + 80, d * 25 + (middley - spacetwo / 2)))
                        Color(0,0,0,1)
                        RoundedRectangle(size=(13, 13), pos=(middlex + 82.5, d * 25 + (middley + 2.5 - spacetwo / 2)),
                                         radius=[(6, 6), (6, 6), (6, 6), (6, 6)])
                        biases = "{:.2f}".format(value)
                        #draw.nodeslabel = Label(text=biases, font_size=10,size_hint=(None,None), pos=(middlex +49, d * 50 + (middley-31.5 - spacetwo / 2)),
                                                #color=(1,1,1, 0.25 + colorlayerone/1.33))
                        #draw.add_widget(draw.nodeslabel)
                    if mydata.list_hidden[0] < 21:
                        for j in range(mydata.list_hidden[0]):
                            mypoints = [middlex -62, j * 25 + (middley+9 - spaceone / 2), middlex+80, d * 25 + (middley+9 - spacetwo / 2)]
                            MyApp.listoflinesappend(self, 1, d, j, linecolorinput, draw.canvas, mydata.networklist, mypoints)
                    else:
                        for j in range(20):
                            mypoints = [middlex -62, j * 25 + (middley+9 - spaceone / 2), middlex+80, d * 25 + (middley+9 - spacetwo / 2)]
                            MyApp.listoflinesappend(self, 1, d, j, linecolorinput, draw.canvas, mydata.networklist, mypoints)
            else:
                for d in range(20):
                    if d < 8 or d > 11:
                        with draw.canvas:
                            value = mydata.networklist[1][d]['weights'][-1]
                            colorlayerone = abs(mydata.networklist[1][d]['weights'][-1]) / valuefloat
                            Color(1, 1, 1, 0.25 + colorlayerone / 1.33)
                            RoundedRectangle(size=(18, 18), pos=(middlex + 80, d * 25 + (middley - spacetwo / 2)))
                            Color(0,0,0,1)
                            RoundedRectangle(size=(13, 13), pos=(middlex + 82.5, d * 25 + (middley + 2.5 - spacetwo / 2)),
                                             radius=[(6, 6), (6, 6), (6, 6), (6, 6)])
                            biases = "{:.2f}".format(value)
                            #draw.nodeslabel = Label(text=biases, font_size=10,size_hint=(None,None), pos=(middlex +49, d * 50 + (middley-31.5 - spacetwo / 2)),
                                                    #color=(1,1,1, 0.25 + colorlayerone/1.33))
                            #draw.add_widget(draw.nodeslabel)
                        if mydata.list_hidden[0] < 21:
                            for j in range(mydata.list_hidden[0]):
                                mypoints = [middlex -62, j * 25 + (middley+9 - spaceone / 2), middlex+80, d * 25 + (middley+9 - spacetwo / 2)]
                                MyApp.listoflinesappend(self, 1, d, j, linecolorinput, draw.canvas, mydata.networklist, mypoints)
                        else:
                            for j in range(20):
                                mypoints = [middlex -62, j * 25 + (middley+9 - spaceone / 2), middlex+80, d * 25 + (middley+9 - spacetwo / 2)]
                                MyApp.listoflinesappend(self, 1, d, j, linecolorinput, draw.canvas, mydata.networklist, mypoints)
                    else:
                        with draw.canvas:
                            Ellipse(pos=(middlex + 87, d * 25 + (middley - spacetwo / 2) + 9), size=(5, 5))
            for e in range(numoutputs):
                if mydata.list_hidden[1] < 21:
                    for d in range(mydata.list_hidden[1]):
                        mypoints = [middlex+98, d * 25 + (middley+9 - spacetwo / 2), middlex + 230, e * 50 + (middley +20 - spaceoutput / 2)]
                        MyApp.listoflinesappend(self, 2, e, d, linecoloroutput, draw.canvas, mydata.networklist, mypoints)
                else:
                    for d in range(20):
                        mypoints = [middlex + 98, d * 25 + (middley + 9 - spacetwo / 2), middlex + 230,e * 50 + (middley + 20 - spaceoutput / 2)]
                        MyApp.listoflinesappend(self, 2, e, d, linecoloroutput, draw.canvas, mydata.networklist,mypoints)
            draw.add_widget(Label(text=str(mydata.list_hidden[1]), pos=(middlex + 38, middley + 220), size_hint=(None, None),font_size=13))

        if len(mydata.list_hidden) == 3:
            # hiddenlayerone
            if mydata.list_hidden[2] < 21:
                spacethree = mydata.list_hidden[2] * 18 + (mydata.list_hidden[2] - 1) * 7
            else:
                spacethree = 20 * 18 + (20 - 1) * 7
            if mydata.list_hidden[1] < 21:
                spacetwo = mydata.list_hidden[1] * 18 + (mydata.list_hidden[1] - 1) * 7
            else:
                spacetwo = 20 * 18 + (20 - 1) * 7
            valuefloat = 0
            linecolorinput = 0
            self.hiddenlayeronelabelone.pos = (middletitlex - 158, middletitley - 50)
            titlelayout.add_widget(self.hiddenlayeronelabelone)
            for a in range(mydata.list_hidden[0]):
                if valuefloat < abs(mydata.networklist[0][a]['weights'][-1]):
                    valuefloat = abs(mydata.networklist[0][a]['weights'][-1])

                for i in range(numdata):
                    if linecolorinput < abs(mydata.networklist[0][a]['weights'][i]):
                        linecolorinput = abs(mydata.networklist[0][a]['weights'][i])

            if mydata.list_hidden[0] < 21:
                for a in range(mydata.list_hidden[0]):
                    with draw.canvas:
                        value = mydata.networklist[0][a]['weights'][-1]
                        colorlayerone = abs(mydata.networklist[0][a]['weights'][-1]) / valuefloat
                        Color(1, 1, 1, 0.25 + colorlayerone / 1.33)
                        RoundedRectangle(size=(18, 18), pos=(middlex - 130, a * 25 + (middley - spaceone / 2)))
                        Color(0,0,0,1)
                        RoundedRectangle(size=(13, 13), pos=(middlex + 2.5 - 130, a * 25 + (middley + 2.5 - spaceone / 2)),
                                         radius=[(6, 6), (6, 6), (6, 6), (6, 6)])
                        biases = "{:.2f}".format(value)
                        #draw.nodeslabel = Label(text=biases,size_hint=(None,None), font_size=10, pos=(middlex-161.5, a * 50 + (middley-31.5 - spaceone / 2)),
                                                #color=(1,1,1, 0.25 + colorlayerone/1.33))
                        #draw.add_widget(draw.nodeslabel)

                    if numdata < 10:
                        for i in range(numdata):
                            mypoints = [middlex - 205, i * 50 + (middley - space / 2) + 17.5, middlex - 130,
                                             a * 25 + (middley + 9 - spaceone / 2)]
                            MyApp.listoflinesappend(self, 0, a, i, linecolorinput, draw.canvas, mydata.networklist,
                                                    mypoints)

                    else:
                        for i in range(10):
                            mypoints = [middlex - 205, i * 50 + (middley - space / 2) + 17.5, middlex - 130,
                                             a * 25 + (middley + 9 - spaceone / 2)]
                            MyApp.listoflinesappend(self, 0, a, i, linecolorinput, draw.canvas, mydata.networklist,
                                                    mypoints)
            else:
                for a in range(20):
                    if a < 8 or a > 11:
                        with draw.canvas:
                            value = mydata.networklist[0][a]['weights'][-1]
                            colorlayerone = abs(mydata.networklist[0][a]['weights'][-1]) / valuefloat
                            Color(1, 1, 1, 0.25 + colorlayerone / 1.33)
                            RoundedRectangle(size=(18, 18), pos=(middlex - 130, a * 25 + (middley - spaceone / 2)))
                            Color(0, 0, 0, 1)
                            RoundedRectangle(size=(13, 13),
                                             pos=(middlex + 2.5 - 130, a * 25 + (middley + 2.5 - spaceone / 2)),
                                             radius=[(6, 6), (6, 6), (6, 6), (6, 6)])
                            biases = "{:.2f}".format(value)
                            # draw.nodeslabel = Label(text=biases,size_hint=(None,None), font_size=10, pos=(middlex-161.5, a * 50 + (middley-31.5 - spaceone / 2)),
                            # color=(1,1,1, 0.25 + colorlayerone/1.33))
                            # draw.add_widget(draw.nodeslabel)
                    else:
                        with draw.canvas:
                            Ellipse(pos=(middlex - 123, a * 25 + (middley - spaceone / 2) + 9), size=(5, 5))

                    if numdata < 10:
                        for i in range(numdata):
                            mypoints = [middlex - 205, i * 50 + (middley - space / 2) + 17.5, middlex - 130,
                                        a * 25 + (middley + 9 - spaceone / 2)]
                            MyApp.listoflinesappend(self, 0, a, i, linecolorinput, draw.canvas, mydata.networklist,
                                                    mypoints)

                    else:
                        for i in range(10):
                            mypoints = [middlex - 205, i * 50 + (middley - space / 2) + 17.5, middlex - 130,
                                        a * 25 + (middley + 9 - spaceone / 2)]
                            MyApp.listoflinesappend(self, 0, a, i, linecolorinput, draw.canvas, mydata.networklist,
                                                    mypoints)
            draw.add_widget(
                Label(text=str(mydata.list_hidden[0]), pos=(middlex - 171, middley + 220), size_hint=(None, None),
                      font_size=13))
            # hiddenlayertwo
            valuefloat = 0
            linecolorinput = 0

            self.hiddenlayeronelabeltwo.pos = (middletitlex - 28, middletitley - 50)
            titlelayout.add_widget(self.hiddenlayeronelabeltwo)
            for d in range(mydata.list_hidden[1]):
                if valuefloat < abs(mydata.networklist[1][d]['weights'][-1]):
                    valuefloat = abs(mydata.networklist[1][d]['weights'][-1])

                for j in range(mydata.list_hidden[0]):
                    if linecolorinput < abs(mydata.networklist[1][d]['weights'][j]):
                        linecolorinput = abs(mydata.networklist[1][d]['weights'][j])

            if mydata.list_hidden[1] < 21:
                for d in range(mydata.list_hidden[1]):
                    with draw.canvas:
                        value = mydata.networklist[1][d]['weights'][-1]
                        colorlayerone = abs(mydata.networklist[1][d]['weights'][-1]) / valuefloat
                        Color(1, 1, 1, 0.25 + colorlayerone / 1.33)
                        RoundedRectangle(size=(18, 18), pos=(middlex, d * 25 + (middley - spacetwo / 2)))
                        Color(0,0,0,1)
                        RoundedRectangle(size=(13, 13), pos=(middlex + 2.5, d * 25 + (middley + 2.5 - spacetwo / 2)),
                                         radius=[(6, 6), (6, 6), (6, 6), (6, 6)])
                        biases = "{:.2f}".format(value)
                        #draw.nodeslabel = Label(text=biases, font_size=10,size_hint=(None,None), pos=(middlex-31.5, d * 50 + (middley-31.5 - spacetwo / 2)),
                                                #color=(1,1,1, 0.25 + colorlayerone/1.33))
                        #draw.add_widget(draw.nodeslabel)

                    if mydata.list_hidden[0] < 21:
                        for j in range(mydata.list_hidden[0]):
                            mypoints = [middlex-112, j * 25 + (middley+9 - spaceone / 2), middlex, d * 25 + (middley+9 - spacetwo / 2)]
                            MyApp.listoflinesappend(self, 1, d, j, linecolorinput, draw.canvas, mydata.networklist, mypoints)
                    else:
                        for j in range(20):
                            mypoints = [middlex-112, j * 25 + (middley+9 - spaceone / 2), middlex, d * 25 + (middley+9 - spacetwo / 2)]
                            MyApp.listoflinesappend(self, 1, d, j, linecolorinput, draw.canvas, mydata.networklist, mypoints)
            else:
                for d in range(20):
                    if d < 8 or d > 11:
                        with draw.canvas:
                            value = mydata.networklist[1][d]['weights'][-1]
                            colorlayerone = abs(mydata.networklist[1][d]['weights'][-1]) / valuefloat
                            Color(1, 1, 1, 0.25 + colorlayerone / 1.33)
                            RoundedRectangle(size=(18, 18), pos=(middlex, d * 25 + (middley - spacetwo / 2)))
                            Color(0, 0, 0, 1)
                            RoundedRectangle(size=(13, 13), pos=(middlex + 2.5, d * 25 + (middley + 2.5 - spacetwo / 2)),
                                             radius=[(6, 6), (6, 6), (6, 6), (6, 6)])
                            biases = "{:.2f}".format(value)
                            # draw.nodeslabel = Label(text=biases, font_size=10,size_hint=(None,None), pos=(middlex-31.5, d * 50 + (middley-31.5 - spacetwo / 2)),
                            # color=(1,1,1, 0.25 + colorlayerone/1.33))
                            # draw.add_widget(draw.nodeslabel)
                    else:
                        with draw.canvas:
                            Ellipse(pos=(middlex + 7, d * 25 + (middley - spacetwo / 2) + 9), size=(5, 5))

                    if mydata.list_hidden[0] < 21:
                        for j in range(mydata.list_hidden[0]):
                            mypoints = [middlex - 112, j * 25 + (middley + 9 - spaceone / 2), middlex,
                                        d * 25 + (middley + 9 - spacetwo / 2)]
                            MyApp.listoflinesappend(self, 1, d, j, linecolorinput, draw.canvas, mydata.networklist,
                                                    mypoints)
                    else:
                        for j in range(20):
                            mypoints = [middlex - 112, j * 25 + (middley + 9 - spaceone / 2), middlex,
                                        d * 25 + (middley + 9 - spacetwo / 2)]
                            MyApp.listoflinesappend(self, 1, d, j, linecolorinput, draw.canvas, mydata.networklist,
                                                    mypoints)
            draw.add_widget(
                Label(text=str(mydata.list_hidden[0]), pos=(middlex - 40, middley + 220), size_hint=(None, None),
                      font_size=13))
            #hiddenlayerthree
            valuefloat = 0
            linecoloroutput = 0
            linecolorinput = 0

            self.hiddenlayeronelabelthree.pos = (middletitlex + 101, middletitley - 50)
            titlelayout.add_widget(self.hiddenlayeronelabelthree)
            for d in range(mydata.list_hidden[2]):
                if valuefloat < abs(mydata.networklist[2][d]['weights'][-1]):
                    valuefloat = abs(mydata.networklist[2][d]['weights'][-1])


                for j in range(mydata.list_hidden[1]):
                    if linecolorinput < abs(mydata.networklist[2][d]['weights'][j]):
                        linecolorinput = abs(mydata.networklist[2][d]['weights'][j])

                for e in range(numoutputs):
                    if linecoloroutput < abs(mydata.networklist[3][e]['weights'][d]):
                        linecoloroutput = abs(mydata.networklist[3][e]['weights'][d])

            if mydata.list_hidden[2] < 21:
                for d in range(mydata.list_hidden[2]):
                    with draw.canvas:
                        value = mydata.networklist[2][d]['weights'][-1]
                        colorlayerone = abs(mydata.networklist[2][d]['weights'][-1]) / valuefloat
                        Color(1, 1, 1, 0.25 + colorlayerone / 1.33)
                        RoundedRectangle(size=(18, 18), pos=(middlex+130, d * 25 + (middley - spacethree / 2)))
                        Color(0,0,0,1)
                        RoundedRectangle(size=(13, 13), pos=(middlex + 132.5, d * 25 + (middley + 2.5 - spacethree / 2)),
                                         radius=[(6, 6), (6, 6), (6, 6), (6, 6)])
                        biases = "{:.2f}".format(value)
                        #draw.nodeslabel = Label(text=biases, font_size=10,size_hint=(None,None), pos=(middlex+99, d * 50 + (middley-31.5 - spacethree / 2)),
                                                #color=(1,1,1, 0.25 + colorlayerone/1.33))
                        #draw.add_widget(draw.nodeslabel)

                    if mydata.list_hidden[1] < 21:
                        for l in range(mydata.list_hidden[1]):
                            mypoints = [middlex+18, l * 25 + (middley+9 - spacetwo / 2), middlex+130, d * 25 + (middley+9 - spacethree / 2)]
                            MyApp.listoflinesappend(self, 2, d, l, linecolorinput, draw.canvas, mydata.networklist, mypoints)
                    else:
                        for l in range(20):
                            mypoints = [middlex+18, l * 25 + (middley+9 - spacetwo / 2), middlex+130, d * 25 + (middley+9 - spacethree / 2)]
                            MyApp.listoflinesappend(self, 2, d, l, linecolorinput, draw.canvas, mydata.networklist, mypoints)
            else:
                for d in range(20):
                    if d < 8 or d > 11:
                        with draw.canvas:
                            value = mydata.networklist[2][d]['weights'][-1]
                            colorlayerone = abs(mydata.networklist[2][d]['weights'][-1]) / valuefloat
                            Color(1, 1, 1, 0.25 + colorlayerone / 1.33)
                            RoundedRectangle(size=(18, 18), pos=(middlex + 130, d * 25 + (middley - spacethree / 2)))
                            Color(0, 0, 0, 1)
                            RoundedRectangle(size=(13, 13),
                                             pos=(middlex + 132.5, d * 25 + (middley + 2.5 - spacethree / 2)),
                                             radius=[(6, 6), (6, 6), (6, 6), (6, 6)])
                            biases = "{:.2f}".format(value)
                            # draw.nodeslabel = Label(text=biases, font_size=10,size_hint=(None,None), pos=(middlex+99, d * 50 + (middley-31.5 - spacethree / 2)),
                            # color=(1,1,1, 0.25 + colorlayerone/1.33))
                            # draw.add_widget(draw.nodeslabel)
                    else:
                        with draw.canvas:
                            Ellipse(pos=(middlex +137, d * 25 + (middley - spacethree / 2) + 9), size=(5, 5))

                    if mydata.list_hidden[1] < 21:
                        for l in range(mydata.list_hidden[1]):
                            mypoints = [middlex + 18, l * 25 + (middley + 9 - spacetwo / 2), middlex + 130,
                                        d * 25 + (middley + 9 - spacethree / 2)]
                            MyApp.listoflinesappend(self, 2, d, l, linecolorinput, draw.canvas, mydata.networklist,
                                                    mypoints)
                    else:
                        for l in range(20):
                            mypoints = [middlex + 18, l * 25 + (middley + 9 - spacetwo / 2), middlex + 130,
                                        d * 25 + (middley + 9 - spacethree / 2)]
                            MyApp.listoflinesappend(self, 2, d, l, linecolorinput, draw.canvas, mydata.networklist,
                                                    mypoints)

            for e in range(numoutputs):
                if mydata.list_hidden[2] < 21:
                    for d in range(mydata.list_hidden[2]):
                        mypoints = [middlex+148, d * 25 + (middley + 9 - spacethree / 2), middlex+230, e * 50 + (middley + 20 - spaceoutput / 2)]
                        MyApp.listoflinesappend(self, 3, e, d, linecoloroutput, draw.canvas, mydata.networklist, mypoints)
                else:
                    for d in range(20):
                        mypoints = [middlex+148, d * 25 + (middley + 9 - spacethree / 2), middlex+230, e * 50 + (middley + 20 - spaceoutput / 2)]
                        MyApp.listoflinesappend(self, 3, e, d, linecoloroutput, draw.canvas, mydata.networklist, mypoints)
            draw.add_widget(Label(text=str(mydata.list_hidden[0]), pos=(middlex+89, middley + 220), size_hint=(None, None),font_size=13))

        # output layer
        for c in range(numoutputs):
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
        c = 0
        for row in mydata.outputimageslist:
            image = Filechooser.WidgetTexture(row)
            draw.add_widget(image)
            image.pos = middlex - 115, c * 50 + (middley - spaceoutput / 2) - 240
            c += 1
        draw.add_widget(Label(text=str(numoutputs), pos=(middlex + 200, middley + 220), size_hint=(None, None)))


        # input layer
        if numdata < 11:
            for b in range(numdata):
                with draw.canvas:
                    Color(.6, .87, .89)
                    RoundedRectangle(size=(35, 35), pos=(middlex - 240, b * 50 + (middley - space / 2)))
                    Color(0, 0, 0)
                    RoundedRectangle(size=(30, 30), pos=(middlex - 237.5, b * 50 + (middley + 2.5 - space / 2)),
                                     radius=[(6, 6), (6, 6), (6, 6), (6, 6)])
        else:
            for b in range(10):
                if b < 4 or b > 5:
                    with draw.canvas:
                        Color(.6, .87, .89)
                        RoundedRectangle(size=(35, 35), pos=(middlex - 240, b * 50 + (middley - space / 2)))
                        Color(0, 0, 0)
                        RoundedRectangle(size=(30, 30), pos=(middlex - 237.5, b * 50 + (middley + 2.5 - space / 2)),
                                         radius=[(6, 6), (6, 6), (6, 6), (6, 6)])
                else:
                    with draw.canvas:
                        Color(.4 + b / 9, .67 + b / 9, .69 + b / 9)
                        Ellipse(pos=(middlex - 225, b * 50 + (middley - space / 2)+ .7), size=(5, 5))
                        Ellipse(pos=(middlex - 225, b * 50 + (middley - space / 2) + 27.3), size=(5, 5))
        draw.add_widget(Label(text=str(numdata), pos=(middlex - 270, middley + 220), size_hint=(None, None)))


    class Lblleft():

        def __init__(self, type, dataerror):
            global scores, varerrorlayout, n_epoch, n_folds, l_rate
            global list_hidden, network, n_outputs, n_inputs, currentpage, list_hidden, outputimageslist

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
            self.outputimageslist = outputimageslist.copy()

    class LineParam():
        def __init__(self, layer, hidden, offset, maximum, value, color, linewidget, canvas):
            self.layer = layer
            self.hidden = hidden
            self.offset = offset
            self.maximum = maximum
            self.linewidget = linewidget
            self.canvas = canvas
            self.value = value
            self.color = color

        def remove(self):
            self.canvas.remove(self.linewidget)

        def change(self):
            newcolor = self.value / self.maximum
            if (abs(newcolor - self.color) > 0.02):
                self.canvas.remove(self.linewidget)
                self.color = newcolor
                self.canvas.add(Color(self.color, self.color, self.color))
                self.canvas.add(self.linewidget)
                return 1
            return 0

    def listoflinesappend(self, layer, hidden, offset, maximum, canvas, networklist, linepoints):
        value = abs(networklist[layer][hidden]['weights'][offset])
        color = value / maximum
        with canvas:
            Color(color, color, color)
            linewidget = Line(points=linepoints)
        listoflines.append(MyApp.LineParam(layer, hidden, offset, maximum, value, color, linewidget, canvas))

    #When self.btnError is pressed
    #The lbls on the left
    def leftlabels(self, textlayout, leftpanelitem, *largs):
        global scores, varerrorlayout, n_epoch, n_folds, l_rate, labelcycles, labelfolds, errorlbl
        global list_hidden, network, n_outputs, n_inputs, currentpage


        lbldate= Label(text=leftpanelitem.datestr, markup=True, halign="left", size_hint=(1, .1))
        lbldate.bind(size=lbldate.setter('text_size'))
        labelfile = (Label(text='[b] File: [/b][i]' + basename(leftpanelitem.filename), markup=True, halign="left",
                           size_hint=(1, .1)))
        labelfile.bind(size=labelfile.setter('text_size'))
        labeltype = (Label(text='[b] [color=4BDCFF]Test: [/b][/color]' + leftpanelitem.type, markup=True,
                           halign="left",
                           size_hint=(1, .1), color=(1, 1, 1)))
        labeltype.bind(size=labeltype.setter('text_size'))
        labelcycles.text = '[b] [color=D464EB]Cycles: [/b][/color]' + str(leftpanelitem.n_epoch)
        labelcycles.bind(size=labelcycles.setter('text_size'))
        labelrate = (Label(text='[b] [color=D464EB]Rate: [/b][/color]' + str(leftpanelitem.l_rate), markup=True,
                           halign="left",
                           size_hint=(1, .1), color=(1, 1, 1)))
        labelrate.bind(size=labelrate.setter('text_size'))
        labelfolds.text = '[b] [color=D464EB]Folds: [/b][/color]' + str(leftpanelitem.n_folds)
        labelfolds.bind(size=labelfolds.setter('text_size'))
        errorlbl.text = ''
        errorlbl.bind(size=errorlbl.setter('text_size'))
        varerrorlayout = True
        labelaccuracy = (Label(text='[b] Accuracy: [/b]' + str("{:.2f}".format(leftpanelitem.dataerror)) + '%', markup=True,
                                   halign="left", size_hint=(1, .1)))
        labelaccuracy.bind(size=labelaccuracy.setter('text_size'))




        if currentpage == 1:
            self.btnlesssidetext.disabled = True
        if currentpage != 1:
            self.btnlesssidetext.disabled = False

        textlayout.clear_widgets()

        textlayout.add_widget(lbldate)
        textlayout.add_widget(labelfile)
        textlayout.add_widget(labeltype)
        textlayout.add_widget(labelcycles)
        textlayout.add_widget(labelrate)
        textlayout.add_widget(labelfolds)
        textlayout.add_widget(errorlbl)
        textlayout.add_widget(labelaccuracy)



        self.lblcurrentpage.text = ('[b]Page:[/b]' + ' ' + str("{:.0f}".format(currentpage)) + '/' + str("{:.0f}".format(len(leftpanellist))))

    def establishbtnpages(self):
        global currentpage
        if currentpage == 1:
            self.btnlesssidetext.disabled = True
        else:
            self.btnlesssidetext.disabled = False
        if currentpage == len(leftpanellist):
            self.btnplussidetext.disabled = True
        else:
            self.btnplussidetext.disabled = False

    #Change between pages
    def leftlabelswitch(self, textlayout, dataerror, type, *largs):
        global currentpage
        if type == 'btnplus':
            currentpage += 1
            mydata = leftpanellist[currentpage -1]
            self.drawinglblblackkv(self, textlayout, mydata)
        if type == 'btnless':
            currentpage -= 1
            mydata = leftpanellist[currentpage - 1]
            self.drawinglblblackkv(textlayout, mydata)
        MyApp.establishbtnpages(self)

        self.lblcurrentpage.text = ('[b]Page:[/b]' + ' ' + str("{:.0f}".format(currentpage)) + '/' + str(
            "{:.0f}".format(len(leftpanellist))))

    #Plain drawing small size
    def drawingtryout(self, draw, titlelayout, *largs):
        global a, b, list_hidden, network, n_outputs, n_inputs, accuracylabel, progressbar

        a = 0
        b = 0
        c = 0


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


        titlelayout.remove_widget(self.hiddenlayeronelabelone)
        titlelayout.remove_widget(self.hiddenlayeronelabeltwo)
        titlelayout.remove_widget(self.hiddenlayeronelabelthree)



        if len(list_hidden) == 1:
            # hiddenlayerone
            #self.hiddenlayeronelabelone.pos =(middletitlex - 28, middletitley - 50)
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
            #self.hiddenlayeronelabelone.pos=(middletitlex-108, middletitley - 50)
            #titlelayout.add_widget(self.hiddenlayeronelabelone)
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
            #self.hiddenlayeronelabelone.pos=(middletitlex-158, middletitley-50)
            #titlelayout.add_widget(self.hiddenlayeronelabelone)
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
        titlelayout.remove_widget(self.hiddenlayeronelabelone)
        titlelayout.remove_widget(self.hiddenlayeronelabeltwo)
        titlelayout.remove_widget(self.hiddenlayeronelabelthree)
        titlelayout.remove_widget(self.outputlabel)
        titlelayout.remove_widget(self.inputlabel)

    def storeleftpanellist(self):
        try:
            mystore = JsonStore('neuroncfg.json')
            mystore.clear()
            mystore.put('leftpanelhistorycount', name="numberofitems", value=len(leftpanellist))
            i = 0
            for panel in leftpanellist:
                mystore.put('leftpanelhistory' + str(i), name="lelftpanellist", value=panel.__dict__)
                i += 1
        except:
            os.remove('neuroncfg.json')

    def restoreleftpanellist(self):
        global currentpage, leftpanellist
        try:
            mystore = JsonStore('neuroncfg.json')
            if mystore.exists('leftpanelhistorycount'):
                mynumberofitems = mystore.get('leftpanelhistorycount')['value']
                leftpanellist.clear()
                for index in range(mynumberofitems):
                    mydatadict = mystore.get('leftpanelhistory' + str(index))['value']
                    mydataclasss = MyApp.Lblleft(type='', dataerror=0)
                    for key in mydatadict:
                        setattr(mydataclasss, key, mydatadict[key])
                    leftpanellist.append(mydataclasss)
                    currentpage = len(leftpanellist)
                    MyApp.establishbtnpages(self)
        except:
            os.remove('neuroncfg.json')

    def addclasstopanel(self, mypannelclass):
        global currentpage
        leftpanellist.append(mypannelclass)
        currentpage = len(leftpanellist)
        MyApp.establishbtnpages(self)
        MyApp.storeleftpanellist(self)

    def settingslayout(self):
        global prediction, biaseslabels, varerrorlayout
        prediction = False
        varerrorlayout = False
        self.btnlayout.height = 300
        self.layoutmainscreen.height = 350
        self.layoutinfoleft.size_hint_x = 0.2
        self.layoutinforight.size_hint_x = 0.2
        self.layouttrainscreen = self.parentlayout.ids.idLayoutNeuron.ids.idAnchorLayoutNeuron
        for element in biaseslabels:
            layoutneuronmainscreen.remove_widget(element)
        biaseslabels = list()

        self.layout.remove_widget(self.btnTrain)
        self.layout.remove_widget(self.btnError)
        self.layout.remove_widget(self.btnPredict)
        self.layoutinfotop.remove_widget(self.btnSettings)
        self.layoutinfotop.remove_widget(self.btnBack)
        self.layout.remove_widget(self.btnPredict)
        self.layout.remove_widget(self.btnProcessPredict)
        self.layout.remove_widget(self.btnCancel)
        self.layout.remove_widget(self.btnnewinput)
        self.clear(layoutneuronmainscreen, layoutlabels)
        self.layout.add_widget(self.layoutsettingsvertical)

        self.btnTrain.disabled = True
        self.btnError.disabled = True
        self.btnPredict.disabled = True

    def btnTrainpressed(self):
            global progressbar, accuracylabel, scores, n_epoch, l_rate, numdata, valuetrain
            global network, n_inputs, n_outputs, n_folds, dataset, list_hidden, trainclock
            global printth, stoptrainning, currentfold, currentcycle, timestart
            layoutprogressbar = self.parentlayout.ids.mainseparatorprogressbar
            layoutprogressbar.remove_widget(progressbar)
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

            currentfold = 0
            currentcycle = 0
            progressbar.value = 0
            valuetrain = 0
            stoptrainning = False
            printth = 0
            timestart = time.monotonic()
            trainclock = Clock.schedule_interval(layoutpbtrainnext, 1 / 10)
            network = initialize_network(n_inputs, list_hidden, n_outputs, transfer_sigmoid, transfer_sigmoid_derivative)
            threading.Thread(target=train_network, args=(network, dataset, l_rate, n_epoch, n_outputs, 'train', myevent, self)).start()
            self.layout.remove_widget(lbltimeleft)
            self.layout.add_widget(lbltimeleft)
            #train_network(network, dataset, l_rate, n_epoch, n_outputs)
            # for row in dataset:
            #      prediction = predict(network, row)
            #      accuracy_metric(dataset,prediction)
            self.cancellayout()

    def btnErrorpressed(self):
            global progressbar, accuracylabel, scores, n_epoch, l_rate, numdata, valuepb, printth
            global network, n_inputs, n_outputs, n_folds, dataset, list_hidden, trainclock, stoptrainning
            global myevent, timestart, errorlbl
            layoutprogressbar = self.parentlayout.ids.mainseparatorprogressbar
            layoutprogressbar.remove_widget(progressbar)
            layoutprogressbar.add_widget(progressbar)
            progressbar.value = 0
            valuepb = 0
            stoptrainning = False
            timestart = time.monotonic()
            trainclock = Clock.schedule_interval(layoutpbnext, 1 / 10)

            printth = 0
            self.layout.remove_widget(lbltimeleft)
            self.layout.add_widget(lbltimeleft)

            threading.Thread(target=self.evaluate_algorithm, args=(dataset, back_propagation, n_folds, myevent, self, l_rate, n_epoch, list_hidden)).start()
            self.cancellayout()

    def mainlayout_func(self):
        global prediction, n_inputs, n_outputs, dataset, numdata, listinputs, filename
        self.layoutmainscreen.height = 580
        self.btnlayout.height = 50
        self.layoutinfoleft.size_hint_x = 0.1
        self.layoutinforight.size_hint_x = 0.1


        self.layouttrainscreen = self.parentlayout.ids.idLayoutNeuron.ids.idAnchorLayoutNeuron
        self.layouttrainscreen.anchor_x = 'center'

        self.layout.remove_widget(self.layoutsettingsvertical)
        self.layout.remove_widget(self.btnCancel)
        self.layout.remove_widget(self.btnnewinput)
        self.layout.remove_widget(self.btnTrain)
        self.layout.remove_widget(self.btnError)
        self.layout.remove_widget(self.btnPredict)
        self.layout.add_widget(self.btnTrain)
        self.layout.add_widget(self.btnError)
        self.layout.add_widget(self.btnPredict)
        self.layoutinfotop.remove_widget(self.btnSettings)
        self.layoutinfotop.add_widget(self.btnSettings)
        self.layoutinfotop.remove_widget(self.btnBack)
        self.btnSettings.disabled = False
        self.btnBack.disabled = False
        self.layout.remove_widget(self.btnProcessPredict)
        layoutprogressbar = self.parentlayout.ids.mainseparatorprogressbar
        layoutprogressbar.remove_widget(progressbar)
        layoutprogressbar.remove_widget(progressbar)
        layoutprogressbar.add_widget(progressbar)

        if prediction == False:
            if varerrorlayout == False:
                myevent.drawingmainlayout(self)
                self.btnPredict.disabled = True
            else:
                self.btnPredict.disabled = True
        else:
            self.btnPredict.disabled = False


    def predictlayout(self):
        self.layouttrainscreen = self.parentlayout.ids.idLayoutNeuron.ids.idAnchorLayoutNeuron
        self.layouttrainscreen.anchor_x = 'center'
        layout.remove_widget(self.btnTrain)
        layout.remove_widget(self.btnError)
        layout.remove_widget(self.btnPredict)
        layout.remove_widget(self.btnCancel)
        self.layoutinfotop.remove_widget(self.btnSettings)
        self.layoutinfotop.remove_widget(self.btnBack)
        layout.remove_widget(self.btnnewinput)
        self.btnSettings.disabled = False
        self.btnBack.disabled = False

        layout.add_widget(self.btnProcessPredict)
        self.layoutinfotop.add_widget(self.btnBack)
        self.layoutinfotop.add_widget(self.btnSettings)

        MyApp.drawinginputs(self,layoutneuronmainscreen, layoutlabels)

    def predictdonelayout(self):
        global listinputs
        self.layouttrainscreen = self.parentlayout.ids.idLayoutNeuron.ids.idAnchorLayoutNeuron
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
            self.layouttrainscreen.anchor_x = 'center'
            layout.remove_widget(self.btnTrain)
            layout.remove_widget(self.btnError)
            layout.remove_widget(self.btnPredict)
            layout.remove_widget(self.btnCancel)
            self.layoutinfotop.remove_widget(self.btnSettings)
            self.layoutinfotop.remove_widget(self.btnBack)
            layout.remove_widget(self.btnProcessPredict)
            self.btnSettings.disabled = False
            self.btnBack.disabled = False
            self.layoutinfotop.add_widget(self.btnBack)
            self.layoutinfotop.add_widget(self.btnSettings)
            layout.add_widget(self.btnnewinput)
            for row in input:
                prediction = predict(network, row)
                #accuracy_metric(dataset,prediction)
            MyApp.drawingvisualize(self, layoutneuronmainscreen)

    def trainlayout(self):
        self.layouttrainscreen = self.parentlayout.ids.idLayoutNeuron.ids.idAnchorLayoutNeuron
        self.layoutinfoleft.size_hint_x = 0.4
        self.layoutinforight.size_hint_x = 0.0
        self.layout.remove_widget(self.btnTrain)
        self.layout.remove_widget(self.btnError)
        self.layout.remove_widget(self.btnPredict)
        self.layout.remove_widget(self.btnCancel)
        self.layoutinfotop.remove_widget(self.btnSettings)
        self.layoutinfotop.remove_widget(self.btnBack)
        layoutprogressbar = self.parentlayout.ids.mainseparatorprogressbar
        layoutprogressbar.remove_widget(progressbar)
        progressbar.value=0
        self.btnSettings.disabled = False
        self.btnBack.disabled = False

        self.layoutinfotop.add_widget(self.btnBack)
        self.layoutinfotop.add_widget(self.btnSettings)
        mydata = MyApp.Lblleft(type='train', dataerror=0)
        MyApp.addclasstopanel(self, mydata)
        self.drawinglblblackkv(self.layoutinfomiddle, mydata)


    def cancellayout(self):
        self.layout.remove_widget(self.btnTrain)
        self.layout.remove_widget(self.btnError)
        self.layout.remove_widget(self.btnPredict)
        self.layout.remove_widget(self.btnnewinput)
        self.layoutinfotop.remove_widget(self.btnBack)
        self.layoutinfotop.remove_widget(self.btnSettings)
        self.btnSettings.disabled = True
        self.btnBack.disabled = True
        self.layout.add_widget(self.btnCancel)
        self.layoutinfotop.add_widget(self.btnBack)
        self.layoutinfotop.add_widget(self.btnSettings)

    def errorlayout(self, accuracy):
        self.layouttrainscreen = self.parentlayout.ids.idLayoutNeuron.ids.idAnchorLayoutNeuron
        self.layoutinfoleft.size_hint_x = 0.4
        self.layoutinforight.size_hint_x = 0.0
        self.layout.remove_widget(self.btnTrain)
        self.layout.remove_widget(self.btnError)
        self.layout.remove_widget(self.btnPredict)
        self.layout.remove_widget(self.btnCancel)
        self.layoutinfotop.remove_widget(self.btnSettings)
        self.layoutinfotop.remove_widget(self.btnBack)
        layoutprogressbar = self.parentlayout.ids.mainseparatorprogressbar
        layoutprogressbar.remove_widget(progressbar)
        progressbar.value = 0
        self.btnSettings.disabled = False
        self.btnBack.disabled = False

        self.layoutinfotop.add_widget(self.btnBack)
        self.layoutinfotop.add_widget(self.btnSettings)
        mydata = MyApp.Lblleft(type='Find error', dataerror=accuracy)
        self.addclasstopanel(mydata)
        MyApp.leftlabels(self, self.layoutinfomiddle, mydata)

    def btnplusneuronpressed (self):
        self.btnlessneuron.disabled = False
        if list_hidden[0] >= 40:
            list_hidden[0] = 40
        else:
            list_hidden[0] = list_hidden[0] + 1
            self.drawingtryout(layoutneuronmainscreen, layoutlabels)
            self.lblhiddenlayer.text = str(list_hidden[0])
            if list_hidden[0] == 40:
                self.btnplusneuron.disabled = True
    def btnlessneuronpressed(self):
        self.btnplusneuron.disabled = False
        if list_hidden[0] <= 1:
            list_hidden[0] = 1
        else:
            list_hidden[0] = list_hidden[0] - 1
            self.lblhiddenlayer.text = str(list_hidden[0])
            self.drawingtryout(layoutneuronmainscreen, layoutlabels)
            if list_hidden[0] == 1:
                self.btnlessneuron.disabled = True

    def btnplusneurontwopressed(self):
        self.btnlessneurontwo.disabled = False

        if list_hidden[1] >= 40:
            list_hidden[1] = 40
        else:
            list_hidden[1] = list_hidden[1] + 1
            self.lblhiddenlayertwo.text = str(list_hidden[1])
            self.drawingtryout(layoutneuronmainscreen, layoutlabels)
            if list_hidden[1] == 40:
                self.btnplusneurontwo.disabled = True
    def btnlessneurontwopressed(self):
        self.btnplusneurontwo.disabled = False
        if list_hidden[1] <= 1:
            list_hidden[1] = 1
        else:
            list_hidden[1] = list_hidden[1] - 1
            self.lblhiddenlayertwo.text = str(list_hidden[1])
            self.drawingtryout(layoutneuronmainscreen, layoutlabels)
            if list_hidden[1] == 1:
                self.btnlessneurontwo.disabled = True

    def btnplusneuronthreepressed(self):
        self.btnlessneuronthree.disabled = False

        if list_hidden[2] >= 40:
            list_hidden[2] = 40
        else:
            list_hidden[2] = list_hidden[2] + 1
            self.lblhiddenlayerthree.text = str(list_hidden[2])
            self.drawingtryout(layoutneuronmainscreen, layoutlabels)
            if list_hidden[2] == 40:
                self.btnplusneuronthree.disabled = True
    def btnlessneuronthreepressed(self):
        self.btnplusneuronthree.disabled = False
        if list_hidden[2] <= 1:
            list_hidden[2] = 1
        else:
            list_hidden[2] = list_hidden[2] - 1
            self.lblhiddenlayerthree.text = str(list_hidden[2])
            self.drawingtryout(layoutneuronmainscreen, layoutlabels)
            if list_hidden[2] == 1:
                self.btnlessneuronthree.disabled = True

    def btnlayerplusneuronpressed(self):
        self.btnlayerlessneuron.disabled = False
        if len(list_hidden) >= 3:
            pass
        else:
            numlayers = str(len(list_hidden))
            self.lbllayer.text = numlayers
            if len(list_hidden) == 1:
                list_hidden.append(4)
                self.drawingtryout(layoutneuronmainscreen, layoutlabels)

                self.layoutsettingsupleft.add_widget(self.btnlessneurontwo)
                self.layoutsettingsupleft.add_widget(self.neurontwolayout)
                self.layoutsettingsupleft.add_widget(self.btnplusneurontwo)
                self.lbllayer.text = str(len(list_hidden))
                self.lblhiddenlayertwo.text = str(list_hidden[1])
                self.btnlayerplusneuron.disabled = False
            elif len(list_hidden) == 2:
                list_hidden.append(4)
                self.drawingtryout(layoutneuronmainscreen, layoutlabels)
                self.lblhiddenlayerthree.text = str(list_hidden[2])

                self.layoutsettingsupleft.add_widget(self.btnlessneuronthree)
                self.layoutsettingsupleft.add_widget(self.neuronthreelayout)
                self.layoutsettingsupleft.add_widget(self.btnplusneuronthree)
                self.lbllayer.text = str(len(list_hidden))
                self.btnlayerplusneuron.disabled = True
    def btnlayerlessneuronpressed(self):
        self.btnlayerplusneuron.disabled = False
        if len(list_hidden) <= 1:
            pass

        else:
            del list_hidden[-1]
            numlayers = str(len(list_hidden))
            self.lbllayer.text = numlayers
            self.drawingtryout(layoutneuronmainscreen, layoutlabels)
            if len(list_hidden) == 2:
                self.layoutsettingsupleft.remove_widget(self.btnplusneuronthree)
                self.layoutsettingsupleft.remove_widget(self.neuronthreelayout)
                self.layoutsettingsupleft.remove_widget(self.btnlessneuronthree)
                self.btnlayerlessneuron.disabled = False

            elif len(list_hidden) == 1:
                self.layoutsettingsupleft.remove_widget(self.btnplusneurontwo)
                self.layoutsettingsupleft.remove_widget(self.neurontwolayout)
                self.layoutsettingsupleft.remove_widget(self.btnlessneurontwo)

                self.btnlayerlessneuron.disabled = True

    def btnplusfoldspressed (self):
        global n_folds
        self.btnlessfolds.disabled = False
        if n_folds >= 10:
            n_folds = 10
        else:
            n_folds += 1
            self.numfolds.text = str(n_folds)
            if n_folds == 10:
                self.btnplusfolds.disabled = True
    def btnlessfoldspressed(self):
        self.btnplusfolds.disabled = False
        global n_folds
        if n_folds <= 1:
            n_folds = 1
        else:
            n_folds = n_folds - 1
            self.numfolds.text = str(n_folds)
            if n_folds == 1:
                self.btnlessfolds.disabled = True

    def btnplusratepressed (self):
        global l_rate
        self.btnlessrate.disabled = False
        if l_rate >= 1.0:
            l_rate = 1.0
        else:
            l_rate = l_rate + 0.1
            self.numrate.text = str("{:.1f}".format(l_rate))
            if l_rate == 1.0:
                self.btnplusrate.disabled = True
    def btnlessratepressed(self):
        global l_rate
        self.btnplusrate.disabled = False
        if l_rate <= 0.1:
            l_rate = 0.1
        else:
            l_rate = l_rate - 0.1
            self.numrate.text = str("{:.1f}".format(l_rate))
            if l_rate == 0.1:
                self.btnlessrate.disabled = True

    def btnplusepochpressed (self):
        global n_epoch
        self.btnlessepoch.disabled = False
        if n_epoch >= 500:
            n_epoch = 500
        else:
            n_epoch += 10
            self.numepoch.text = str(n_epoch)
            if n_epoch == 500:
                self.btnplusepoch.disabled = True

    def filechooserfunction(self):
        global popupfilechooser
        content = Filechooser()
        popupfilechooser = Popup(title="Load file", content=content,
                            size_hint=(0.9, 0.9),auto_dismiss= False, separator_color= (.67,.57,.69))
        popupfilechooser.open()

    def btncancelpressed(self):
        global myevent
        myevent.trainnetwork('')


    def evaluate_algorithm(self, dataset, algorithm, n_folds, eventp, myapp, *args):
        global progressbar
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
            predicted = algorithm(train_set, test_set, eventp, myapp, *args)
            actual = [row[-1] for row in fold]
            accuracy = accuracy_metric(actual, predicted)
            scores.append(accuracy)
        eventp.evevaluatealgorithm(myapp)

    def btnlessepochpressed(self):
        global n_epoch
        self.btnplusepoch.disabled = False
        if n_epoch <= 10:
            n_epoch = 10
        else:
            n_epoch = n_epoch - 10
            self.numepoch.text = str(n_epoch)
            if n_epoch == 10:
                self.btnlessepoch.disabled = True

    def build(self):
        global layoutneuronmainscreen, layoutlabels, lblfilename, currentpage
        global myevent
        class MyEventDispatcher(EventDispatcher):
            global lbltimeleft
            def __init__(self, **kwargs):
                self.register_event_type('on_test')
                self.register_event_type('on_trainfinished')
                self.register_event_type('on_drawingmainlayout')
                self.register_event_type('on_evreaddataset')
                self.register_event_type('on_evshowfile')
                super(MyEventDispatcher, self).__init__(**kwargs)
            @mainthread
            def evevaluatealgorithm(self, myapp):
                self.dispatch('on_test', myapp)

            def on_test(self, myapp):
                global trainclock
                trainclock.cancel()
                layoutprogressbar = myapp.parentlayout.ids.mainseparatorprogressbar
                layoutprogressbar.remove_widget(progressbar)
                progressbar.value = 0
                print('Scores: %s' % scores)
                print('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))
                accuracylabel = (sum(scores) / float(len(scores)))
                myapp.errorlayout(accuracylabel)
                myapp.layout.remove_widget(lbltimeleft)

            @mainthread
            def trainnetwork(self, myapp):
                global stoptrainning
                stoptrainning = True
                self.dispatch('on_trainfinished', myapp)

            def on_trainfinished(self, myapp):
                global trainclock
                trainclock.cancel()
                myapp.layoutprogressbar = myapp.parentlayout.ids.mainseparatorprogressbar
                myapp.layoutprogressbar.remove_widget(progressbar)
                myapp.layout.remove_widget(lbltimeleft)
                progressbar.value = 0
                myapp.trainlayout()
                for layer in network:
                    print(layer)

            @mainthread
            def drawingmainlayout(self, myapp):
                self.dispatch('on_drawingmainlayout', myapp)

            def on_drawingmainlayout(self, myapp):
                #MyApp.drawingblank(btnOk, layoutneuronmainscreen, layoutlabels)
                global filename

                threading.Thread(target=readdataset, args=(filename, myevent, myapp, None, None)).start()



            @mainthread
            def evreaddataset(self, progressvarvalue, myisimage, mydataset, myapp):
                self.dispatch('on_evreaddataset', progressvarvalue, myisimage, mydataset, myapp)

            def on_evreaddataset(self, progressvarvalue, myisimage, mydataset, myapp):
                global dataset, network, n_inputs, n_outputs
                progressbar.value = progressvarvalue
                if progressvarvalue == 100:
                    progressbar.value = 100.0
                    dataset = mydataset
                    n_inputs = len(dataset[0]) - 1
                    n_outputs = len(set([row[-1] for row in dataset]))
                    network = initialize_network_no_weights(n_inputs, list_hidden, n_outputs)
                    myapp.drawinglblblackkv(myapp.layoutinfomiddle, MyApp.Lblleft(type='train', dataerror=0))
                    myapp.layoutprogressbar = myapp.parentlayout.ids.mainseparatorprogressbar
                    myapp.layoutprogressbar.remove_widget(progressbar)
                    myapp.layout.remove_widget(lbltimeleft)
                    myapp.btnTrain.disabled = False
                    myapp.btnError.disabled = False
                    myapp.btnPredict.disabled = False
            @mainthread
            def evshowfile(self, filechooser, filenameselected):
                self.dispatch('on_evshowfile', filechooser, filenameselected)

            def on_evshowfile(self, filechooser, filenameselected):
                try:
                    Filechooser.filechoosershowdatasetcont(filechooser, filenameselected)
                except Exception as e:  # work on python 3.x
                    print('filechoosershowdatasetcont: ' + str(e))


        list_hidden.append(6)
        numlayers = str(len(list_hidden))
        myevent = MyEventDispatcher()

        self.parentlayout = LayoutScreen()
        layoutneuronmainscreen = self.parentlayout.ids.idLayoutNeuron.ids.idLayoutNeuronMainScreen.ids.idrelativelayoutneuronmainscreen
        layoutlabels = self.parentlayout.ids.idLayoutNeuron.ids.idLayoutNeuronMainScreen.ids.idLabelsLayoutNeuronMainScreen
        self.layoutmainscreen = self.parentlayout.ids.idLayoutNeuron.ids.idLayoutNeuronMainScreen
        self.layout = self.parentlayout.ids.idLayoutMenu.ids.idBoxLayoutMenu
        self.btnlayout = self.parentlayout.ids.idLayoutMenu
        self.layoutinfoleft = self.parentlayout.ids.idboxlayoutleft
        self.layoutinforight = self.parentlayout.ids.idboxlayoutright

        self.hiddenlayeronelabelone = Label(text='[b]  Hidden\nlayer one', markup=True, font_size=15, color=(1, 1, 1),
                                       size_hint=(None, None))
        self.hiddenlayeronelabeltwo = Label(text='[b]  Hidden\nlayer two', markup=True, size_hint=(None, None), font_size=15,
                                       color=(1, 1, 1))
        self.hiddenlayeronelabelthree = Label(text='[b]   Hidden\nlayer three', markup=True, size_hint=(None, None),
                                         font_size=15, color=(1, 1, 1))
        self.outputlabel = Label(text='Output', markup=True, font_size=17, size_hint=(None, None), color=(1, 1, 1))
        self.inputlabel = Label(text='Input', markup=True, font_size=17, size_hint=(None, None), color=(1, 1, 1))
        self.lblcurrentpage = Label(text=('[b]Page:[/b]' + ' ' + str("{:.0f}".format(currentpage)) + '/' + '0'), markup=True)
        lblfilename = ''
        self.btnplussidetext = Button(text='[b]>', markup=True, size_hint=(.2, .8),
                                 pos_hint={'center_x': .5, 'center_y': .5},
                                 background_normal='ButtonPlusP.png',
                                 background_down='ButtonDeactivated.png',
                                 background_disabled_normal='ButtonDeactivated.png', font_size=20)
        self.btnlesssidetext = Button(text='[b]<', markup=True, size_hint=(.2, .8),
                                 pos_hint={'center_x': 1, 'center_y': .5},
                                 background_normal='ButtonPlusP.png',
                                 background_down='ButtonDeactivated.png',
                                 background_disabled_normal='ButtonDeactivated.png', font_size=20)

        neuronlayout = BoxLayout(orientation='vertical', size_hint=(.3,1))
        self.neurontwolayout = BoxLayout(orientation='vertical', size_hint=(.3,1))
        self.neuronthreelayout = BoxLayout(orientation='vertical', size_hint=(.3,1))
        numlayerslayout = BoxLayout(orientation='vertical', size_hint=(.3,1))
        epochlayout = BoxLayout(orientation ='vertical', pos_hint={'top':1}, size_hint=(.3, .8))
        ratelayout = BoxLayout(orientation ='vertical', pos_hint={'top':1}, size_hint=(.3, .8))
        foldslayout = BoxLayout(orientation ='vertical', pos_hint={'top':1}, size_hint=(.3, .8))
        self.layoutsettingsvertical = BoxLayout(orientation ='vertical')
        layoutsettingsup = BoxLayout(orientation='horizontal', spacing = 10)
        layoutsettingsdown = BoxLayout(orientation='horizontal',spacing = 10)
        self.layoutsettingsupleft = BoxLayout(orientation='horizontal',spacing = 10, size_hint=(.7,1))
        layoutsettingsupright = BoxLayout(orientation='horizontal',spacing = 10, size_hint=(.3,1))
        self.layoutinfotop = self.parentlayout.ids.idlayouttop
        self.layoutinfomiddle = self.parentlayout.ids.idlayoutmiddle
        self.layoutinfodown = self.parentlayout.ids.idlayoutdown
        self.layoutbtninfo = self.parentlayout.ids.idlayoutlbldown
        layoutfilenameright = self.parentlayout.ids.idboxlayoutright
        lblfilename = self.parentlayout.ids.idlabelfilename


        self.btnPredict = Button(color=(1, 1, 1, 1), text='[b]PREDICT', markup = True,background_disabled_normal='ButtonDeactivated.png', font_size=18, size_hint=(.95, .9), pos_hint={'center_x':.5, 'center_y':.5}, background_normal='ButtonPlusP.png', background_down='ButtonDeactivated.png')
        self.btnProcessPredict = Button(color=(1, 1, 1, 1), text='[b]PREDICT', markup = True,background_disabled_normal='ButtonDeactivated.png', font_size=18, size_hint=(.95, .9), pos_hint={'center_x':.5, 'center_y':.5}, background_normal='ButtonPlusP.png', background_down='ButtonDeactivated.png')
        self.btnTrain = Button(text='[b]TRAIN',markup = True, background_normal='ButtonPlusP.png', background_down='ButtonDeactivated.png', size_hint=(.95,.9), background_disabled_normal='ButtonDeactivated.png',
                       pos_hint={'center_x':.5, 'center_y':.5}, color = (1,1,1,1), font_size=18)
        self.btnError = Button(text='[b]ERROR', markup=True, background_normal='ButtonPlusP.png',
                          background_down='ButtonDeactivated.png', size_hint=(.95, .9),
                          pos_hint={'center_x': .5, 'center_y': .5},background_disabled_normal='ButtonDeactivated.png', color=(1, 1, 1, 1), font_size=18)
        self.btnOk = Button(text='[b]OK', markup=True, background_normal='ButtonPlusB.png',
                          background_down='ButtonDeactivated.png', size_hint=(.8, .8),
                          pos_hint={'center_x': .5, 'center_y': .5}, color=(0, 0, 0, 1), font_size=18)
        self.btnOpen = Button(text='[b]OPEN', markup=True, background_normal='ButtonPlusP.png',
                       background_down='ButtonPlusGDown.png', size_hint=(.8, .8),
                       pos_hint={'center_x': .5, 'center_y': .5}, color=(1, 1, 1, 1), font_size=18)
        self.btnBack = Button(text='[b]', markup=True, background_normal='ButtonBack.png',
                       background_down='ButtonBack.png', size_hint=(None, None), size=(60,60),
                       pos_hint={'center_x': 1, 'center_y': .5}, color=(1, 1, 1, 1),background_disabled_normal='ButtonBackDisabled.png')
        self.btnSettings = Button(background_normal='ButtonSettings.png', background_down='ButtonSettings.png', size_hint=(None, None), size=(60,60), pos_hint={'center_x':1, 'center_y':.5},background_disabled_normal='ButtonSettingsDisabled.png')
        self.btnnewinput=Button(color=(1,1,1,1), text='[b]New Input',markup = True, font_size=18, size_hint=(.95,.9),
                       pos_hint={'center_x':.5, 'center_y':.5},background_normal='ButtonPlusP.png', background_down='ButtonDeactivated.png')
        self.btnCancel = Button(color=(1,1,1,1), text='[b]CANCEL',markup = True, font_size=18, size_hint=(.95,.9), pos_hint={'center_x':.5, 'center_y':.5},background_normal='ButtonPlusP.png', background_down='ButtonDeactivated.png')



        box = CheckBox()
        lblbox = Label(text='Sigmoid:', color=(1, 1, 1, 1), font_size=(14))
        boxshowbiaslabel = CheckBox()

        # Left Neuron Switcher
        self.btnplusneuron = Button(text = '[b]+',markup = True, size_hint=(.2,.8),pos_hint={'center_x':.5, 'center_y':.5},
                               background_normal='ButtonPlusP.png',
                               background_down='ButtonDeactivated.png',
                               background_disabled_normal='ButtonDeactivated.png', font_size=20)
        self.btnlessneuron = Button(text = '[b]-',markup = True,size_hint=(.2, .8),pos_hint={'center_x':.5, 'center_y':.5},
                               background_normal='ButtonPlusP.png',
                               background_down='ButtonDeactivated.png',
                               background_disabled_normal='ButtonDeactivated.png', font_size=20)
        self.lblhiddenlayer = Label(text=str(list_hidden[0]))
        self.lblneuron = Label(text='[b]Neurons', markup=True)

        # Second neuron switcher
        self.btnplusneurontwo = Button(text = '[b]+',markup = True,size_hint=(.2, .8),pos_hint={'center_x':.5, 'center_y':.5},
                                  background_normal='ButtonPlusP.png',
                                  background_down='ButtonDeactivated.png',
                                  background_disabled_normal='ButtonDeactivated.png', font_size=20)
        self.btnlessneurontwo = Button(text = '[b]-',markup = True,size_hint=(.2, .8), pos_hint={'center_x':.5, 'center_y':.5},
                                  background_normal='ButtonPlusP.png',
                                  background_down='ButtonDeactivated.png',
                                  background_disabled_normal='ButtonDeactivated.png', font_size=20)
        self.lblhiddenlayertwo = Label(text='4')
        self.lblneurontwo = Label(text='[b]Neurons', markup=True)

        # Right/Third neuron switcher
        self.btnplusneuronthree = Button(text = '[b]+',markup = True,size_hint=(.2, .8),pos_hint= {'center_x':1, 'center_y':.5},
                                    background_normal='ButtonPlusP.png',
                                    background_down='ButtonDeactivated.png',
                                    background_disabled_normal='ButtonDeactivated.png', font_size=20)
        self.btnlessneuronthree = Button(text = '[b]-',markup = True,size_hint=(.2, .8),pos_hint={'center_x':1, 'center_y':.5},
                                    background_normal='ButtonPlusP.png',
                                    background_down='ButtonDeactivated.png',
                                    background_disabled_normal='ButtonDeactivated.png', font_size=20)
        self.lblhiddenlayerthree = Label(text='4')
        self.lblneuronthree = Label(text='[b]Neurons', markup=True)

        # Number of layers switcher
        self.btnlayerplusneuron = Button(text = '[b]+',markup = True,size_hint=(.2, .8),pos_hint={'center_x':1, 'center_y':.5},
                                    background_normal='ButtonPlusP.png',
                                    background_down='ButtonDeactivated.png',
                                    background_disabled_normal='ButtonDeactivated.png', font_size=20)
        self.btnlayerlessneuron = Button(text = '[b]-',markup = True,size_hint=(.2, .8),pos_hint={'center_x':1, 'center_y':.5},
                                    background_normal='ButtonPlusP.png',
                                    background_down='ButtonDeactivated.png',
                                    background_disabled_normal='ButtonDeactivated.png', font_size=20)
        self.lbllayer = Label(text=numlayers)
        self.lblnumlayers = Label(text='[b] Hiddenlayers',markup = True, font_size=(11))
        lblblack = Widget(size_hint=(.1,.1))
        lblblacktwo = Widget(size_hint=(.1,.1))
        lblblackthree = Widget(size_hint=(.1,.1))
        canvas= Widget(size_hint=(.02,.1))
        visualize = Label(size_hint=(.1,.1))

        self.btnplusepoch = Button(text='[b]+', markup=True, size_hint=(.2, .8),
                              pos_hint={'center_x': 1, 'center_y': .5},
                              background_normal='ButtonPlusP.png',
                              background_down='ButtonDeactivated.png',
                              background_disabled_normal='ButtonDeactivated.png', font_size=20)
        self.btnlessepoch = Button(text='[b]-', markup=True, size_hint=(.2, .8),
                                    pos_hint={'center_x': 1, 'center_y': .5},
                                    background_normal='ButtonPlusP.png',
                                    background_down='ButtonDeactivated.png',
                                    background_disabled_normal='ButtonDeactivated.png', font_size=20)
        self.lblepoch = Label(text='[b] CYCLES:',markup=True, font_size=(11), pos_hint={'center_x':.5, 'center_y':.5})
        self.numepoch = Label(text=str(n_epoch))
        self.btnplusrate = Button(text='[b]+', markup=True, size_hint=(.2, .8),
                              pos_hint={'center_x': 1, 'center_y': .5},
                              background_normal='ButtonPlusP.png',
                              background_down='ButtonDeactivated.png',
                              background_disabled_normal='ButtonDeactivated.png', font_size=20)
        self.btnlessrate = Button(text='[b]-', markup=True, size_hint=(.2, .8),
                              pos_hint={'center_x': 1, 'center_y': .5},
                              background_normal='ButtonPlusP.png',
                              background_down='ButtonDeactivated.png',
                              background_disabled_normal='ButtonDeactivated.png', font_size=20)
        lblrate = Label(text='[b]RATE:',markup=True, font_size=(11), pos_hint={'center_x': .5, 'center_y': .5})
        self.numrate = Label(text=str(l_rate))
        self.btnplusfolds = Button(text='[b]+', markup=True, size_hint=(.2, .8),
                             pos_hint={'center_x': 1, 'center_y': .5},
                             background_normal='ButtonPlusP.png',
                             background_down='ButtonDeactivated.png',
                             background_disabled_normal='ButtonDeactivated.png', font_size=20)
        self.btnlessfolds = Button(text='[b]-', markup=True, size_hint=(.2, .8),
                             pos_hint={'center_x': 1, 'center_y': .5},
                             background_normal='ButtonPlusP.png',
                             background_down='ButtonDeactivated.png',
                             background_disabled_normal='ButtonDeactivated.png', font_size=20)
        lblfolds = Label(text='[b]FOLDS:',markup=True, font_size=(11), pos_hint={'center_x': .5, 'center_y': .5})
        self.numfolds = Label(text=str(n_folds))

        MyApp.restoreleftpanellist(self)

        self.btnplussidetext.disabled = True
        self.btnlesssidetext.disabled = True

        layoutsettingsup.add_widget(self.layoutsettingsupleft)
        layoutsettingsup.add_widget(lblblack)
        layoutsettingsup.add_widget(layoutsettingsupright)
        neuronlayout.add_widget(self.lblneuron)
        neuronlayout.add_widget(self.lblhiddenlayer)
        self.layoutsettingsupleft.add_widget(lblblacktwo)
        self.layoutsettingsupleft.add_widget(self.btnlessneuron)
        self.layoutsettingsupleft.add_widget(neuronlayout)
        self.layoutsettingsupleft.add_widget(self.btnplusneuron)
        layoutsettingsupright.add_widget(self.btnlayerlessneuron)
        layoutsettingsupright.add_widget(numlayerslayout)
        layoutsettingsupright.add_widget(self.btnlayerplusneuron)
        layoutsettingsupright.add_widget(visualize)
        layoutsettingsdown.add_widget(lblblackthree)
        layoutsettingsdown.add_widget(self.btnOk)
        layoutsettingsdown.add_widget(self.btnOpen)
        layoutsettingsdown.add_widget(self.btnlessepoch)
        layoutsettingsdown.add_widget(epochlayout)
        layoutsettingsdown.add_widget(self.btnplusepoch)
        layoutsettingsdown.add_widget(self.btnlessrate)
        layoutsettingsdown.add_widget(ratelayout)
        layoutsettingsdown.add_widget(self.btnplusrate)
        layoutsettingsdown.add_widget(self.btnlessfolds)
        layoutsettingsdown.add_widget(foldslayout)
        layoutsettingsdown.add_widget(self.btnplusfolds)
        numlayerslayout.add_widget(self.lblnumlayers)
        numlayerslayout.add_widget(self.lbllayer)
        self.btnlayerlessneuron.disabled = True
        epochlayout.add_widget(self.lblepoch)
        epochlayout.add_widget(self.numepoch)
        ratelayout.add_widget(lblrate)
        ratelayout.add_widget(self.numrate)
        foldslayout.add_widget(lblfolds)
        foldslayout.add_widget(self.numfolds)
        self.layoutsettingsvertical.add_widget(layoutsettingsup)
        self.layoutsettingsvertical.add_widget(layoutsettingsdown)
        self.layoutbtninfo.add_widget(self.btnlesssidetext)
        self.layoutbtninfo.add_widget(self.btnplussidetext)
        self.layoutinfodown.add_widget(Label(
            text='-----------------------------------------------------------------------------------------------------------------------------------------------',
            pos_hint={'center_x': .5, 'center_y': .5}, size_hint=(1, .1), color=(.6, .87, .89, 1)))
        self.layoutinfodown.add_widget(self.lblcurrentpage)


        self.neurontwolayout.add_widget(self.lblneurontwo)
        self.neurontwolayout.add_widget(self.lblhiddenlayertwo)
        self.neuronthreelayout.add_widget(self.lblneuronthree)
        self.neuronthreelayout.add_widget(self.lblhiddenlayerthree)

        try:
            MyApp.restoreleftpanellist(self)
            mydata = leftpanellist[-1]
            currentpage = len(leftpanellist)
            self.drawinglblblackkv(self.layoutinfomiddle, mydata)
            self.establishbtnpages()
        except:
            pass


        self.settingslayout()


        self.btnTrain.bind(on_press=lambda x:self.btnTrainpressed())
        self.btnPredict.bind(on_press=lambda x:self.predictlayout())
        self.btnProcessPredict.bind(on_press=lambda x:self.predictdonelayout())
        self.btnnewinput.bind(on_press=lambda x:self.predictlayout())
        self.btnplusneuron.bind(on_press=lambda x:self.btnplusneuronpressed())
        self.btnlessneuron.bind(on_press=lambda x:self.btnlessneuronpressed())
        self.btnplusneurontwo.bind(on_press=lambda x:self.btnplusneurontwopressed())
        self.btnlessneurontwo.bind(on_press=lambda x:self.btnlessneurontwopressed())
        self.btnplusneuronthree.bind(on_press=lambda x:self.btnplusneuronthreepressed())
        self.btnlessneuronthree.bind(on_press=lambda x:self.btnlessneuronthreepressed())
        self.btnlayerplusneuron.bind(on_press=lambda x:self.btnlayerplusneuronpressed())
        self.btnlayerlessneuron.bind(on_press=lambda x:self.btnlayerlessneuronpressed())
        self.btnSettings.bind(on_press=lambda x:self.settingslayout())
        self.btnCancel.bind(on_press=lambda x:self.btncancelpressed())
        self.btnOk.bind(on_press=lambda x:self.mainlayout_func())
        self.btnBack.bind(on_press=lambda x:self.mainlayout_func())
        self.btnError.bind(on_press=lambda x:self.btnErrorpressed())
        self.btnplusfolds.bind(on_press=lambda x:self.btnplusfoldspressed())
        self.btnlessfolds.bind(on_press=lambda x:self.btnlessfoldspressed())
        self.btnplusepoch.bind(on_press=lambda x:self.btnplusepochpressed())
        self.btnlessepoch.bind(on_press=lambda x:self.btnlessepochpressed())
        self.btnplusrate.bind(on_press=lambda x:self.btnplusratepressed())
        self.btnlessrate.bind(on_press=lambda x:self.btnlessratepressed())
        self.btnOpen.bind(on_press=lambda x:self.filechooserfunction())
        self.btnplussidetext.bind(on_press=lambda x:self.leftlabelswitch(self.layoutinfomiddle, 0, 'btnplus'))
        self.btnlesssidetext.bind(on_press=lambda x:self.leftlabelswitch(self.layoutinfomiddle, 0, 'btnless'))


        return self.parentlayout

        




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

class PopupLoading(Label):
    anglef = NumericProperty(40)
    angleb = NumericProperty(0)

    def start(self):
        Animation.cancel_all(self)
        self.anglef = 40
        self.angleb = 0
        self.anim = Animation(angleb=360, anglef=360, duration=1)
        self.anim.bind(on_complete=self.repeat)
        self.anim.start(self)

    def repeat(self, animation, incr_crude_clock):
        self.start()

class Filechooser(BoxLayout):

    def myselected(self, path, filenameselected):
        self.ids.text_input.text = basename(filenameselected)

    class WidgetTexture(Widget):
        def __init__(self, imagebits, **kwargs):
            super(Filechooser.WidgetTexture, self).__init__(**kwargs)

            sidesquare = sqrt(len(imagebits) - 1)
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


            with self.canvas:
                Color(1,1,1)
                self.rect = Rectangle(texture=self.texture, pos=self.pos, size=(sidesquare, sidesquare))

            self.bind(pos=self.update_rect,
                      size=self.update_rect)

        def update_rect(self, *args):
            self.rect.pos = (self.center_x - self.rect.size[0] / 2, self.center_y - self.rect.size[1] / 2)

    def filechoosershowdatasetcont(self, filenameselected):

        if self.mytexturedataset == None:
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
            self.popup.dismiss()
            self.pagenum = 1
            self.pags = 0
            containerbox = BoxLayout(orientation='vertical', spacing = 10)
            popupshow = Popup(title='DATASET:', content=containerbox, separator_color=(.6,1,1, 1),
                              auto_dismiss=False)
            popupbtn = Button(text='OK', on_press=popupshow.dismiss, background_normal='ButtonPlusP.png',
                              background_down='ButtonDeactivated.png', background_disabled_normal='ButtonDeactivated.png',
                              pos_hint={'center_x': .5, 'center_y': .5}, size_hint=(.4,1))
            self.mainlayout = GridLayout(cols=1)
            middlelayout = BoxLayout(orientation = 'horizontal')
            self.btnnextpage = Button(text='[b]>', markup=True, size_hint=(.05, .2),
                             pos_hint={'center_x': 1, 'center_y': .5},
                             background_normal='ButtonPlusP.png',
                             background_down='ButtonDeactivated.png',
                             background_disabled_normal='ButtonDeactivated.png', font_size=20)
            self.btnpreviouspage = Button(text='[b]<', markup=True, size_hint=(.05, .2),
                                 pos_hint={'center_x': 1, 'center_y': .5},
                                 background_normal='ButtonPlusP.png',
                                 background_down='ButtonDeactivated.png',
                                 background_disabled_normal='ButtonDeactivated.png', font_size=20)
            mybtnlayout = BoxLayout(orientation='horizontal', size_hint=(1,.1))
            typelayout = GridLayout( cols = 4)
            infolayout = BoxLayout(orientation='horizontal', size_hint=(1,.1))
            self.imagedatabox = CheckBox(group = 'typeobject', size_hint=(.4,1))
            lblimagedatabox = Label(text='Images dataset',halign="left")
            lblimagedatabox.bind(size=lblimagedatabox.setter('text_size'))
            self.numberdatabox = CheckBox(group = 'typeobject', size_hint=(.4,1))
            lblnumberdatabox = Label(text='Numbers dataset',halign="left")
            lblnumberdatabox.bind(size=lblnumberdatabox.setter('text_size'))
            self.outputfirstbox = CheckBox(group='output', size_hint=(.4,1))
            lbloutputfirstbox = Label(text='Outputs on first column',halign="left")
            lbloutputfirstbox.bind(size=lbloutputfirstbox.setter('text_size'))
            self.outputlastbox = CheckBox(group='output', size_hint=(.4,1))
            lbloutputlastbox = Label(text='Outputs on last column',halign="left")
            lbloutputlastbox.bind(size=lbloutputlastbox.setter('text_size'))

            self.lblpage = Label(text='', markup = True, pos_hint={'center_x': .5, 'center_y': .5},
                            size_hint=(1, .1), color=(.7,.9,.9))
            outputslast = ([row[-1] for row in self.mytexturedataset])
            setoutputslast = set(outputslast)

            self.btnnoutputs = Button(text='[b]nº outputs: [/b]' + str(len(setoutputslast)),markup = True, pos_hint={'center_x': .5, 'center_y': .5},
                            size_hint=(1, .9), background_normal='ButtonPlusP.png',
                             background_down='ButtonDeactivated.png')
            lblnelements = Label(text='[b]nº elements: [/b]' + str(len(outputslast)), markup = True, pos_hint={'center_x': .5, 'center_y': .5},
                            size_hint=(.8, .1))

            self.lblelementsize = Label(text='', markup = True, pos_hint={'center_x': .5, 'center_y': .5},
                            size_hint=(.8, .1))




            self.outputlastbox.active = True
            self.outputchanged = False
            self.btnpreviouspage.disabled = True
            self.btnnextpage.disabled = False
            self.btnnextpage.bind(on_press=lambda x:self.numberdatasetfunc('plusbtn'))
            self.btnpreviouspage.bind(on_press=lambda x:self.numberdatasetfunc('lessbtn'))
            self.outputfirstbox.bind(on_press=lambda x:self.outputdatasetfunc())
            self.numberdatabox.bind(active=lambda x,y:self.resetandnumberdatasetfunc(''))
            self.btnnoutputs.bind(on_press=lambda x:self.outputoptionsfunc())

            if self.optionalimage == True:
                typelayout.add_widget(self.numberdatabox)
                typelayout.add_widget(lblnumberdatabox)
            if self.optionaloutput == True:
                typelayout.add_widget(self.outputlastbox)
                typelayout.add_widget(lbloutputlastbox)
            if self.optionalimage == True:
                typelayout.add_widget(self.imagedatabox)
                typelayout.add_widget(lblimagedatabox)
            if self.optionaloutput == True:
                typelayout.add_widget(outputfirstbox)
                typelayout.add_widget(lbloutputfirstbox)
            mybtnlayout.add_widget(popupbtn)
            mybtnlayout.add_widget(typelayout)
            middlelayout.add_widget(self.btnpreviouspage)
            middlelayout.add_widget(self.mainlayout)
            middlelayout.add_widget(self.btnnextpage)
            containerbox.add_widget(middlelayout)
            containerbox.add_widget(infolayout)
            infolayout.add_widget(self.btnnoutputs)
            infolayout.add_widget(lblnelements)
            infolayout.add_widget(self.lblelementsize)
            infolayout.add_widget(self.lblpage)
            containerbox.add_widget(mybtnlayout)
            popupshow.open()

            if self.isimage == True:
                self.imagedatabox.active = True
            else:
                self.numberdatabox.active = True
            self.numberdatasetfunc('')

    def filechoosershowdataset(self, path, filenameselected, eventp):
        global myevent, filename
        filename = os.path.join(path, filenameselected)
        lblfilename.text = ('[b] Current file: [/b]' + '[i]' + filenameselected)
        self.optionaloutput, self.optionalimage, self.isimage, self.mytexturedataset = showdataset(filename)
        eventp.evshowfile(self, filenameselected)


    def numberdatasetfunc(self, plusless):
        global outputimageslist

        self.mainlayout.clear_widgets()
        if plusless == 'plusbtn':
            self.pagenum += 1
        elif plusless == 'lessbtn':
            self.pagenum -= 1

        self.btnactivation()

        #Show number
        if self.numberdatabox.active == True:
            nitemsperpage = 14
            pagsfloat = (len(self.mytexturedataset) / nitemsperpage)
            pagsaux = float(int(pagsfloat))
            self.pags = int(pagsfloat)
            if pagsaux != pagsfloat:
                self.pags += 1
            self.lblelementsize.text = '[b]element size: [/b]' + str(len(self.mytexturedataset[0]))
            self.lblpage.text = '[b]Page: [/b]' + str(self.pagenum) + '/' + str(self.pags)

            if len(self.mytexturedataset[0]) > 20:
                self.mainlayout.cols = 11
                linesperpage = 14
                for rowtexture in range(linesperpage):
                    index = rowtexture + (self.pagenum - 1) * linesperpage
                    pos = 0
                    if index >= len(self.mytexturedataset):
                        break
                    firstline = self.mytexturedataset[rowtexture + (self.pagenum-1) * linesperpage]
                    for number in firstline:
                        if pos < 5 or pos >= (len(firstline) - 5):
                            datalbl = Label(text=str(number))
                            self.mainlayout.add_widget(datalbl)
                        elif pos == 5 :
                            datalbl = Label(text='...')
                            self.mainlayout.add_widget(datalbl)
                        pos += 1
            else:
                self.mainlayout.cols = len(self.mytexturedataset[0])
                for rowtexture in range(14):
                    pos = rowtexture+(self.pagenum-1)*14
                    if pos >= len(self.mytexturedataset):
                        break
                    for number in self.mytexturedataset[pos]:
                        mystrrow = number + ' '
                        datalbl = Label(text =mystrrow)
                        self.mainlayout.add_widget(datalbl)


        #Show Images
        else:
            points = sqrt(len(self.mytexturedataset[0]) - 1)
            cols = int((Window.width - 200) / points)
            self.mainlayout.cols = int(cols)
            rows = int((Window.height - 200) / points)
            self.mainlayout.rows = int (rows)
            nitemsperpage = (self.mainlayout.cols * self.mainlayout.rows)
            pagsfloat = (len(self.mytexturedataset) / nitemsperpage)
            pagsaux = float(int(pagsfloat))
            self.pags = int(pagsfloat)
            self.lblelementsize.text = '[b]image size: [/b]' + str(int(points)) + ' x ' + str(int(points))
            if pagsaux != pagsfloat:
                self.pags += 1

            self.lblpage.text = '[b]Page: [/b]' + str(self.pagenum) + '/' + str(self.pags)
            for rowtexture in range(nitemsperpage): #in range cols * rows
                index = rowtexture+(self.pagenum-1)*nitemsperpage
                if index >= len(self.mytexturedataset):
                    break
                imagedata = self.WidgetTexture(self.mytexturedataset[index])
                self.mainlayout.add_widget(imagedata)
            outputslast = ([row[-1] for row in self.mytexturedataset])
            setoutputslast = sorted(set(outputslast))
            self.myoutputimageslist = list()
            for number in setoutputslast:
                for row in self.mytexturedataset:
                    if row[-1] == number:
                        self.myoutputimageslist.append(row)
                        break
            outputimageslist = self.myoutputimageslist.copy()
        self.btnactivation()

    def resetandnumberdatasetfunc(self, plusless):
        self.pagenum = 1
        self.numberdatasetfunc(plusless)

    def outputdatasetfunc(self):

        if self.outputfirstbox.active == True and self.outputchanged == False:
            self.optionaloutput, self.optionalimage, self.isimage, self.mytexturedataset = showdataset(filenameselected)
            self.numberdatasetfunc('')
            self.outputchanged = True
        elif self.outputlastbox.active == True and self.outputchanged == True:
            self.mytexturedataset = datasetoutputlastcolumn(self.mytexturedataset)
            self.numberdatasetfunc('')
            self.outputchanged = False
        else:
            pass

    def outputoptionsfunc(self):
        containeroutput = BoxLayout(orientation='vertical')
        outputbox = GridLayout()
        outputbox.clear_widgets()
        self.popupoutputs = Popup(title='OUTPUTS:', content=containeroutput, separator_color=(.6,1,1, 1),
                      auto_dismiss=False, size_hint=(.6,.8))
        popupoutputbtn = Button(text='OK', background_normal='ButtonPlusP.png',
                                background_down='ButtonDeactivated.png',
                                background_disabled_normal='ButtonDeactivated.png', pos_hint={'center_x': .5, 'center_y': .5}, size_hint=(1, .2))
        popupoutputbtn.bind(on_press= lambda x:self.popupoutputsdismiss() )
        outputslast = ([row[-1] for row in self.mytexturedataset])
        setoutputslast = sorted(set(outputslast))

        if self.imagedatabox.active == True:
            outputbox.cols = len(setoutputslast)
        else:
            outputbox.rows = len(setoutputslast)

        if self.imagedatabox.active == True:
            # for number in setoutputslast:
            #     for row in self.mytexturedataset:
            #         if row[-1] == number:
            #             imagedata = WidgetTexture(row)
            #             outputbox.add_widget(imagedata)
            #             break
            for row in self.myoutputimageslist:
                image = Filechooser.WidgetTexture(row)
                outputbox.remove_widget(image)
                outputbox.add_widget(image)

        for number in setoutputslast:
            lbloutput = Label(text=str(number))
            outputbox.add_widget(lbloutput)


        containeroutput.add_widget(outputbox)
        containeroutput.add_widget(popupoutputbtn)

        self.popupoutputs.open()

    def btnactivation(self):
        if self.pagenum <= 1:
            self.btnpreviouspage.disabled = True
        else:
            self.btnpreviouspage.disabled = False
        if self.pagenum >= int(self.pags):
            self.btnnextpage.disabled = True
        else:
            self.btnnextpage.disabled = False

    def popupoutputsdismiss(self):
        self.popupoutputs.dismiss()

    def open(self, path, filenameselected):
        global popupfilechooser
        global myevent

        popuploading = PopupLoading()
        self.popup = Popup(title=filenameselected, content=popuploading,
                      size_hint=(0.9, 0.9), auto_dismiss=False, separator_color=(.67, .57, .69))
        self.popup.open()

        popuploading.start()

        threading.Thread(target=Filechooser.filechoosershowdataset, args=(self, path, filenameselected, myevent)).start()
        popupfilechooser.dismiss()

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


    def popupdismiss(self):
        popupfilechooser.dismiss()







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



