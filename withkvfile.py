


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


#para poder volver a hacer train va a haber que duplicarlo y hacer un segundo train en el que primero se eliminen los widgets


numdata = 2
numoutputs = 1
n_outputs = 1
n_inputs = 2
list_hidden = list()
network = 2
a = 0
b = 0
c = 0
d = 0
e = 1
g = 0
i = 0
j = 0
l = 0
ti = TextInput(size=(40, 30), size_hint=(None, None))
ti2 = TextInput(size=(40, 30), size_hint=(None, None))
ti3 = TextInput(size=(40, 30), size_hint=(None, None))



jose= True


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
    for epoch in range(n_epoch):
        sum_error = 0
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
    return outputs
#accuracy
def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i][-1] == predicted[i]:
            correct += 1
    return (correct / float(len(actual))) * 100.0


class MyApp(App):
    Input = ''
    def background(self,wid,*largs):
        with wid.canvas:
            # Background
            Color(0.05, 0.05, 0.05, 1)
            Rectangle(size=(2000, 600))
            print(800)
            Color(0, 0, 0, 1)
            Rectangle(size=(2000, 600), pos=(0, 61))
            Color(1, 1, 1, 1)
            Line(points=[0, 60, 2000, 60])
            Color(1, 1, 1, 1)
            Line(points=[0, 60, 2000, 60])
    #when Go is pressed
    def drawingvisualize(self, draw, *largs):
        global network, ti, ti2, ti3
        a = 0
        b = 0
        c = 0
        d = 0
        e = 0
        g = 0
        i = 0
        j = 0
        l = 0
        Input = [float(ti.text), float(ti2.text), float(ti3.text)]
        predict(network, Input)

        numoutputs = n_outputs
        numdata = n_inputs
        middlex = draw.width / 2
        middley = draw.height / 2
        space = numdata * 40 + (numdata - 1) * 20
        spaceoutput = numoutputs * 40 + (numoutputs - 1) * 20


        # output layer
        for x in range(numoutputs):
            with draw.canvas:
                valuefloat= network[1][c]['output']
                Color(abs(valuefloat)/2,abs(valuefloat)/3,abs(valuefloat)/2)
                RoundedRectangle(size=(35, 35), pos=(middlex + 232.5, c * 50 + (middley+2.5 - spaceoutput / 2)), radius=[(6,6),(6,6),(6,6),(6,6)])
                outputtext = "{:.2f}".format(valuefloat)
                draw.add_widget(Label(text='[b]Probability:    [/b]'+ outputtext,markup = True, font_size=12, pos=(middlex + 50 , c * 50 + (middley-220 - spaceoutput / 2)), color = (1,1, 0.83)))
                c = c + 1

        inputtext = Label(text='[b]' + ti.text, markup = True, size=(30, 30), pos=(middlex-562, (middley-222 - space / 2)), color=(1,1,1,1))
        inputtext2 = Label(text='[b]' + ti2.text, markup = True, size=(30, 30), pos=(middlex-562, 50 + (middley-222 - space / 2)), color=(1,1,1, 1))
        inputtext3 = Label(text='[b]' + ti3.text, markup = True, size=(30, 30), pos=(middlex-562, 50 * 2 + (middley-222 - space / 2)), color=(1,1,1, 1))
        draw.add_widget(inputtext)
        draw.add_widget(inputtext2)
        draw.add_widget(inputtext3)
        draw.remove_widget(ti)
        draw.remove_widget(ti2)
        draw.remove_widget(ti3)

    # When Train is pressed
    def drawinglblblacktwo(self, draw, *largs):
        global a
        global b
        global list_hidden
        global network
        global n_outputs
        global n_inputs
        a = 0
        b = 0
        c = 0
        d = 0
        e = 0
        g = 0
        i = 0
        j = 0
        l = 0
        numoutputs = n_outputs
        numdata = n_inputs
        space = numdata * 30 + (numdata - 1) * 20
        spaceone = list_hidden[0] * 30 + (list_hidden[0] - 1) * 20
        spaceoutput = numoutputs * 30 + (numoutputs - 1) * 20


        with draw.canvas:
            # Clear
            Color(0, 0, 0, 1)
            Rectangle(size=(1300, 720), pos=(0, 111))
            Rectangle(size=(690, 800), pos=(0, 81))
        ti.pos = (70, 0 * 50 + (340 - space / 2))
        draw.remove_widget(ti)
        ti2.pos = (70, 1 * 50 + (340 - space / 2))
        draw.remove_widget(ti2)
        ti3.pos = (70, 2 * 50 + (340 - space / 2))
        draw.remove_widget(ti3)

        if len(list_hidden) == 1:
            # hiddenlayerone
            valuefloat = 0
            linecoloroutput = 0
            linecolorinput = 0
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
                    Color(0.8 -colorlayerone/10, 0.45 + colorlayerone / 3.23, 1 , colorlayerone/2 + 0.5)
                    RoundedRectangle(size=(30, 30), pos=(370, a * 50 + (340 - spaceone / 2)))
                    biases = "{:.2f}".format(value)
                    draw.nodeslabel = Label(text=biases, font_size=10, pos=(335, a * 50 + (305 - spaceone / 2)),
                                            color=(0, 0, 0, .9))
                    draw.add_widget(draw.nodeslabel)
                i = 0
                e = 0
                for y in range(numdata):
                    color = abs(network[0][a]['weights'][i]) / linecolorinput
                    with draw.canvas:
                        # Color(color, color-0.24, .15, .8)
                        # Color(0.37,color-0.17,color)
                        # Color(color, color-0.21, .57)
                        Color(color, color, color)
                        Line(points=[180, i * 50 + (325 - space / 2) + 30, 370, a * 50 + (340 - spaceone / 2) + 15])
                        i = i + 1
                for z in range(numoutputs):
                    color = abs(network[1][e]['weights'][a]) / linecoloroutput
                    with draw.canvas:
                        Color(color, color, color)
                        Line(points=[400, a * 50 + (340 - spaceone / 2) + 15, 590,
                                     e * 50 + (340 - spaceoutput / 2) + 15])
                    e = e + 1
                a = a + 1

        if len(list_hidden) == 2:
            # hiddenlayerone
            spacetwo = list_hidden[1] * 30 + (list_hidden[1] - 1) * 20
            valuefloat = 0
            linecoloroutput = 0
            linecolorinput = 0
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
                    Color(0.8 -colorlayerone/10, 0.45 + colorlayerone / 3.23, 1 , colorlayerone/2 + 0.5)
                    RoundedRectangle(size=(30, 30), pos=(315, a * 50 + (340 - spaceone / 2)))
                    biases = "{:.2f}".format(value)
                    draw.nodeslabel = Label(text=biases, font_size=10, pos=(281, a * 50 + (305 - spaceone / 2)),
                                            color=(0, 0, 0, .9))
                    draw.add_widget(draw.nodeslabel)
                i = 0
                e = 0
                for y in range(numdata):
                    color = abs(network[0][a]['weights'][i]) / linecolorinput
                    with draw.canvas:
                        Color(color, color, color)
                        Line(points=[180, i * 50 + (325 - space / 2) + 30, 315, a * 50 + (340 - spaceone / 2) + 15])
                        i = i + 1
                a = a + 1

            # hiddenlayertwo
            valuefloat = 0
            linecoloroutput = 0
            linecolorinput = 0
            d = 0
            a = 0
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
                    Color(0.67 - colorlayerone / 4.55, 1, 0.49 + colorlayerone / 2.94,colorlayerone/2 + 0.5)
                    RoundedRectangle(size=(30, 30), pos=(425, d * 50 + (340 - spacetwo / 2)))
                    biases = "{:.2f}".format(value)
                    draw.nodeslabel = Label(text=biases, font_size=10, pos=(391, d * 50 + (305 - spacetwo / 2)),
                                            color=(0, 0, 0, .9))
                    draw.add_widget(draw.nodeslabel)
                j = 0
                e = 0
                for y in range(list_hidden[0]):
                    color = abs(network[1][d]['weights'][j]) / linecolorinput
                    with draw.canvas:
                        Color(color, color, color)
                        Line(points=[345, j * 50 + (355 - spaceone / 2), 425, d * 50 + (355 - spacetwo / 2)])
                        j = j + 1
                for z in range(numoutputs):
                    color = abs(network[2][e]['weights'][d]) / linecoloroutput
                    with draw.canvas:
                        Color(color, color, color)
                        Line(points=[455, d * 50 + (355 - spacetwo / 2), 590, e * 50 + (355 - spaceoutput / 2)])
                    e = e + 1
                d = d + 1

        if len(list_hidden) == 3:
            # hiddenlayerone
            spacethree = list_hidden[2] * 30 + (list_hidden[2] - 1) * 20
            spacetwo = list_hidden[1] * 30 + (list_hidden[1] - 1) * 20
            spacetwo = list_hidden[1] * 30 + (list_hidden[1] - 1) * 20
            valuefloat = 0
            linecoloroutput = 0
            linecolorinput = 0
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
                    Color(0.8 -colorlayerone/10, 0.45 + colorlayerone / 3.23, 1 , colorlayerone/2 + 0.5)
                    RoundedRectangle(size=(30, 30), pos=(260, a * 50 + (340 - spaceone / 2)))
                    biases = "{:.2f}".format(value)
                    draw.nodeslabel = Label(text=biases, font_size=10, pos=(226, a * 50 + (305 - spaceone / 2)),
                                            color=(0, 0, 0, .9))
                    draw.add_widget(draw.nodeslabel)
                i = 0
                e = 0
                for y in range(numdata):
                    color = abs(network[0][a]['weights'][i]) / linecolorinput
                    with draw.canvas:
                        Color(color, color, color)
                        Line(points=[180, i * 50 + (325 - space / 2) + 30, 260,
                                     a * 50 + (340 - spaceone / 2) + 15])
                        i = i + 1
                a = a + 1

            # hiddenlayertwo
            valuefloat = 0
            linecoloroutput = 0
            linecolorinput = 0
            d = 0
            a = 0
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
                    Color(0.67 - colorlayerone / 4.55, 1, 0.49 + colorlayerone / 2.94, colorlayerone/2 + 0.5)
                    RoundedRectangle(size=(30, 30), pos=(370, d * 50 + (340 - spacetwo / 2)))
                    biases = "{:.2f}".format(value)
                    draw.nodeslabel = Label(text=biases, font_size=10, pos=(336, d * 50 + (305 - spacetwo / 2)),
                                            color=(0, 0, 0, .9))
                    draw.add_widget(draw.nodeslabel)
                j = 0
                e = 0
                for y in range(list_hidden[0]):
                    color = abs(network[1][d]['weights'][j]) / linecolorinput
                    with draw.canvas:
                        Color(color, color, color)
                        Line(points=[290, j * 50 + (355 - spaceone / 2), 370, d * 50 + (355 - spacetwo / 2)])
                        j = j + 1
                d = d + 1

            #hiddenlayerthree
            valuefloat = 0
            linecoloroutput = 0
            linecolorinput = 0
            d = 0
            a = 0
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
                    Color(0.64 + colorlayerone / 2.78, 0, 1 - colorlayerone / 4.55, colorlayerone/2 + 0.5)
                    RoundedRectangle(size=(30, 30), pos=(480, d * 50 + (340 - spacethree / 2)))
                    biases = "{:.2f}".format(value)
                    draw.nodeslabel = Label(text=biases, font_size=10, pos=(446, d * 50 + (305 - spacethree / 2)),
                                            color=(0, 0, 0, .9))
                    draw.add_widget(draw.nodeslabel)
                l = 0
                e = 0
                for y in range(list_hidden[1]):
                    color = abs(network[2][d]['weights'][l]) / linecolorinput
                    with draw.canvas:
                        Color(color, color, color)
                        Line(points=[400, l * 50 + (355 - spacetwo / 2), 480, d * 50 + (355 - spacethree / 2)])
                        l = l + 1
                for z in range(numoutputs):
                    color = abs(network[3][e]['weights'][d]) / linecoloroutput
                    with draw.canvas:
                        Color(color, color, color)
                        Line(points=[510, d * 50 + (355 - spacethree / 2), 590, e * 50 + (355 - spaceoutput / 2)])
                    e = e + 1
                d = d + 1

        draw.add_widget(Label(text='[b] OUTPUT ', markup=True, font_size=15, pos=(640, 500)))
        draw.add_widget(Label(text='[b] INPUT', markup=True, font_size=15, pos=(30, 500)))

        # output layer
        for x in range(numoutputs):
            outputoptions = str(c)
            lbloutputoptions = Label(text='[b]' + outputoptions, markup=True,
                                     pos=(580, c * 50 + (305 - spaceoutput / 2)), color=(1, 1, 1, 1))
            draw.add_widget(lbloutputoptions)
            value = network[-1][c]['weights'][-1]
            print(value)
            with draw.canvas:
                Color(1,.9,.49)
                RoundedRectangle(size=(30, 30), pos=(590, c * 50 + (340 - spaceoutput / 2)))
                biases = "{:.2f}".format(value)
                draw.nodeslabel = Label(text=biases, font_size=10, pos=(556, c * 50 + (305 - spaceoutput / 2)),
                                        color=(0, 0, 0, .9))
            c = c + 1


        draw.add_widget(ti)
        draw.add_widget(ti2)
        draw.add_widget(ti3)

        # input layer
        for x in range(numdata):
            with draw.canvas:
                Color(0.67 - b / 4.55, 1, 0.49 + b / 2.94)
                RoundedRectangle(size=(30, 30), pos=(150, b * 50 + (340 - space / 2)))
                b = b + 1
    def drawinglblblackkv(self, draw, titlelayout, *largs):
        global a
        global b
        global list_hidden
        global network
        global n_outputs
        global n_inputs
        a = 0
        b = 0
        c = 0
        d = 0
        e = 0
        g = 0
        i = 0
        j = 0
        l = 0
        numoutputs = n_outputs
        numdata = n_inputs
        space = numdata * 40 + (numdata - 1) * 20
        spaceone = list_hidden[0] * 40 + (list_hidden[0] - 1) * 20
        spaceoutput = numoutputs * 40 + (numoutputs - 1) * 20
        middlex = draw.width/2
        middley = draw.height/2
        #ima=(Image(source='lineaverde.png',size_hint=(.1,.1),pos=(middlex-200,draw.height+80)))
        #draw.add_widget(ima)


        with draw.canvas:
            # Clear
            Color(.05, .05, .05, 1)
            #Rectangle(size=draw.size, pos=(0, 0))
            #Rectangle(size=(1400,1400), pos=(-400,-65))
            #ima = (Image(source='backgroundmain.png', size_hint=(1, 1)))
            #draw.add_widget(ima)
        ti.pos = (0,(middley - space / 2))
        draw.remove_widget(ti)
        ti2.pos = (0, 1 * 50 + (middley - space / 2))
        draw.remove_widget(ti2)
        ti3.pos = (0, 2 * 50 + (middley - space / 2))
        draw.remove_widget(ti3)

        if len(list_hidden) == 1:
            # hiddenlayerone
            valuefloat = 0
            linecoloroutput = 0
            linecolorinput = 0
            #draw.add_widget(Label(text='[b]  Hidden\nlayer one', markup=True, font_size=15, pos=(middlex-290, middley - 10),  color=(.5,.6,.6)))
            draw.add_widget(Label(text='[b]  Hidden\nlayer one', markup=True, font_size=15, pos=(middlex, draw.height-15),  color=(.5,.6,.6), size_hint=(None,None)))
            # draw.add_widget(
            #     Label(text='hola', pos=(0, 0), size_hint=(None,None), size=(30,30)))
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
                    Color(0.6- colorlayerone/10, 0.78 + colorlayerone/33.33, 0.93- colorlayerone/3.85, colorlayerone/3.33 + 0.75)
                    RoundedRectangle(size=(40, 40), pos=(middlex, a * 50 + (middley - spaceone / 2)))
                    Color(colorlayerone/3,colorlayerone/2.1,colorlayerone/2,1)
                    RoundedRectangle(size=(35, 35), pos=(middlex+2.5, a * 50 + (middley+2.5 - spaceone / 2)), radius=[(6,6),(6,6),(6,6),(6,6)])
                    biases = "{:.2f}".format(value)
                    draw.nodeslabel = Label(text=biases, font_size=9, pos=(middlex-290, a * 50 + (middley-220 - spaceone / 2)),
                                            color=(1, 1, 1, .9))
                    draw.add_widget(draw.nodeslabel)
                i = 0
                e = 0
                for y in range(numdata):
                    color = abs(network[0][a]['weights'][i]) / linecolorinput
                    with draw.canvas:
                        # Color(color, color-0.24, .15, .8)
                        # Color(0.37,color-0.17,color)
                        # Color(color, color-0.21, .57)
                        Color(color, color, color)
                        Line(points=[middlex - 200, i * 50 + (middley-10 - space / 2) + 30, middlex, a * 50 + (middley+20 - spaceone / 2) ])
                        i = i + 1
                for z in range(numoutputs):
                    color = abs(network[1][e]['weights'][a]) / linecoloroutput
                    with draw.canvas:
                        Color(color, color, color)
                        Line(points=[middlex+40, a * 50 + (middley  - spaceone / 2)+ 20, middlex+230,
                                     e * 50 + (middley+ 20 - spaceoutput / 2) ])
                    e = e + 1
                a = a + 1

        if len(list_hidden) == 2:
            # hiddenlayerone
            spacetwo = list_hidden[1] * 40 + (list_hidden[1] - 1) * 20
            valuefloat = 0
            linecoloroutput = 0
            linecolorinput = 0
            draw.add_widget(Label(text='[b]  Hidden\nlayer one', markup=True, font_size=15, pos=(middlex-368, middley - 10), color=(.5,.6,.6)))
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
                    Color(0.59, 0.65+colorlayerone/7.69, 0.93, 0.5+colorlayerone/2)
                    RoundedRectangle(size=(40, 40), pos=(middlex-80, a * 50 + (middley - spaceone / 2)))
                    Color(colorlayerone/3,colorlayerone/2,colorlayerone/2,1)
                    RoundedRectangle(size=(35, 35), pos=(middlex + 2.5 - 80, a * 50 + (middley + 2.5 - spaceone / 2)),
                                     radius=[(6, 6), (6, 6), (6, 6), (6, 6)])
                    biases = "{:.2f}".format(value)
                    draw.nodeslabel = Label(text=biases, font_size=10, pos=(middlex-369, a * 50 + (middley -220 - spaceone / 2)),
                                            color=(1, 1, 1, .9))
                    draw.add_widget(draw.nodeslabel)
                i = 0
                e = 0
                for y in range(numdata):
                    color = abs(network[0][a]['weights'][i]) / linecolorinput
                    with draw.canvas:
                        Color(color, color, color)
                        Line(points=[middlex - 200, i * 50 + (middley-10 - space / 2) + 30, middlex-80, a * 50 + (middley + 20 - spaceone / 2) ])
                        i = i + 1
                a = a + 1

            # hiddenlayertwo
            valuefloat = 0
            linecoloroutput = 0
            linecolorinput = 0
            d = 0
            a = 0
            draw.add_widget(Label(text='[b]  Hidden\nlayer two', markup=True, font_size=15, pos=(middlex-213, middley - 10), color=(.5,.6,.6)))
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
                    Color(0.59, 0.65+colorlayerone/7.69, 0.93, 0.5+colorlayerone/2)
                    RoundedRectangle(size=(40, 40), pos=(middlex + 80, d * 50 + (middley - spacetwo / 2)))
                    Color(colorlayerone/3,colorlayerone/2.5,colorlayerone/2,1)
                    RoundedRectangle(size=(35, 35), pos=(middlex + 82.5, d * 50 + (middley + 2.5 - spacetwo / 2)),
                                     radius=[(6, 6), (6, 6), (6, 6), (6, 6)])
                    biases = "{:.2f}".format(value)
                    draw.nodeslabel = Label(text=biases, font_size=10, pos=(middlex -209, d * 50 + (middley-220 - spacetwo / 2)),
                                            color=(1, 1, 1, .9))
                    draw.add_widget(draw.nodeslabel)
                j = 0
                e = 0
                for y in range(list_hidden[0]):
                    color = abs(network[1][d]['weights'][j]) / linecolorinput
                    with draw.canvas:
                        Color(color, color, color)
                        Line(points=[middlex -40, j * 50 + (middley+20 - spaceone / 2), middlex+80, d * 50 + (middley+20 - spacetwo / 2)])
                        j = j + 1
                for z in range(numoutputs):
                    color = abs(network[2][e]['weights'][d]) / linecoloroutput
                    with draw.canvas:
                        Color(color, color, color)
                        Line(points=[middlex+120, d * 50 + (middley+20 - spacetwo / 2), middlex + 230, e * 50 + (middley +20 - spaceoutput / 2)])
                    e = e + 1
                d = d + 1

        if len(list_hidden) == 3:
            # hiddenlayerone
            spacethree = list_hidden[2] * 40 + (list_hidden[2] - 1) * 20
            spacetwo = list_hidden[1] * 40 + (list_hidden[1] - 1) * 20
            valuefloat = 0
            linecolorinput = 0
            draw.add_widget(Label(text='  Hidden\nlayer one', markup=True, font_size=15, pos=(middlex-420, middley -10), color=(.5,.6,.6)))
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
                    Color(0.49 + colorlayerone / 9.09, 0.9 - colorlayerone / 33.3, 0.67 + colorlayerone / 4.55, 0.5 + colorlayerone / 2)
                    RoundedRectangle(size=(40, 40), pos=(middlex - 130, a * 50 + (middley - spaceone / 2)))
                    Color(colorlayerone/3,colorlayerone/2,colorlayerone/2,1)
                    RoundedRectangle(size=(35, 35), pos=(middlex + 2.5 - 130, a * 50 + (middley + 2.5 - spaceone / 2)),
                                     radius=[(6, 6), (6, 6), (6, 6), (6, 6)])
                    biases = "{:.2f}".format(value)
                    draw.nodeslabel = Label(text=biases, font_size=10, pos=(middlex-419, a * 50 + (middley-220 - spaceone / 2)),
                                            color=(1, 1, 1, .9))
                    draw.add_widget(draw.nodeslabel)
                i = 0
                e = 0
                for y in range(numdata):
                    color = abs(network[0][a]['weights'][i]) / linecolorinput
                    with draw.canvas:
                        Color(color, color, color)
                        Line(points=[middlex - 200, i * 50 + (middley-10 - space / 2) + 30, middlex-130,
                                     a * 50 + (middley+ 20 - spaceone / 2) ])
                        i = i + 1
                a = a + 1

            # hiddenlayertwo
            valuefloat = 0
            linecolorinput = 0
            d = 0
            a = 0
            draw.add_widget(Label(text='  Hidden\nlayer two', markup=True, font_size=15, pos=(middlex-290, middley -10), color=(.5,.6,.6)))
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
                    Color(0.59, 0.65+colorlayerone/7.69, 0.93, 0.5+colorlayerone/2)
                    RoundedRectangle(size=(40, 40), pos=(middlex, d * 50 + (middley - spacetwo / 2)))
                    Color(colorlayerone/3,colorlayerone/2.1,colorlayerone/2,1)
                    RoundedRectangle(size=(35, 35), pos=(middlex + 2.5, d * 50 + (middley + 2.5 - spacetwo / 2)),
                                     radius=[(6, 6), (6, 6), (6, 6), (6, 6)])
                    biases = "{:.2f}".format(value)
                    draw.nodeslabel = Label(text=biases, font_size=10, pos=(middlex-289, d * 50 + (middley-220 - spacetwo / 2)),
                                            color=(1, 1, 1, .9))
                    draw.add_widget(draw.nodeslabel)
                j = 0
                e = 0
                for y in range(list_hidden[0]):
                    color = abs(network[1][d]['weights'][j]) / linecolorinput
                    with draw.canvas:
                        Color(color, color, color)
                        Line(points=[middlex-90, j * 50 + (middley+20 - spaceone / 2), middlex, d * 50 + (middley+20 - spacetwo / 2)])
                        j = j + 1
                d = d + 1

            #hiddenlayerthree
            valuefloat = 0
            linecoloroutput = 0
            linecolorinput = 0
            d = 0
            a = 0
            draw.add_widget(Label(text='   Hidden\nlayer three', markup=True, font_size=15, pos=(middlex-160, middley -10), color=(.5,.6,.6)))
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
                    Color(0.58 + colorlayerone / 16.65, 0.65-colorlayerone/7.69, 0.92, 0.5+colorlayerone/2)
                    RoundedRectangle(size=(40, 40), pos=(middlex+130, d * 50 + (middley - spacethree / 2)))
                    Color(colorlayerone/3,colorlayerone/2.5,colorlayerone/2,1)
                    RoundedRectangle(size=(35, 35), pos=(middlex + 132.5, d * 50 + (middley + 2.5 - spacethree / 2)),
                                     radius=[(6, 6), (6, 6), (6, 6), (6, 6)])
                    biases = "{:.2f}".format(value)
                    draw.nodeslabel = Label(text=biases, font_size=10, pos=(middlex-160, d * 50 + (middley-220 - spacethree / 2)),
                                            color=(1, 1, 1, .9))
                    draw.add_widget(draw.nodeslabel)
                l = 0
                e = 0
                for y in range(list_hidden[1]):
                    color = abs(network[2][d]['weights'][l]) / linecolorinput
                    with draw.canvas:
                        Color(color, color, color)
                        Line(points=[middlex+40, l * 50 + (middley+20 - spacetwo / 2), middlex+130, d * 50 + (middley+20 - spacethree / 2)])
                        l = l + 1
                for z in range(numoutputs):
                    color = abs(network[3][e]['weights'][d]) / linecoloroutput
                    with draw.canvas:
                        Color(color, color, color)
                        Line(points=[middlex+170, d * 50 + (middley+20 - spacethree / 2), middlex+230, e * 50 + (middley+20 - spaceoutput / 2)])
                    e = e + 1
                d = d + 1



        # output layer
        for x in range(numoutputs):
            outputoptions = str(c)
            lbloutputoptions = Label(text='[b]' + outputoptions, markup=True,
                                     pos=(middlex-30, c * 50 + (middley-220 - spaceoutput / 2)), color=(1, 1, 1, 1))
            draw.add_widget(lbloutputoptions)
            value = network[-1][c]['weights'][-1]
            print(value)
            with draw.canvas:
                #Color(0.45,1,0.83)
                Color(.64,.52,.91)
                RoundedRectangle(size=(40, 40), pos=(middlex + 230, c * 50 + (middley - spaceoutput / 2)))
                Color(0, 0, 0)
                RoundedRectangle(size=(35, 35), pos=(middlex + 232.5, c * 50 + (middley+2.5 - spaceoutput / 2)), radius=[(6,6),(6,6),(6,6),(6,6)])
                biases = "{:.2f}".format(value)
                #draw.nodeslabel = Label(text=biases, font_size=10, pos=(middlex + 196, c * 50 + (middley-34 - spaceoutput / 2)),color=(1, 1, 1, .9))
                draw.add_widget(Label(text='Output', markup=True, font_size=17, pos=(middlex-60, middley-10), color=(.4,.3,.4)))
                draw.add_widget(Label(text='Input', markup=True, font_size=17, pos=(middlex-530, middley-10), color=(.4,.7,.6)))
            c = c + 1

        # input layer
        for x in range(numdata):
            with draw.canvas:
                # Color(1,1, .53)
                Color(.49, .90, .67)
                RoundedRectangle(size=(40, 40), pos=(middlex - 240, b * 50 + (middley - space / 2)))
                Color(0, 0, 0)
                RoundedRectangle(size=(35, 35), pos=(middlex - 237.5, b * 50 + (middley + 2.5 - space / 2)),
                                 radius=[(6, 6), (6, 6), (6, 6), (6, 6)])
                b = b + 1

        draw.add_widget(ti)
        draw.add_widget(ti2)
        draw.add_widget(ti3)

        #data
        draw.add_widget(Label(text='[b] Accuracy: [\b]', markup = True, pos=(middlex-300, -290)))




    def build(self):

        list_hidden.append(6)
        numlayers = str(len(list_hidden))


        if jose:
            parentlayout = LayoutScreen()
        else:
            parentlayout = BoxLayout(orientation='vertical', size_hint=(1,1))


        if jose:
            layout = parentlayout.ids.idLayoutMenu.ids.idBoxLayoutMenu
        else:
            layout = BoxLayout(orientation = 'horizontal',size_hint=(1,None),height = 50, spacing=5)
            parentlayout.add_widget(layout)

        neuronlayout = BoxLayout(orientation='vertical')
        neurontwolayout = BoxLayout(orientation='vertical')
        neuronthreelayout = BoxLayout(orientation='vertical')
        numlayerslayout = BoxLayout(orientation='vertical')
        ratelayout = BoxLayout(orientation = 'vertical',pos_hint={'top':1}, size_hint=(0.8,0.8))

        btnGo = Button(color=(1,1,1,1), text='[b]GO',markup = True, font_size=20, size_hint=(.95,.9),
                       pos_hint={'center_x':.5, 'center_y':.5},background_normal='ButtonPlusP.png', background_down='ButtonPlusPDown.png')
        btnTrain = Button(text='[b]TRAIN',markup = True, background_normal='ButtonPlusG.png', background_down='ButtonPlusADown.png', size_hint=(.95,.9),
                       pos_hint={'center_x':.5, 'center_y':.5})
        btnRefresh = Button(text='[b]REFRESH',markup = True, background_normal='ButtonPlusG.png', background_down='ButtonPlusADown.png')
        btnnewinput=Button(color=(1,1,1,1), text='[b]New Input',markup = True, font_size=20, size_hint=(.95,.9),
                       pos_hint={'center_x':.5, 'center_y':.5},background_normal='ButtonPlusP.png', background_down='ButtonPlusPDown.png')


        box = CheckBox()
        lblbox = Label(text='Sigmoid:', color=(1, 1, 1, 1), font_size=(14))
        boxshowbiaslabel = CheckBox()
        lblshowbias = Label(text='Show biases:', color=(1, 1, 1, 1), font_size=(14))

        # Left Neuron Switcher
        btnplusneuron = Button(text = '[b]+',markup = True, size_hint=(.4,.5),pos_hint={'center_x':.5, 'center_y':.5},
                               background_normal='ButtonPlusG.png',
                               background_down='ButtonPlusRDown.png',
                               background_disabled_normal='ButtonDeactivated.png')
        btnlessneuron = Button(text = '[b]-',markup = True,size_hint=(.4, .5),pos_hint={'center_x':.5, 'center_y':.5},
                               background_normal='ButtonPlusG.png',
                               background_down='ButtonPlusRDown.png',
                               background_disabled_normal='ButtonDeactivated.png')
        lblhiddenlayer = Label(text=str(list_hidden[0]))
        lblneuron = Label(text='Neurons')

        # Second neuron switcher
        btnplusneurontwo = Button(text = '[b]+',markup = True,size_hint=(.4, .5),pos_hint={'center_x':.5, 'center_y':.5},
                                  background_normal='ButtonPlusG.png',
                                  background_down='ButtonPlusPDown.png',
                                  background_disabled_normal='ButtonDeactivated.png')
        btnlessneurontwo = Button(text = '[b]-',markup = True,size_hint=(.4, .5), pos_hint={'center_x':.5, 'center_y':.5},
                                  background_normal='ButtonPlusG.png',
                                  background_down='ButtonPlusPDown.png',
                                  background_disabled_normal='ButtonDeactivated.png')
        lblhiddenlayertwo = Label(text='4')
        lblneurontwo = Label(text='Neurons')

        # Right/Third neuron switcher
        btnplusneuronthree = Button(text = '[b]+',markup = True,size_hint=(.4, .5),pos_hint={'center_x':.5, 'center_y':.5},
                                    background_normal='ButtonPlusG.png',
                                    background_down='ButtonPlusADown.png',
                                    background_disabled_normal='ButtonDeactivated.png')
        btnlessneuronthree = Button(text = '[b]-',markup = True,size_hint=(.4, .5),pos_hint={'center_x':.5, 'center_y':.5},
                                    background_normal='ButtonPlusG.png',
                                    background_down='ButtonPlusADown.png',
                                    background_disabled_normal='ButtonDeactivated.png')
        lblhiddenlayerthree = Label(text='4')
        lblneuronthree = Label(text='Neurons')

        # Number of layers switcher
        btnlayerplusneuron = Button(text = '[b]+',markup = True,size_hint=(.4, .5),pos_hint={'center_x':.5, 'center_y':.5},
                                    background_normal='ButtonPlusG.png',
                                    background_down='ButtonPlusGDown.png',
                                    background_disabled_normal='ButtonDeactivated.png')
        btnlayerlessneuron = Button(text = '[b]-',markup = True,size_hint=(.4, .5),pos_hint={'center_x':.5, 'center_y':.5},
                                    background_normal='ButtonPlusG.png',
                                    background_down='ButtonPlusGDown.png',
                                    background_disabled_normal='ButtonDeactivated.png')
        lbllayer = Label(text=numlayers)
        lblnumlayers = Label(text='Hiddenlayers',font_size=(11),pos_hint={'center_x':.5, 'center_y':.5})
        lblblack = Widget()
        lblblacktwo = Widget()
        canvas= Widget(size_hint=(.1,.1))
        visualize = Label(size_hint=(.1,.1))

        lblrate = Label(text='RANGE:',font_size=(11),pos_hint={'center_x':.5, 'center_y':.5})
        tirate = TextInput(pos_hint={'center_x':.5, 'center_y':.5})


        if jose:
            pass
        else:
            self.background(canvas)
            layout.add_widget(canvas)
        layout.add_widget(btnTrain)
        layout.add_widget(lblblacktwo)
        ratelayout.add_widget(lblrate)
        ratelayout.add_widget(tirate)
        #layout.add_widget(ratelayout)
        layout.add_widget(btnlessneuron)
        neuronlayout.add_widget(lblneuron)
        neuronlayout.add_widget(lblhiddenlayer)
        layout.add_widget(neuronlayout)
        layout.add_widget(btnplusneuron)
        layout.add_widget(lblblack)
        layout.add_widget(btnlayerlessneuron)
        numlayerslayout.add_widget(lblnumlayers)
        numlayerslayout.add_widget(lbllayer)
        layout.add_widget(numlayerslayout)
        layout.add_widget(btnlayerplusneuron)
        btnlayerlessneuron.disabled = True
        layout.add_widget(visualize)


        neurontwolayout.add_widget(lblneurontwo)
        neurontwolayout.add_widget(lblhiddenlayertwo)

        neuronthreelayout.add_widget(lblneuronthree)
        neuronthreelayout.add_widget(lblhiddenlayerthree)

        #layout.add_widget(box)
        #layout.add_widget(lblbox)
        #layout.add_widget(boxshowbiaslabel)

        def btnplusneuronpressed (btnplusneuron):

            btnlessneuron.disabled = False

            if list_hidden[0] >= 10:
                list_hidden[0] = 10
            else:
                list_hidden[0] = list_hidden[0] + 1
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
                if list_hidden[0] == 1:
                    btnlessneuron.disabled = True

        def btnplusneurontwopressed(btnplusneurontwo):
            btnlessneurontwo.disabled = False

            if list_hidden[1] >= 10:
                list_hidden[1] = 10
            else:
                list_hidden[1] = list_hidden[1] + 1
                lblhiddenlayertwo.text = str(list_hidden[1])
                if list_hidden[1] == 10:
                    btnplusneurontwo.disabled = True
        def btnlessneurontwopressed(btnlessneurontwo):
            btnplusneurontwo.disabled = False
            if list_hidden[1] <= 1:
                list_hidden[1] = 1
            else:
                list_hidden[1] = list_hidden[1] - 1
                lblhiddenlayertwo.text = str(list_hidden[1])
                if list_hidden[1] == 1:
                    btnlessneurontwo.disabled = True

        def btnplusneuronthreepressed(btnplusneuronthree):
            btnlessneuronthree.disabled = False

            if list_hidden[2] >= 10:
                list_hidden[2] = 10
            else:
                list_hidden[2] = list_hidden[2] + 1
                lblhiddenlayerthree.text = str(list_hidden[2])
                if list_hidden[2] == 10:
                    btnplusneuronthree.disabled = True
        def btnlessneuronthreepressed(btnlessneuronthree):
            btnplusneuronthree.disabled = False
            if list_hidden[2] <= 1:
                list_hidden[2] = 1
            else:
                list_hidden[2] = list_hidden[2] - 1
                lblhiddenlayerthree.text = str(list_hidden[2])
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
                    layout.remove_widget(btnlayerplusneuron)
                    layout.remove_widget(numlayerslayout)
                    layout.remove_widget(btnlayerlessneuron)
                    layout.remove_widget(lblblack)

                    layout.add_widget(btnlessneurontwo)
                    layout.add_widget(neurontwolayout)
                    layout.add_widget(btnplusneurontwo)
                    layout.add_widget(lblblack)
                    layout.add_widget(btnlayerlessneuron)
                    layout.add_widget(numlayerslayout)
                    layout.add_widget(btnlayerplusneuron)
                    lbllayer.text = str(len(list_hidden))
                    btnlayerplusneuron.disabled = False
                elif len(list_hidden) == 2:
                    list_hidden.append(4)
                    layout.remove_widget(btnlayerplusneuron)
                    layout.remove_widget(btnlayerlessneuron)
                    layout.remove_widget(numlayerslayout)
                    layout.remove_widget(lblblack)

                    layout.add_widget(btnlessneuronthree)
                    layout.add_widget(neuronthreelayout)
                    layout.add_widget(btnplusneuronthree)
                    layout.add_widget(lblblack)
                    layout.add_widget(btnlayerlessneuron)
                    layout.add_widget(numlayerslayout)
                    layout.add_widget(btnlayerplusneuron)
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
                if len(list_hidden) == 2:
                    layout.remove_widget(btnplusneuronthree)
                    layout.remove_widget(neuronthreelayout)
                    layout.remove_widget(btnlessneuronthree)
                    btnlayerlessneuron.disabled = False

                elif len(list_hidden) == 1:
                    layout.remove_widget(btnplusneurontwo)
                    layout.remove_widget(neurontwolayout)
                    layout.remove_widget(btnlessneurontwo)

                    btnlayerlessneuron.disabled = True


        def btngopressed(bntGo):
            if jose:
                #layoutbtgo = parentlayout.ids.idLayoutNeuron.ids.idLayoutNeuronMainScreen
                layoutbntgo = parentlayout.ids.idLayoutNeuron.ids.idLayoutNeuronMainScreen.ids.idrelativelayoutneuronmainscreen
                self.drawingvisualize(layoutbntgo)
                layout.remove_widget(btnRefresh)
                layout.remove_widget(btnGo)
                layout.remove_widget(lblshowbias)
                layout.remove_widget(boxshowbiaslabel)
                layout.add_widget(btnnewinput)
                layout.add_widget(lblshowbias)
                layout.add_widget(boxshowbiaslabel)

            else:
                self.drawingvisualize(visualize)

        def btntrainpressed(bntTrain):
            global network, n_inputs, n_outputs
            seed(1)
            dataset = [[1, 2, 3, 1],
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
                       [0, 1, 2, 0]]
            '''dataset = [[0, 0, 1, 0],
                       [1, 1, 1, 1],
                       [1, 0, 1, 1],
                       [0, 1, 0, 0]]'''

            n_inputs = len(dataset[0]) - 1
            n_outputs = len(set([row[-1] for row in dataset]))
            network = initialize_network(n_inputs, list_hidden, n_outputs)
            train_network(network, dataset, 0.5, 20, n_outputs)
            # for row in dataset:
            #     prediction = predict(network, row)
            #     accuracy_metric(dataset,prediction)
            for layer in network:
                print(layer)


            if jose:
                layoutneuronmainscreen = parentlayout.ids.idLayoutNeuron.ids.idLayoutNeuronMainScreen.ids.idrelativelayoutneuronmainscreen
                layoutlabels = parentlayout.ids.idLayoutNeuron.ids.idLayoutNeuronMainScreen.ids.idLabelsLayoutNeuronMainScreen
                self.drawinglblblackkv(layoutneuronmainscreen, layoutlabels)
            else:
                self.drawinglblblacktwo(lblblacktwo)

            layout.remove_widget(btnplusneuronthree)
            layout.remove_widget(neuronthreelayout)
            layout.remove_widget(btnlessneuronthree)
            layout.remove_widget(btnplusneurontwo)
            layout.remove_widget(neurontwolayout)
            layout.remove_widget(btnlessneurontwo)
            layout.remove_widget(btnlayerplusneuron)
            layout.remove_widget(numlayerslayout)
            layout.remove_widget(btnlayerlessneuron)
            layout.remove_widget(lblblack)
            layout.remove_widget(btnTrain)
            layout.remove_widget(btnplusneuron)
            layout.remove_widget(btnlessneuron)
            layout.remove_widget(neuronlayout)
            layout.remove_widget(lblblack)

            layout.add_widget(btnGo)
            layout.add_widget(btnRefresh)
            layout.add_widget(lblshowbias)
            layout.add_widget(boxshowbiaslabel)

        def btnrefreshpressed(btnRefresh):
            with btnRefresh.canvas:
                # Clear
                Color(0, 0, 0, 1)
                Rectangle(size=(1300, 720), pos=(0, 111))
                Rectangle(size=(690, 800), pos=(0, 81))

        #box.active = True
        btnTrain.bind(on_press=btntrainpressed)
        btnGo.bind(on_press=btngopressed)
        btnplusneuron.bind(on_press=btnplusneuronpressed)
        btnlessneuron.bind(on_press=btnlessneuronpressed)
        btnplusneurontwo.bind(on_press=btnplusneurontwopressed)
        btnlessneurontwo.bind(on_press=btnlessneurontwopressed)
        btnplusneuronthree.bind(on_press=btnplusneuronthreepressed)
        btnlessneuronthree.bind(on_press=btnlessneuronthreepressed)
        btnlayerplusneuron.bind(on_press=btnlayerplusneuronpressed)
        btnlayerlessneuron.bind(on_press=btnlayerlessneuronpressed)
        btnRefresh.bind(on_press=btnrefreshpressed)
        #boxshowbiaslabel.bind(on_press=showbiaslabels)


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



