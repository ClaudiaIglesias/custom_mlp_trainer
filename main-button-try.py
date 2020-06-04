


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



hiddenlayer=6
hiddenlayertwo = 7
hiddenlayerthree = 7
layers = 1
numdata = 2
numoutputs = 1
a = 0
b = 0
c = 0
d = 0
e = 1
g = 0
i = 0
j = 0
l = 0


def on_resize(width, height):
    print('a')

def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
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

def predict(network, row):
    outputs = forward_propagate(network, row)
    print(outputs)



class First(Widget):
    def __init__(self,**kwargs):
        super(First, self).__init__(**kwargs)
        global hiddenlayer
        global layers

        width,height = Window.size
        print(width)

        numhiddenlayer = str(hiddenlayer)
        numhiddenlayertwo = str(hiddenlayertwo)
        numhiddenlayerthree = str(hiddenlayerthree)
        numlayers = str(layers)


        with self.canvas:
            #Background
            Color(0.05, 0.05, 0.05, 1)
            Rectangle(size=(width, height))
            print(width)
            Color(0, 0, 0, 1)
            Rectangle(size=(width, height), pos=(0, 81))
            Color(1,1,1,1)
            Line(points=[0,80,width,80])
            Color(1, 1, 1, 1)
            Line(points=[0, 80, width, 80])

        self.btnGo = Button(text='GO', size = (80, 30), pos=(20, 40), background_normal = 'ButtonPlus.png',
                                    background_down = 'ButtonPlusDown.png')
        self.btnTrain = Button(text='TRAIN', size = (80,30), pos=(20,5), background_color = (.5, .5, .5, 1))
        self.box = CheckBox(pos=(720, 50))
        self.lblbox = Label(text='Sigmoid:', pos=(675, 50), color=(1, 1, 1, 1), font_size=(14))
        self.boxshowbiaslabel = CheckBox(pos=(590, 10))
        self.lblshowbias = Label(text='Show biases', pos=(550, 10), color=(1, 1, 1, 1), font_size=(14))
        self.btnadjustment = Button(text='adjust screen', size = (40,30), pos=(10,400), background_color = (.8, .8, .8, 0.5))


        #Left Neuron Switcher
        self.btnplusneuron = Button(size=(25, 25), pos=(345, 18),
                                    background_normal = 'ButtonPlus.png',
                                    background_down = 'ButtonPlusDown.png',
                                    background_disabled_normal = 'ButtonDeactivated.png')
        self.lblplusbutton = Label(text='+', pos=(308,-19), font_size=(20), color=(0,0,0,1))
        self.btnlessneuron = Button(size=(25, 25), pos=(399, 18),
                                    background_normal='ButtonPlus.png',
                                    background_down='ButtonPlusDown.png',
                                    background_disabled_normal = 'ButtonDeactivated.png')
        self.lbllessbutton = Label(text='-', pos=(362, -19), font_size=(20), color=(0, 0, 0, 1))
        self.lblhiddenlayer = Label(text= numhiddenlayer, pos=(335,-20))
        self.lblneuron = Label(text='Neurons', pos=(335,10))

        #Second neuron switcher
        self.btnplusneurontwo = Button(size=(25, 25), pos=(398, 18),
                                    background_normal='ButtonPlus.png',
                                    background_down='ButtonPlusDown.png',
                                    background_disabled_normal='ButtonDeactivated.png')
        self.lblplusbuttontwo = Label(text='+', pos=(361, -19), font_size=(20), color=(0, 0, 0, 1))
        self.btnlessneurontwo = Button(size=(25, 25), pos=(452, 18),
                                    background_normal='ButtonPlus.png',
                                    background_down='ButtonPlusDown.png',
                                    background_disabled_normal='ButtonDeactivated.png')
        self.lbllessbuttontwo = Label(text='-', pos=(415, -19), font_size=(20), color=(0, 0, 0, 1))
        self.lblhiddenlayertwo = Label(text=numhiddenlayertwo, pos=(388, -20))
        self.lblneurontwo = Label(text='Neurons', pos=(388, 10))

        #Right/Third neuron switcher
        self.btnplusneuronthree = Button(size=(25, 25), pos=(453, 18),
                                    background_normal='ButtonPlus.png',
                                    background_down='ButtonPlusDown.png',
                                    background_disabled_normal='ButtonDeactivated.png')
        self.lblplusbuttonthree = Label(text='+', pos=(416, -19), font_size=(20), color=(0, 0, 0, 1))
        self.btnlessneuronthree = Button(size=(25, 25), pos=(507, 18),
                                    background_normal='ButtonPlus.png',
                                    background_down='ButtonPlusDown.png',
                                    background_disabled_normal='ButtonDeactivated.png')
        self.lbllessbuttonthree = Label(text='-', pos=(470, -19), font_size=(20), color=(0, 0, 0, 1))
        self.lblhiddenlayerthree = Label(text=numhiddenlayerthree, pos=(445, -20))
        self.lblneuronthree = Label(text='Neurons', pos=(445, 10))

        #Number of layers switcher
        self.btnlayerplusneuron = Button(size=(25, 25), pos=(680, 18),
                                         background_normal='ButtonPlus.png',
                                         background_down='ButtonPlusDown.png',
                                         background_disabled_normal = 'ButtonDeactivated.png')
        self.lbllayerplusbutton = Label(text='+', pos=(643, -19), font_size=(20), color=(0, 0, 0, 1))
        self.btnlayerlessneuron = Button(size=(25, 25), pos=(734, 18),
                                         background_normal='ButtonPlus.png',
                                         background_down='ButtonPlusDown.png',
                                         background_disabled_normal = 'ButtonDeactivated.png')
        self.lbllayerlessbutton = Label(text='-', pos=(697, -19), font_size=(20), color=(0, 0, 0, 1))
        self.lbllayer = Label(text=numlayers, pos=(670, -20))
        self.lblnumlayers = Label(text='Hidden layers', pos=(670, 10))

        self.add_widget(self.btnGo)
        self.add_widget(self.btnTrain)
        self.add_widget(self.btnplusneuron)
        self.add_widget(self.lblplusbutton)
        self.add_widget(self.btnlessneuron)
        self.add_widget(self.lbllessbutton)
        self.add_widget(self.btnlayerlessneuron)
        self.add_widget(self.btnlayerplusneuron)
        self.add_widget(self.lbllayerlessbutton)
        self.add_widget(self.lbllayerplusbutton)
        self.add_widget(self.box)
        self.add_widget(self.lblbox)
        self.add_widget(self.boxshowbiaslabel)
        self.add_widget(self.lblhiddenlayer)
        self.add_widget(self.lblneuron)
        self.add_widget(self.lblnumlayers)
        self.add_widget(self.lbllayer)
        self.add_widget(self.btnadjustment)


        self.box.active = True
        self.btnTrain.bind(on_press=self.btnpressed)
        self.btnGo.bind(on_press=self.btngopressed)
        self.btnplusneuron.bind(on_press=self.btnplusneuronpressed)
        self.btnlessneuron.bind(on_press=self.btnlessneuronpressed)
        self.btnplusneurontwo.bind(on_press=self.btnplusneurontwopressed)
        self.btnlessneurontwo.bind(on_press=self.btnlessneurontwopressed)
        self.btnplusneuronthree.bind(on_press=self.btnplusneuronthreepressed)
        self.btnlessneuronthree.bind(on_press=self.btnlessneuronthreepressed)
        self.btnlayerplusneuron.bind(on_press=self.btnlayerplusneuronpressed)
        self.btnlayerlessneuron.bind(on_press=self.btnlayerlessneuronpressed)
        self.boxshowbiaslabel.bind(on_press=self.showbiaslabels)
        self.btnadjustment.bind(on_press=self.btnadjusmentspressed)

        Window.bind(on_resize=self.check_resize)


    def btnpressed(self, instance):
        global a
        global b
        global hiddenlayer
        a = 0
        b = 0
        c = 0
        d = 0
        e = 0
        g = 0
        i = 0
        j = 0
        l = 0
        screen_width = self.get_right()
        screen_height = self.get_top

        # Test training backpropagation algorithm
        seed(1)
        dataset = [[1, 2, 1],
                   [2, 1, 2],
                   [1, 1, 1],
                   [0, 0, 0],
                   [1, 0, 1],
                   [3, 1, 3],
                   [2, 0, 2],
                   [0, 2, 0],
                   [3, 2, 3],
                   [0, 1, 0]]
        n_inputs = len(dataset[0]) - 1
        n_outputs = len(set([row[-1] for row in dataset]))
        network = initialize_network(n_inputs, hiddenlayer, n_outputs)
        train_network(network, dataset, 0.5, 20, n_outputs)
        for layer in network:
            print(layer)


        numoutputs = n_outputs
        space = numdata * 30 + (numdata - 1) * 20
        spaceone = hiddenlayer * 30 + (hiddenlayer - 1) * 20
        spacetwo = hiddenlayertwo * 30 + (hiddenlayertwo - 1) * 20
        spacethree = hiddenlayerthree * 30 + (hiddenlayerthree - 1) * 20
        spaceoutput = numoutputs * 30 + (numoutputs - 1) * 20

        with self.canvas:
            #Clear
            Color(1, 0, 0, 1)
            Rectangle(size=(screen_width,720), pos=(0,111))
            Rectangle(size=(690,800), pos=(0,81))


        if layers == 1:
            # hiddenlayerone
            for x in range(hiddenlayer):
                with self.canvas:
                    Color(1, 1, .9, abs(network[0][a]['weights'][2]))
                    RoundedRectangle(size=(30, 30), pos=(370, a*50+(340-spaceone/2)))
                    biases = str(network[0][a]['weights'][2])
                    self.nodeslabel = Label(text=biases, font_size=10, pos=(335, a * 50 + (300 - spaceone / 2)))
                    self.add_widget(self.nodeslabel)
                i = 0
                e = 0
                for y in range(numdata):
                    with self.canvas:
                        Color(1,1,.9,abs(network[0][a]['weights'][i]))
                        Line(points=[180, i*50+(325-space/2)+30, 370, a*50+(340-spaceone/2)+15])
                        i = i + 1
                for z in range(numoutputs):
                    with self.canvas:
                        #Color(1, 1, .9, abs(network[1][e]['weights'][0]))
                        Line(points=[400, a * 50 + (340 - spaceone / 2) + 15, 590, e * 50 + (340 - spaceoutput / 2)+ 15])
                        e = e + 1
                a = a + 1

        if layers == 2:
            # hiddenlayerone
            for x in range(hiddenlayer):
                with self.canvas:
                    Color(1, 1, .9, 1)
                    RoundedRectangle(size=(30, 30), pos=(315, a*50+(340-spaceone/2)))
                    i = 0
                    for x in range(numdata):
                        with self.canvas:
                            Line(points=[180, i*50+(325-space/2)+30, 315, a*50+(340-spaceone/2)+15])
                        i = i + 1
                    a = a + 1

            # hiddenlayertwo
            for x in range(hiddenlayertwo):
                with self.canvas:
                    Color(1, 1, .9, 1)
                    RoundedRectangle(size=(30, 30), pos=(425, d*50+(340-spacetwo/2)))
                    Line(points=[455, d*50+(340-spacetwo/2)+15, 590, 340])
                    j = 0
                    for x in range(hiddenlayer):
                         with self.canvas:
                            Line(points=[345, j*50+(340-spaceone/2)+15, 425, d*50+(340-spacetwo/2)+15])
                            j = j + 1
                d = d + 1

        if layers == 3:
            # hiddenlayerone
            for x in range(hiddenlayer):
                with self.canvas:
                    Color(1, 1, .9, 1)
                    RoundedRectangle(size=(30, 30), pos=(260, a*50+(340-spaceone/2)))
                    i = 0
                    for x in range(numdata):
                        with self.canvas:
                            Line(points=[180, i*50+(325-space/2)+30, 260, a*50+(340-spaceone/2)+15])
                        i = i + 1
                    a = a + 1
            # hiddenlayertwo
            for x in range(hiddenlayertwo):
                with self.canvas:
                    Color(1, 1, .9, 1)
                    RoundedRectangle(size=(30, 30), pos=(370, d*50+(340-spacetwo/2)))
                    j = 0
                    for x in range(hiddenlayer):
                        with self.canvas:
                            Line(points=[290, j*50+(340-spaceone/2)+15, 370, d*50+(340-spacetwo/2)+15])
                            j = j + 1
                    d = d + 1
            # hiddenlayerthree
            for x in range(hiddenlayerthree):
                with self.canvas:
                    Color(1, 1, .9, 1)
                    RoundedRectangle(size=(30, 30), pos=(480, g*50+(340-spacethree/2)))
                    Line(points=[510, g*50+(340-spacethree/2)+15, 590, 340])
                    l = 0
                    for x in range(hiddenlayertwo):
                        with self.canvas:
                            Line(points=[400, l*50+(340-spacetwo/2)+15, 480, g*50+(340-spacethree/2)+15])
                            l = l + 1
                    g = g + 1


        self.add_widget(Label(text='OUTPUT', font_size=15, pos=(640, 500)))
        self.add_widget(Label(text='INPUT', font_size=15, pos=(120, 500)))


        #output layer
        for x in range(numoutputs):
            with self.canvas:
                Color(1, 1, .9,abs(network[1][c]['output']))
                RoundedRectangle(size=(30, 30), pos=(590,c*50+(340-spaceoutput/2)))
                outputtext = str(network[1][c]['output'])
                self.add_widget(Label(text=outputtext, font_size=10, pos=(640, c*50+(305-spaceoutput/2))))
                c = c + 1

        # input layer
        for x in range(numdata):
            with self.canvas:
                Color(1, 1, .9, 1)
                RoundedRectangle(size=(30, 30), pos=(150,b*50+(340-space/2)))
                b = b + 1
    def btngopressed(selfself, instance):
        predict(network, dataset[0])


    def btnplusneuronpressed(self, instance):
        global hiddenlayer
        self.btnlessneuron.disabled = False

        if hiddenlayer >= 10:
            hiddenlayer = 10
        else:
            hiddenlayer = hiddenlayer + 1
            numhiddenlayer = str(hiddenlayer)
            self.lblhiddenlayer.text = numhiddenlayer
            if hiddenlayer == 10:
                self.btnplusneuron.disabled = True
    def btnlessneuronpressed(self, instance):
        global hiddenlayer
        self.btnplusneuron.disabled = False
        if hiddenlayer <= 1:
            hiddenlayer = 1
        else:
            hiddenlayer = hiddenlayer - 1
            numhiddenlayer = str(hiddenlayer)
            self.lblhiddenlayer.text = numhiddenlayer
            if hiddenlayer == 1:
                self.btnlessneuron.disabled = True

    def btnplusneurontwopressed(self, instance):
        global hiddenlayertwo
        self.btnlessneurontwo.disabled = False

        if hiddenlayertwo >= 10:
            hiddenlayertwo = 10
        else:
            hiddenlayertwo = hiddenlayertwo + 1
            numhiddenlayertwo = str(hiddenlayertwo)
            self.lblhiddenlayertwo.text = numhiddenlayertwo
            if hiddenlayertwo == 10:
                self.btnplusneurontwo.disabled = True
    def btnlessneurontwopressed(self, instance):
        global hiddenlayertwo
        self.btnplusneurontwo.disabled = False
        if hiddenlayertwo <= 1:
            hiddenlayertwo = 1
        else:
            hiddenlayertwo = hiddenlayertwo - 1
            numhiddenlayertwo = str(hiddenlayertwo)
            self.lblhiddenlayertwo.text = numhiddenlayertwo
            if hiddenlayertwo == 1:
                self.btnlessneurontwo.disabled = True

    def btnplusneuronthreepressed(self, instance):
        global hiddenlayerthree
        self.btnlessneuronthree.disabled = False

        if hiddenlayerthree >= 10:
            hiddenlayerthree = 10
        else:
            hiddenlayerthree = hiddenlayerthree + 1
            numhiddenlayerthree = str(hiddenlayerthree)
            self.lblhiddenlayerthree.text = numhiddenlayerthree
            if hiddenlayerthree == 10:
                self.btnplusneuronthree.disabled = True
    def btnlessneuronthreepressed(self, instance):
        global hiddenlayerthree
        self.btnplusneuronthree.disabled = False
        if hiddenlayerthree <= 1:
            hiddenlayerthree = 1
        else:
            hiddenlayerthree = hiddenlayerthree - 1
            numhiddenlayerthree = str(hiddenlayerthree)
            self.lblhiddenlayerthree.text = numhiddenlayerthree
            if hiddenlayerthree == 1:
                self.btnlessneuronthree.disabled = True

    def btnlayerplusneuronpressed(self,instance):
        global layers
        self.btnlayerlessneuron.disabled = False
        if layers >= 3:
            layers = 3
        else:
            layers = layers + 1
            numlayers = str(layers)
            self.lbllayer.text = numlayers
            if layers == 2:
                self.add_widget(self.btnplusneurontwo)
                self.add_widget(self.lblplusbuttontwo)
                self.add_widget(self.btnlessneurontwo)
                self.add_widget(self.lbllessbuttontwo)
                self.add_widget(self.lblhiddenlayertwo)
                self.add_widget(self.lblneurontwo)

                self.btnplusneuron.pos = (290,18)
                self.lblplusbutton.pos = (253,-19)
                self.btnlessneuron.pos = (344,18)
                self.lbllessbutton.pos = (307,-19)
                self.lblhiddenlayer.pos = (280,-20)
                self.lblneuron.pos = (280,10)

                self.btnlayerplusneuron.disabled = False
            if layers == 3:
                self.add_widget(self.btnplusneuronthree)
                self.add_widget(self.lblplusbuttonthree)
                self.add_widget(self.btnlessneuronthree)
                self.add_widget(self.lbllessbuttonthree)
                self.add_widget(self.lblhiddenlayerthree)
                self.add_widget(self.lblneuronthree)
                #button one
                self.btnplusneuron.pos = (235, 18)
                self.lblplusbutton.pos = (198, -19)
                self.btnlessneuron.pos = (289, 18)
                self.lbllessbutton.pos = (252, -19)
                self.lblhiddenlayer.pos = (225, -20)
                self.lblneuron.pos = (225, 10)
                #button two
                self.btnplusneurontwo.pos = (345, 18)
                self.lblplusbuttontwo.pos = (308, -19)
                self.btnlessneurontwo.pos = (399, 18)
                self.lbllessbuttontwo.pos = (362, -19)
                self.lblhiddenlayertwo.pos = (335, -20)
                self.lblneurontwo.pos = (335, 10)

                self.btnlayerplusneuron.disabled = True
    def btnlayerlessneuronpressed(self, instance):
        global layers
        self.btnlayerplusneuron.disabled = False
        if layers <= 1:
            layers =1

        else:
            '''with self.canvas:
                # Background
                Color(0.05, 0.05, 0.05, 1)
                Rectangle(size=(400, 79),pos=(100,0))'''
            layers = layers - 1
            numlayers = str(layers)
            self.lbllayer.text = numlayers
            if layers==2:
                self.remove_widget(self.btnplusneuronthree)
                self.remove_widget(self.lblplusbuttonthree)
                self.remove_widget(self.btnlessneuronthree)
                self.remove_widget(self.lbllessbuttonthree)
                self.remove_widget(self.lblhiddenlayerthree)
                self.remove_widget(self.lblneuronthree)

                self.btnplusneuron.pos = (290, 18)
                self.lblplusbutton.pos = (253, -19)
                self.btnlessneuron.pos = (344, 18)
                self.lbllessbutton.pos = (307, -19)
                self.lblhiddenlayer.pos = (280, -20)
                self.lblneuron.pos = (280, 10)

                self.btnplusneurontwo.pos = (398, 18)
                self.lblplusbuttontwo.pos = (361, -19)
                self.btnlessneurontwo.pos = (452, 18)
                self.lbllessbuttontwo.pos = (415, -19)
                self.lblhiddenlayertwo.pos = (388, -20)
                self.lblneurontwo.pos = (388, 10)

                self.btnlayerlessneuron.disabled = False
            if layers==1:
                self.remove_widget(self.btnplusneurontwo)
                self.remove_widget(self.lblplusbuttontwo)
                self.remove_widget(self.btnlessneurontwo)
                self.remove_widget(self.lbllessbuttontwo)
                self.remove_widget(self.lblhiddenlayertwo)
                self.remove_widget(self.lblneurontwo)

                self.btnplusneuron.pos = (345, 18)
                self.lblplusbutton.pos = (308, -19)
                self.btnlessneuron.pos = (399, 18)
                self.lbllessbutton.pos = (362, -19)
                self.lblhiddenlayer.pos = (335, -20)
                self.lblneuron.pos = (335, 10)

                self.btnlayerlessneuron.disabled = True

    def showbiaslabels(self, instance):
        if self.boxshowbiaslabel.active == True:
            self.remove_widget(self.nodeslabel)

    def btnadjusmentspressed(self, instance):
        First()
    def check_resize(self, instance, x, y):
        self.remove_widget(self)
        print(x)



class MyApp(App):
    def build(self):
        return First()


if __name__ == '__main__':
    MyApp().run()

'''
[{'weights': [-1.3869299223024576, 0.5533519395387496, 0.7492084414852851], 'output': 0.7821549389109715, 'delta': 0.02429160965661226},
 {'weights': [0.6198024909582592, 0.18257887098907608, -0.007792915711487222], 'output': 0.5462496619035293, 'delta': -0.010743041496140476}, 
 {'weights': [0.8776082588562086, 0.37648547195226095, -0.4120711301141641], 'output': 0.4964726367468295, 'delta': -0.021475971070327525}, 
 {'weights': [-1.2567555667187083, 0.6604011876860885, 0.49819620726499286], 'output': 0.7574429428095006, 'delta': 0.019886153513544582}, 
 {'weights': [0.0311622569081563, -0.44735241526602454, 0.22532411266112004], 'output': 0.44369683203059607, 'delta': 0.0041435907052390425}, 
 {'weights': [0.7353553994011518, -0.0451987517770836, 0.674004981521789], 'output': 0.6538779329226149, 'delta': -0.007322949522937171}]
 
[{'weights': [1.3007658104498958, -0.7036980575826011, -1.0330219956423394, 0.9641967178277128, 0.18306814294828025, -0.5104789103527999, -0.03459352141332542], 'output': 0.5991209085508893, 'delta': 0.09628115402399681}, 
{'weights': [0.06782648017587255, -0.48399882682782075, -0.28380500586809804, 0.11934272778253344, -0.050877548678922774, -0.3003139782626832, -0.4278225356327054], 'output': 0.3106658498280419, 'delta': -0.06652989312767253},
{'weights': [-0.9162852190176093, -0.1719723878426436, -0.2780547280317811, -1.073599579165379, 0.38918986675852507, -0.0769766543852941, -0.3439933060823964], 'output': 0.12334523667449426, 'delta': -0.013337467131688176}, 
{'weights': [-1.1229514000136636, 0.4639913275569523, 0.4122285689848092, -1.066089844512006, -0.30351273535827195, -0.02836676758034621, -0.597162002511157], 'output': 0.12395388368054032, 'delta': -0.013460067742033082}]

'''