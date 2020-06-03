
from kivy.app import App
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
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

hiddenlayer=7
hiddenlayertwo = 7
hiddenlayerthree = 7
layers = 1
a = 1
b = 1
c = 300
d = 1
e = 1
f = 300
i = 0
j = 1
k = 1
l = 1
m = 300

class NeuralNet():

    global hiddenlayer
    def __init__(self, layers=[13, hiddenlayer, 1], learning_rate=0.001, iterations=100):
        self.params = {}
        self.layers = layers
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.loss = []
        self.sample_size = None
        self.X = None
        self.y = None

    def init_weights(self):
        '''
        Initialize the weights from a random normal distribution
        '''
        np.random.seed(1)  # Seed the random number generator
        self.params["W1"] = np.random.randn(self.layers[0], self.layers[1])
        self.params['b1'] = np.random.randn(self.layers[1], )
        self.params['W2'] = np.random.randn(self.layers[1], self.layers[2])
        self.params['b2'] = np.random.randn(self.layers[2], )


    def relu(self, Z):
        '''
        The ReLufunction performs a threshold
        operation to each input element where values less
        than zero are set to zero.
        '''
        return np.maximum(0, Z)

    def sigmoid(self, Z):
        '''
        The sigmoid function takes in real numbers in any range and
        squashes it to a real-valued output between 0 and 1.
        '''
        return 1.0 / (1.0 + np.exp(-Z))

    def entropy_loss(self, y, yhat):
        nsample = len(y)
        loss = -1 / nsample * (np.sum(np.multiply(np.log(yhat), y) + np.multiply((1 - y), np.log(1 - yhat))))
        return loss

    def forward_propagation(self):
        '''
        Performs the forward propagation
        '''

        Z1 = self.X.dot(self.params['W1']) + self.params['b1']
        A1 = self.relu(Z1)
        Z2 = A1.dot(self.params['W2']) + self.params['b2']
        yhat = self.sigmoid(Z2)
        loss = self.entropy_loss(self.y, yhat)

        # save calculated parameters
        self.params['Z1'] = Z1
        self.params['Z2'] = Z2
        self.params['A1'] = A1

        return yhat, loss

    def back_propagation(self, yhat):
        '''
        Computes the derivatives and update weights and bias according.
        '''

        def dRelu(x):
            x[x <= 0] = 0
            x[x > 0] = 1
            return x

        dl_wrt_yhat = -(np.divide(self.y, yhat) - np.divide((1 - self.y), (1 - yhat)))
        dl_wrt_sig = yhat * (1 - yhat)
        dl_wrt_z2 = dl_wrt_yhat * dl_wrt_sig

        dl_wrt_A1 = dl_wrt_z2.dot(self.params['W2'].T)
        dl_wrt_w2 = self.params['A1'].T.dot(dl_wrt_z2)
        dl_wrt_b2 = np.sum(dl_wrt_z2, axis=0)

        dl_wrt_z1 = dl_wrt_A1 * dRelu(self.params['Z1'])
        dl_wrt_w1 = self.X.T.dot(dl_wrt_z1)
        dl_wrt_b1 = np.sum(dl_wrt_z1, axis=0)

        # update the weights and bias
        self.params['W1'] = self.params['W1'] - self.learning_rate * dl_wrt_w1
        self.params['W2'] = self.params['W2'] - self.learning_rate * dl_wrt_w2
        self.params['b1'] = self.params['b1'] - self.learning_rate * dl_wrt_b1
        self.params['b2'] = self.params['b2'] - self.learning_rate * dl_wrt_b2

    def fit(self, X, y):
        '''
        Trains the neural network using the specified data and labels
        '''
        self.X = X
        self.y = y
        self.init_weights()  # initialize weights and bias

        for i in range(self.iterations):
            yhat, loss = self.forward_propagation()
            self.back_propagation(yhat)
            self.loss.append(loss)

    def predict(self, X):
        '''
        Predicts on a test data
        '''
        Z1 = X.dot(self.params['W1']) + self.params['b1']
        A1 = self.relu(Z1)
        Z2 = A1.dot(self.params['W2']) + self.params['b2']
        pred = self.sigmoid(Z2)
        return np.round(pred)

    def acc(self, y, yhat):
        '''
        Calculates the accutacy between the predicted valuea and the truth labels
        '''
        acc = int(sum(y == yhat) / len(y) * 100)
        return acc

    def plot_loss(self):
        '''
        Plots the loss curve
        '''
        plt.plot(self.loss)
        plt.xlabel("Iteration")
        plt.ylabel("logloss")
        plt.title("Loss curve for training")
        plt.show()

class NeuralNetTwo():

    global hiddenlayer
    def __init__(self, layers=[13, hiddenlayer, hiddenlayertwo, 1], learning_rate=0.001, iterations=100):
        self.params = {}
        self.layers = layers
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.loss = []
        self.sample_size = None
        self.X = None
        self.y = None

    def init_weights(self):
        '''
        Initialize the weights from a random normal distribution
        '''
        np.random.seed(1)  # Seed the random number generator
        self.params["W1"] = np.random.randn(self.layers[0], self.layers[1])
        self.params['b1'] = np.random.randn(self.layers[1], )
        self.params['W2'] = np.random.randn(self.layers[1], self.layers[2])
        self.params['b2'] = np.random.randn(self.layers[2], )
        self.params['W3'] = np.random.randn(self.layers[2], self.layers[3])
        self.params['b3'] = np.random.randn(self.layers[3], )

    def relu(self, Z):
        '''
        The ReLufunction performs a threshold
        operation to each input element where values less
        than zero are set to zero.
        '''
        return np.maximum(0, Z)

    def sigmoid(self, Z):
        '''
        The sigmoid function takes in real numbers in any range and
        squashes it to a real-valued output between 0 and 1.
        '''
        return 1.0 / (1.0 + np.exp(-Z))

    def entropy_loss(self, y, yhat):
        #analizes how good or bad it is to be able to change it accordingly
        nsample = len(y)
        loss = -1 / nsample * (np.sum(np.multiply(np.log(yhat), y) + np.multiply((1 - y), np.log(1 - yhat))))
        return loss

    def forward_propagation(self):
        '''
        Performs the forward propagation
        '''

        Z1 = self.X.dot(self.params['W1']) + self.params['b1']
        A1 = self.relu(Z1)
        Z2 = A1.dot(self.params['W2']) + self.params['b2']
        yhat = self.sigmoid(Z2)
        Z3 = A2.dot(self.params['W3']) + self.params['b3']
        yhat = self.sigmoid(Z2)
        loss = self.entropy_loss(self.y, yhat)

        # save calculated parameters
        self.params['Z1'] = Z1
        self.params['Z2'] = Z2
        self.params['A1'] = A1
        self.params['Z3'] = Z3
        self.params['A2'] = A2

        return yhat, loss

    def back_propagation(self, yhat):
        '''
        Computes the derivatives and update weights and bias according.
        '''

        def dRelu(x):
            x[x <= 0] = 0
            x[x > 0] = 1
            return x
#No sé cambiar esta parte de aquí
        dl_wrt_yhat = -(np.divide(self.y, yhat) - np.divide((1 - self.y), (1 - yhat)))
        dl_wrt_sig = yhat * (1 - yhat)
        dl_wrt_z2 = dl_wrt_yhat * dl_wrt_sig

        dl_wrt_A1 = dl_wrt_z2.dot(self.params['W2'].T)
        dl_wrt_w2 = self.params['A1'].T.dot(dl_wrt_z2)
        dl_wrt_b2 = np.sum(dl_wrt_z2, axis=0)

        dl_wrt_z1 = dl_wrt_A1 * dRelu(self.params['Z1'])
        dl_wrt_w1 = self.X.T.dot(dl_wrt_z1)
        dl_wrt_b1 = np.sum(dl_wrt_z1, axis=0)

        # update the weights and bias
        self.params['W1'] = self.params['W1'] - self.learning_rate * dl_wrt_w1
        self.params['W2'] = self.params['W2'] - self.learning_rate * dl_wrt_w2
        self.params['W3'] = self.params['W3'] - self.learning_rate * dl_wrt_w3
        self.params['b1'] = self.params['b1'] - self.learning_rate * dl_wrt_b1
        self.params['b2'] = self.params['b2'] - self.learning_rate * dl_wrt_b2
        self.params['b3'] = self.params['b3'] - self.learning_rate * dl_wrt_b3

    def fit(self, X, y):
        '''
        Trains the neural network using the specified data and labels
        '''
        self.X = X
        self.y = y
        self.init_weights()  # initialize weights and bias

        for i in range(self.iterations):
            yhat, loss = self.forward_propagation()
            self.back_propagation(yhat)
            self.loss.append(loss)

    def predict(self, X):
        '''
        Predicts on a test data
        '''
#No estoy segura de cómo cambiarlo. Creo que está bien
        Z1 = X.dot(self.params['W1']) + self.params['b1']
        A1 = self.relu(Z1)
        Z2 = A1.dot(self.params['W2']) + self.params['b2']
        A2 = self.relu(Z2)
        Z3 = A2.dot(self.params['W3']) + self.params['b3']
        pred = self.sigmoid(Z3)

        return np.round(pred)

    def acc(self, y, yhat):
        '''
        Calculates the accutacy between the predicted valuea and the truth labels
        '''
        acc = int(sum(y == yhat) / len(y) * 100)
        return acc

    def plot_loss(self):
        '''
        Plots the loss curve
        '''
        plt.plot(self.loss)
        plt.xlabel("Iteration")
        plt.ylabel("logloss")
        plt.title("Loss curve for training")
        plt.show()

class NeuralNetThree():
    global hiddenlayer

    def __init__(self, layers=[13, hiddenlayer, hiddenlayertwo, 1], learning_rate=0.001, iterations=100):
        self.params = {}
        self.layers = layers
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.loss = []
        self.sample_size = None
        self.X = None
        self.y = None

    def init_weights(self):
        '''
        Initialize the weights from a random normal distribution
        '''
        np.random.seed(1)  # Seed the random number generator
        self.params["W1"] = np.random.randn(self.layers[0], self.layers[1])
        self.params['b1'] = np.random.randn(self.layers[1], )
        self.params['W2'] = np.random.randn(self.layers[1], self.layers[2])
        self.params['b2'] = np.random.randn(self.layers[2], )
        self.params['W3'] = np.random.randn(self.layers[2], self.layers[3])
        self.params['b3'] = np.random.randn(self.layers[3], )

    def relu(self, Z):
        '''
        The ReLufunction performs a threshold
        operation to each input element where values less
        than zero are set to zero.
        '''
        return np.maximum(0, Z)

    def sigmoid(self, Z):
        '''
        The sigmoid function takes in real numbers in any range and
        squashes it to a real-valued output between 0 and 1.
        '''
        return 1.0 / (1.0 + np.exp(-Z))

    def entropy_loss(self, y, yhat):
        # analizes how good or bad it is to be able to change it accordingly
        nsample = len(y)
        loss = -1 / nsample * (np.sum(np.multiply(np.log(yhat), y) + np.multiply((1 - y), np.log(1 - yhat))))
        return loss

    def forward_propagation(self):
        '''
        Performs the forward propagation
        '''

        Z1 = self.X.dot(self.params['W1']) + self.params['b1']
        A1 = self.relu(Z1)
        Z2 = A1.dot(self.params['W2']) + self.params['b2']
        yhat = self.sigmoid(Z2)
        Z3 = A2.dot(self.params['W3']) + self.params['b3']
        yhat = self.sigmoid(Z2)
        loss = self.entropy_loss(self.y, yhat)

        # save calculated parameters
        self.params['Z1'] = Z1
        self.params['Z2'] = Z2
        self.params['A1'] = A1
        self.params['Z3'] = Z3
        self.params['A2'] = A2

        return yhat, loss

    def back_propagation(self, yhat):
        '''
        Computes the derivatives and update weights and bias according.
        '''

        def dRelu(x):
            x[x <= 0] = 0
            x[x > 0] = 1
            return x

        # No sé cambiar esta parte de aquí
        dl_wrt_yhat = -(np.divide(self.y, yhat) - np.divide((1 - self.y), (1 - yhat)))
        dl_wrt_sig = yhat * (1 - yhat)
        dl_wrt_z2 = dl_wrt_yhat * dl_wrt_sig

        dl_wrt_A1 = dl_wrt_z2.dot(self.params['W2'].T)
        dl_wrt_w2 = self.params['A1'].T.dot(dl_wrt_z2)
        dl_wrt_b2 = np.sum(dl_wrt_z2, axis=0)

        dl_wrt_z1 = dl_wrt_A1 * dRelu(self.params['Z1'])
        dl_wrt_w1 = self.X.T.dot(dl_wrt_z1)
        dl_wrt_b1 = np.sum(dl_wrt_z1, axis=0)

        # update the weights and bias
        self.params['W1'] = self.params['W1'] - self.learning_rate * dl_wrt_w1
        self.params['W2'] = self.params['W2'] - self.learning_rate * dl_wrt_w2
        self.params['W3'] = self.params['W3'] - self.learning_rate * dl_wrt_w3
        self.params['b1'] = self.params['b1'] - self.learning_rate * dl_wrt_b1
        self.params['b2'] = self.params['b2'] - self.learning_rate * dl_wrt_b2
        self.params['b3'] = self.params['b3'] - self.learning_rate * dl_wrt_b3

    def fit(self, X, y):
        '''
        Trains the neural network using the specified data and labels
        '''
        self.X = X
        self.y = y
        self.init_weights()  # initialize weights and bias

        for i in range(self.iterations):
            yhat, loss = self.forward_propagation()
            self.back_propagation(yhat)
            self.loss.append(loss)

    def predict(self, X):
        '''
        Predicts on a test data
        '''
        # No estoy segura de cómo cambiarlo. Creo que está bien
        Z1 = X.dot(self.params['W1']) + self.params['b1']
        A1 = self.relu(Z1)
        Z2 = A1.dot(self.params['W2']) + self.params['b2']
        A2 = self.relu(Z2)
        Z3 = A2.dot(self.params['W3']) + self.params['b3']
        pred = self.sigmoid(Z3)

        return np.round(pred)

    def acc(self, y, yhat):
        '''
        Calculates the accutacy between the predicted valuea and the truth labels
        '''
        acc = int(sum(y == yhat) / len(y) * 100)
        return acc

    def plot_loss(self):
        '''
        Plots the loss curve
        '''
        plt.plot(self.loss)
        plt.xlabel("Iteration")
        plt.ylabel("logloss")
        plt.title("Loss curve for training")
        plt.show()

class First(Widget):
    def __init__(self,**kwargs):
        super(First, self).__init__(**kwargs)
        global hiddenlayer
        global layers
        numhiddenlayer = str(hiddenlayer)
        numhiddenlayertwo = str(hiddenlayertwo)
        numhiddenlayerthree = str(hiddenlayerthree)
        numlayers = str(layers)

        with self.canvas:
            #Background
            Color(0.05, 0.05, 0.05, 1)
            Rectangle(size=(800,600))
            Color(0, 0, 0, 1)
            Rectangle(size=(800, 800), pos=(0, 81))
            Color(1,1,1,1)
            Line(points=[0,80,900,80])
            Color(1, 1, 1, 1)
            Line(points=[0, 80, 900, 80])

        self.btn = Button(text='GO', size = (80,30), pos=(25,15), background_color = (.8, .8, .8, 1))
        self.box = CheckBox(pos=(45, 10))
        self.lblbox = Label(text='Sigmoid:', pos=(4, 10), color=(1, 1, 1, 1), font_size=(14))

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

        self.add_widget(self.btn)
        self.add_widget(self.btnplusneuron)
        self.add_widget(self.lblplusbutton)
        self.add_widget(self.btnlessneuron)
        self.add_widget(self.lbllessbutton)
        self.add_widget(self.btnlayerlessneuron)
        self.add_widget(self.btnlayerplusneuron)
        self.add_widget(self.lbllayerlessbutton)
        self.add_widget(self.lbllayerplusbutton)
        #self.add_widget(self.box)
        self.add_widget(self.lblbox)
        self.add_widget(self.lblhiddenlayer)
        self.add_widget(self.lblneuron)
        self.add_widget(self.lblnumlayers)
        self.add_widget(self.lbllayer)


        self.box.active = True
        self.btn.bind(on_press=self.btnpressed)
        self.btnplusneuron.bind(on_press=self.btnplusneuronpressed)
        self.btnlessneuron.bind(on_press=self.btnlessneuronpressed)
        self.btnplusneurontwo.bind(on_press=self.btnplusneurontwopressed)
        self.btnlessneurontwo.bind(on_press=self.btnlessneurontwopressed)
        self.btnplusneuronthree.bind(on_press=self.btnplusneuronthreepressed)
        self.btnlessneuronthree.bind(on_press=self.btnlessneuronthreepressed)
        self.btnlayerplusneuron.bind(on_press=self.btnlayerplusneuronpressed)
        self.btnlayerlessneuron.bind(on_press=self.btnlayerlessneuronpressed)


    def btnpressed(self, instance):
        global a
        global b
        global c
        global hiddenlayer
        a = 1
        b = 1
        c = 300
        d = 1
        f = 300
        g = 1
        h = 300
        i = 0
        j = 1
        k = 300
        l = 1

        with self.canvas:
            #Clear
            Color(0, 0, 0, 1)
            Rectangle(size=(800,800), pos=(0,81))

        if layers == 1:
            NeuralNet()
            # hiddenlayer
            if hiddenlayer % 2 == 0:
                # Even number
                num = int(hiddenlayer / 2)
                for x in range(num):
                    with self.canvas:
                        Color(1, 1, .9, 1)
                        Line(points=[385,a * 50 + 315, 605,340])
                        Line(points=[385, a * 50 + c - 35, 605, 340])
                        RoundedRectangle(size=(30, 30), pos=(370, a * 50 + 300))
                        RoundedRectangle(size=(30, 30), pos=(370, a * 50 + c - 50))
                        i = 0
                        for x in range(10):
                            with self.canvas:
                                Line(points=[180, 115+i, 370, a * 50 + 315])
                                Line(points=[180, 115+i, 370, a * 50 + c - 35])
                                i = i + 50
                        a = a + 1
                        c = c - 100
            else:
                # Odd number
                num = int((hiddenlayer - 1) / 2)
                with self.canvas:
                    Color(1, 1, .9, 1)
                    RoundedRectangle(size=(30, 30), pos=(370, 325))
                for x in range(num):
                    with self.canvas:
                        Color(1, 1, .9, 1)
                        RoundedRectangle(size=(30, 30), pos=(370, a * 50 + 330))
                        RoundedRectangle(size=(30, 30), pos=(370, a * 50 + c - 80))
                        Line(points=[385, a * 50 + 345, 605, 340])
                        Line(points=[385, a * 50 + c - 65, 605, 340])
                        Line(points=[385, 340, 605, 340])
                        i = 0
                        for x in range(10):
                            with self.canvas:
                                Line(points=[180, 115 + i, 370, a * 50 + 345])
                                Line(points=[180, 115 + i, 370, a * 50 + c - 65])
                                Line(points=[180, 115 + i, 370, 340])
                                i = i + 50
                        a = a + 1
                        c = c - 100

        if layers == 2:
            NeuralNetTwo()
            # hiddenlayerone
            if hiddenlayer % 2 == 0:
                # Even number
                num = int(hiddenlayer / 2)
                for x in range(num):
                    with self.canvas:
                        Color(1, 1, .9, 1)
                        RoundedRectangle(size=(30, 30), pos=(315, a * 50 + 300))
                        RoundedRectangle(size=(30, 30), pos=(315, a * 50 + c - 50))
                        i = 0
                        for x in range(10):
                            with self.canvas:
                                Line(points=[180, 115+i, 315, a * 50 + 315])
                                Line(points=[180, 115+i, 315, a * 50 + c - 35])
                            i = i + 50
                        a = a + 1
                        c = c - 100
            else:
                # Odd number
                num = int((hiddenlayer - 1) / 2)
                with self.canvas:
                    Color(1, 1, .9, 1)
                    RoundedRectangle(size=(30, 30), pos=(315, 325))
                for x in range(num):
                    with self.canvas:
                        Color(1, 1, .9, 1)
                        RoundedRectangle(size=(30, 30), pos=(315, a * 50 + 330))
                        RoundedRectangle(size=(30, 30), pos=(315, a * 50 + c - 80))
                        i = 0
                        for x in range(10):
                            with self.canvas:
                                Line(points=[180, 115 + i, 315, a * 50 + 345])
                                Line(points=[180, 115 + i, 315, a * 50 + c - 65])
                                Line(points=[180, 115 + i, 315, 340])
                                i = i + 50
                        a = a + 1
                        c = c - 100

            # hiddenlayertwo
            if hiddenlayertwo % 2 == 0:
                # Even number
                num = int(hiddenlayertwo / 2)
                num2 = int(hiddenlayer / 2)

                for x in range(num):
                    with self.canvas:
                        Color(1, 1, .9, 1)
                        RoundedRectangle(size=(30, 30), pos=(425, d * 50 + 300))
                        RoundedRectangle(size=(30, 30), pos=(425, d * 50 + f - 50))
                        Line(points=[455, d * 50 + 315, 605, 340])
                        Line(points=[455, d * 50 + f - 35, 605, 340])

                        j = 1
                        k = 300
                        if hiddenlayer % 2 == 0:
                            for x in range(num2):
                                 with self.canvas:
                                    Line(points=[345, j * 50 + 315, 425, d * 50 + 315])
                                    Line(points=[345, j * 50 + k - 35, 425, d * 50 + f - 35])
                                    Line(points=[345, j * 50 + 315, 425, d * 50 + f - 35])
                                    Line(points=[345, j * 50 + k - 35, 425, d * 50 + 315])
                                    j = j + 1
                                    k = k - 100
                        else:
                            for x in range(num2):
                                 with self.canvas:
                                    Line(points=[345, j * 50 + 345, 425, d * 50 + 315])
                                    Line(points=[345, j * 50 + k - 65, 425, d * 50 + f - 35])
                                    Line(points=[345, j * 50 + 345, 425, d * 50 + f - 35])
                                    Line(points=[345, j * 50 + k - 65, 425, d * 50 + 315])
                                    Line(points=[345, 345, 425, d * 50 + 315])
                                    Line(points=[345, 345, 425, d * 50 + f - 35])
                                    j = j + 1
                                    k = k - 100
                    d = d + 1
                    f = f - 100
            else:
                # Odd number
                num = int((hiddenlayertwo - 1) / 2)
                num2 = int(hiddenlayer / 2)
                with self.canvas:
                    Color(1, 1, .9, 1)
                    RoundedRectangle(size=(30, 30), pos=(425, 325))
                    Line(points=[455, 340, 605, 340])
                for x in range(num):
                    with self.canvas:
                        Color(1, 1, .9, 1)
                        RoundedRectangle(size=(30, 30), pos=(425, d * 50 + 330))
                        RoundedRectangle(size=(30, 30), pos=(425, d * 50 + f - 80))
                        Line(points=[455,d * 50 + 345, 605, 340])
                        Line(points=[455, d * 50 + f - 65, 605, 340])
                        d = d + 1
                        f = f - 100
                        j = 1
                        k = 300
                        if hiddenlayer % 2 == 0:
                            for x in range(num2):
                                with self.canvas:
                                    Line(points=[345, j * 50 + 315, 425, d * 50 + 300])
                                    Line(points=[345, j * 50 + k - 35, 425, d * 50 + f - 20])
                                    Line(points=[345, j * 50 + 315, 425, d * 50 + f - 20])
                                    Line(points=[345, j * 50 + k - 35, 425, d * 50 + 300])
                                    Line(points=[345, j * 50 + k - 35, 425, 340])
                                    Line(points=[345, j * 50 + 315, 425, 340])
                                    j = j + 1
                                    k = k - 100
                        else:
                            for x in range(num2):
                                with self.canvas:
                                    Line(points=[345, j * 50 + 345, 425, d * 50 + 300])
                                    Line(points=[345, j * 50 + k - 65, 425, d * 50 + f - 20])
                                    Line(points=[345, j * 50 + 345, 425, d * 50 + f - 20])
                                    Line(points=[345, j * 50 + k - 65, 425, d * 50 + 300])
                                    Line(points=[345, 345, 425, d * 50 + 300])
                                    Line(points=[345, 345, 425, d * 50 + f - 20])
                                    Line(points=[345, j * 50 + k - 65, 425, 340])
                                    Line(points=[345, j * 50 + 345, 425, 340])
                                    Line(points=[345, 345, 425, 340])
                                    j = j + 1
                                    k = k - 100

        if layers == 3:
            NeuralNetThree()
            # hiddenlayerone
            if hiddenlayer % 2 == 0:
                # Even number
                num = int(hiddenlayer / 2)
                for x in range(num):
                    with self.canvas:
                        Color(1, 1, .9, 1)
                        RoundedRectangle(size=(30, 30), pos=(260, a * 50 + 300))
                        RoundedRectangle(size=(30, 30), pos=(260, a * 50 + c - 50))
                        i = 0
                        for x in range(10):
                            with self.canvas:
                                Line(points=[180, 115 + i, 260, a * 50 + 315])
                                Line(points=[180, 115 + i, 260, a * 50 + c - 35])
                            i = i + 50
                        a = a + 1
                        c = c - 100
            else:
                # Odd number
                num = int((hiddenlayer - 1) / 2)
                with self.canvas:
                    Color(1, 1, .9, 1)
                    RoundedRectangle(size=(30, 30), pos=(260, 325))
                for x in range(num):
                    with self.canvas:
                        Color(1, 1, .9, 1)
                        RoundedRectangle(size=(30, 30), pos=(260, a * 50 + 330))
                        RoundedRectangle(size=(30, 30), pos=(260, a * 50 + c - 80))
                        i = 0
                        for x in range(10):
                            with self.canvas:
                                Line(points=[180, 115 + i, 260, a * 50 + 345])
                                Line(points=[180, 115 + i, 260, a * 50 + c - 65])
                                Line(points=[180, 115 + i, 260, 340])
                                i = i + 50
                        a = a + 1
                        c = c - 100

            # hiddenlayertwo
            if hiddenlayertwo % 2 == 0:
                # Even number
                num = int(hiddenlayertwo / 2)
                num2 = int(hiddenlayer / 2)
                for x in range(num):
                    with self.canvas:
                        Color(1, 1, .9, 1)
                        RoundedRectangle(size=(30, 30), pos=(370, d * 50 + 300))
                        RoundedRectangle(size=(30, 30), pos=(370, d * 50 + f - 50))
                        j = 1
                        k = 300
                        if hiddenlayer % 2 == 0:
                            for x in range(num2):
                                with self.canvas:
                                    Line(points=[290, j * 50 + 315, 370, d * 50 + 315])
                                    Line(points=[290, j * 50 + k - 35, 370, d * 50 + f - 35])
                                    Line(points=[290, j * 50 + 315, 370, d * 50 + f - 35])
                                    Line(points=[290, j * 50 + k - 35, 370, d * 50 + 315])
                                    j = j + 1
                                    k = k - 100
                        else:
                            for x in range(num2):
                                with self.canvas:
                                    Line(points=[290, j * 50 + 345, 370, d * 50 + 315])
                                    Line(points=[290, j * 50 + k - 65, 370, d * 50 + f - 35])
                                    Line(points=[290, j * 50 + 345, 370, d * 50 + f - 35])
                                    Line(points=[290, j * 50 + k - 65, 370, d * 50 + 315])
                                    Line(points=[290, 345, 370, d * 50 + 315])
                                    Line(points=[290, 345, 370, d * 50 + f - 35])
                                    j = j + 1
                                    k = k - 100
                        d = d + 1
                        f = f - 100
            else:
                # Odd number
                num = int((hiddenlayertwo - 1) / 2)
                num2 = int(hiddenlayer / 2)
                with self.canvas:
                    Color(1, 1, .9, 1)
                    RoundedRectangle(size=(30, 30), pos=(370, 325))
                for x in range(num):
                    with self.canvas:
                        Color(1, 1, .9, 1)
                        RoundedRectangle(size=(30, 30), pos=(370, d * 50 + 330))
                        RoundedRectangle(size=(30, 30), pos=(370, d * 50 + f - 80))
                        d = d + 1
                        f = f - 100
                        j = 1
                        k = 300
                        if hiddenlayer % 2 == 0:
                            for x in range(num2):
                                with self.canvas:
                                    Line(points=[290, j * 50 + 315, 370, d * 50 + 300])
                                    Line(points=[290, j * 50 + k - 35, 370, d * 50 + f - 20])
                                    Line(points=[290, j * 50 + 315, 370, d * 50 + f - 20])
                                    Line(points=[290, j * 50 + k - 35, 370, d * 50 + 300])
                                    Line(points=[290, j * 50 + k - 35, 370, 340])
                                    Line(points=[290, j * 50 + 315, 370, 340])
                                    j = j + 1
                                    k = k - 100
                        else:
                            for x in range(num2):
                                with self.canvas:
                                    Line(points=[290, j * 50 + 345, 370, d * 50 + 300])
                                    Line(points=[290, j * 50 + k - 65, 370, d * 50 + f - 20])
                                    Line(points=[290, j * 50 + 345, 370, d * 50 + f - 20])
                                    Line(points=[290, j * 50 + k - 65, 370, d * 50 + 300])
                                    Line(points=[290, 345, 370, d * 50 + 300])
                                    Line(points=[290, 345, 370, d * 50 + f - 20])
                                    Line(points=[290, j * 50 + k - 65, 370, 340])
                                    Line(points=[290, j * 50 + 345, 370, 340])
                                    Line(points=[290, 345, 370, 340])
                                    j = j + 1
                                    k = k - 100




            # hiddenlayerthree
            if hiddenlayerthree % 2 == 0:
                # Even number
                num = int(hiddenlayerthree / 2)
                num3 = int(hiddenlayertwo / 2)
                for x in range(num):
                    with self.canvas:
                        Color(1, 1, .9, 1)
                        RoundedRectangle(size=(30, 30), pos=(480, g * 50 + 300))
                        RoundedRectangle(size=(30, 30), pos=(480, g * 50 + h - 50))
                        Line(points=[510, g * 50 + 315, 590, 340])
                        Line(points=[510, g * 50 + h - 35, 590, 340])
                        l = 1
                        m = 300
                        if hiddenlayertwo % 2 == 0:
                            for x in range(num3):
                                with self.canvas:
                                    Line(points=[385, l * 50 + 315, 480, g * 50 + 315])
                                    Line(points=[385, l * 50 + m - 35, 480, g * 50 + h - 35])
                                    Line(points=[385, l * 50 + 315, 480, g * 50 + h - 35])
                                    Line(points=[385, l * 50 + m - 35, 480, g * 50 + 315])
                                    l = l + 1
                                    m = m - 100
                        else:
                            for x in range(num3):
                                with self.canvas:
                                    Line(points=[385, l * 50 + 345, 480, g * 50 + 315])
                                    Line(points=[385, l * 50 + m - 65, 480, g * 50 + h - 35])
                                    Line(points=[385, l * 50 + 345, 480, g * 50 + h - 35])
                                    Line(points=[385, l * 50 + m - 65, 480, g * 50 + 315])
                                    Line(points=[385, 345, 480, g * 50 + 315])
                                    Line(points=[385, 345, 480, g * 50 + h - 35])
                                    l = l + 1
                                    m = m - 100
                        g = g + 1
                        h = h - 100
            else:
                # Odd number
                num = int((hiddenlayerthree - 1) / 2)
                num3 = int(hiddenlayertwo / 2)
                with self.canvas:
                    Color(1, 1, .9, 1)
                    RoundedRectangle(size=(30, 30), pos=(480, 325))
                    Line(points=[510, 340, 590, 340])
                for x in range(num):
                    with self.canvas:
                        Color(1, 1, .9, 1)
                        RoundedRectangle(size=(30, 30), pos=(480, g * 50 + 330))
                        RoundedRectangle(size=(30, 30), pos=(480, g * 50 + h - 80))
                        Line(points=[510, g * 50 + 345, 590, 340])
                        Line(points=[510, g * 50 + h - 65, 590, 340])
                        l = 1
                        m = 300
                        if hiddenlayertwo % 2 == 0:
                            for x in range(num3):
                                with self.canvas:
                                    Line(points=[400, l * 50 + 315, 480, g * 50 + 345])
                                    Line(points=[400, l * 50 + m - 35, 480, g * 50 + h - 65])
                                    Line(points=[400, l * 50 + 315, 480, g * 50 + h - 65])
                                    Line(points=[400, l * 50 + m - 35, 480, g * 50 + 345])
                                    Line(points=[400, l * 50 + 315, 480, 340])
                                    Line(points=[400, l * 50 + m - 35, 480, 340])
                                    l = l + 1
                                    m = m - 100
                        else:
                            for x in range(num3):
                                with self.canvas:
                                    Line(points=[400, l * 50 + 345, 480, g * 50 + 345])
                                    Line(points=[400, l * 50 + m - 65, 480, g * 50 + h - 65])
                                    Line(points=[400, l * 50 + 345, 480, g * 50 + h - 65])
                                    Line(points=[400, l * 50 + m - 65, 480, g * 50 + 345])
                                    Line(points=[400, 345, 480, g * 50 + 345])
                                    Line(points=[400, 345, 480, g * 50 + h - 65])
                                    Line(points=[400, l * 50 + 345, 480, 340])
                                    Line(points=[400, l * 50 + m - 65, 480, 340])
                                    l = l + 1
                                    m = m - 100
                        g = g + 1
                        h = h - 100

        #output layer
        with self.canvas:
            Color(1, 1, .9, 1)
            RoundedRectangle(size=(30, 30), pos=(590,325))

        # input layer
        for x in range(13):

            b = b + 1
            with self.canvas:
                Color(1, 1, .9, 1)
                RoundedRectangle(size=(30, 30), pos=(150,b*50))

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
            layer = 3
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






class MyApp(App):
    def build(self):
        return First()


if __name__ == '__main__':
    MyApp().run()


