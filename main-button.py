
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
hiddenlayertwo = hiddenlayer
layers = 1
a = 1
b = 1
c = 300


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
        self.btnplusneuron = Button(size=(25, 25), pos=(343, 18),
                                    background_normal = 'ButtonPlus.png',
                                    background_down = 'ButtonPlusDown.png',
                                    background_disabled_normal = 'ButtonDeactivated.png')
        self.lblplusbutton = Label(text='+', pos=(306,-19), font_size=(20), color=(0,0,0,1))
        self.btnlessneuron = Button(size=(25, 25), pos=(397, 18),
                                    background_normal='ButtonPlus.png',
                                    background_down='ButtonPlusDown.png',
                                    background_disabled_normal = 'ButtonDeactivated.png')
        self.lbllessbutton = Label(text='-', pos=(360, -19), font_size=(20), color=(0, 0, 0, 1))
        self.lblhiddenlayer = Label(text= numhiddenlayer, pos=(333,-20))
        self.lblneuron = Label(text='Neurons', pos=(333,10))

        #Second neuron switcher
        self.btnplusneuron = Button(size=(25, 25), pos=(343, 18),
                                    background_normal='ButtonPlus.png',
                                    background_down='ButtonPlusDown.png',
                                    background_disabled_normal='ButtonDeactivated.png')
        self.lblplusbutton = Label(text='+', pos=(306, -19), font_size=(20), color=(0, 0, 0, 1))
        self.btnlessneuron = Button(size=(25, 25), pos=(397, 18),
                                    background_normal='ButtonPlus.png',
                                    background_down='ButtonPlusDown.png',
                                    background_disabled_normal='ButtonDeactivated.png')
        self.lbllessbutton = Label(text='-', pos=(360, -19), font_size=(20), color=(0, 0, 0, 1))
        self.lblhiddenlayer = Label(text=numhiddenlayer, pos=(333, -20))
        self.lblneuron = Label(text='Neurons', pos=(333, 10))

        #Right neuron switcher

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
        self.btnlayerplusneuron.bind(on_press=self.btnlayerplusneuronpressed)
        self.btnlayerlessneuron.bind(on_press=self.btnlayerlessneuronpressed)


    def btnpressed(self, instance):
        global a
        global b
        global c
        global hiddenlayer
        if layers == 1:
            NeuralNet()
        if layers == 2:
            NeuralNetTwo()
        if layers == 3:
            NeuralNetThree()
        a = 1
        b = 1
        c = 300

        with self.canvas:
            #Clear
            Color(0, 0, 0, 1)
            Rectangle(size=(800,800), pos=(0,81))

        # hiddenlayer
        if hiddenlayer % 2 == 0:
            # Even number
            num = int(hiddenlayer / 2)
            for x in range(num):
                with self.canvas:
                    Color(1, 1, .9, 1)
                    RoundedRectangle(size=(30, 30), pos=(370, a * 50 + 300))
                    RoundedRectangle(size=(30, 30), pos=(370, a * 50 + c - 50))
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
                    a = a + 1
                    c = c - 100


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
                self.remove_widget(self.btnplusneuron)
                self.remove_widget(self.lblplusbutton)
                #self.remove_widget(self.btnlessneuron)
                self.remove_widget(self.lbllessbutton)
                self.btnlessneuron.pos=(100,200)
                self.remove_widget(self.lblneuron)
                self.remove_widget(self.lblhiddenlayer)
                self.btnlayerplusneuron.disabled = False
            if layers == 3:
                self.add_widget(self.btnplusneuron)
                self.add_widget(self.lblplusbutton)
                self.add_widget(self.btnlessneuron)
                self.add_widget(self.lbllessbutton)
                self.add_widget(self.lblhiddenlayer)
                self.add_widget(self.lblneuron)
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
                self.remove_widget(self.btnplusneuron)
                self.remove_widget(self.lblplusbutton)
                self.remove_widget(self.btnlessneuron)
                self.remove_widget(self.lbllessbutton)
                self.remove_widget(self.lblhiddenlayer)
                self.remove_widget(self.lblneuron)
                self.btnlayerlessneuron.disabled = False
            if layers==1:
                self.add_widget(self.btnplusneuron)
                self.add_widget(self.lblplusbutton)
                self.add_widget(self.btnlessneuron)
                self.add_widget(self.lbllessbutton)
                self.add_widget(self.lblhiddenlayer)
                self.add_widget(self.lblneuron)
                self.btnlayerlessneuron.disabled = True



#desactivar botones en neuronlayer y eliminar iconos al cambiar numlayer






class MyApp(App):
    def build(self):
        return First()


if __name__ == '__main__':
    MyApp().run()


