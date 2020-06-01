

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

class NeuralNetwork():

    def __init__(self):
        # seeding for random number generation
        np.random.seed(1)

        # converting weights to a 3 by 1 matrix with values from -1 to 1 and mean of 0
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    def sigmoid(self, x):
        # applying the sigmoid function
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        # computing derivative to the Sigmoid function
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        # training the model to make accurate predictions while adjusting weights continually
        for iteration in range(training_iterations):
            # siphon the training data via  the neuron
            output = self.think(training_inputs)

            # computing error rate for back-propagation
            error = training_outputs - output

            # performing weight adjustments
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))

            self.synaptic_weights += adjustments

    def think(self, inputs):
        # passing the inputs via the neuron to get output
        # converting values to floats

        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output

class First(Widget):
    def __init__(self,**kwargs):
        super(First, self).__init__(**kwargs)
        self.inputnumber = Label(pos=(70, 100), text='', font_size = 10)
        self.startingweights = Label(pos=(50, 50), text='', font_size = 10)
        self.labelfinaldata = Label(pos=(550, 100), text='', font_size = 10)
        self.labelfinaldatanumber = Label(pos=(550, 80), text='', font_size = 10)
        self.finalnumber = Label(pos=(450, 290), text='')
        self.finalweights = Label(pos=(350, 100), text='', font_size = 10)
        self.finalweightsnumber = Label(pos=(350, 50), text='', font_size = 10)
        self.inputlabelone = TextInput(size=(80,40),pos = (10,400))
        self.inputlabeltwo = TextInput(size=(80,40),pos = (10,325))
        self.inputlabelthree = TextInput(size=(80,40),pos = (10,250))
        self.btn = Button(size=(90,40), pos = (10,10), text= 'GO', background_color = (1,0.5,1,1))
        self.add_widget(self.inputnumber)
        self.add_widget(self.finalnumber)
        self.add_widget(self.finalweights)
        self.add_widget(self.finalweightsnumber)
        self.add_widget(self.labelfinaldata)
        self.add_widget(self.labelfinaldatanumber)
        self.add_widget(self.startingweights)
        self.add_widget(self.inputlabelone)
        self.add_widget(self.inputlabeltwo)
        self.add_widget(self.inputlabelthree)
        self.add_widget(self.btn)
        self.btn.bind(on_press=self.btnpressed)
        with self.canvas:
            Color(1,1,1,0.5)
            Rectangle(size=(40, 40), pos=(130, 400))
            Rectangle(size=(40, 40), pos=(130, 325))
            Rectangle(size=(40, 40), pos=(130, 250))
            weightone= Rectangle(size=(40, 40), pos=(250, 438))
            weighttwo= Rectangle(size=(40, 40), pos=(250, 363))
            weightthree= Rectangle(size=(40, 40), pos=(250, 287))
            weightfour = Rectangle(size=(40, 40), pos=(250, 212))
            finalnumberrectangle = Rectangle(size=(40, 40), pos=(370, 325))


    def btnpressed(self, instance):

        # initializing the neuron class
        neural_network = NeuralNetwork()

        self.inputnumber.text ="Beginning Randomly Generated Weights: "
        self.startingweights.text = str(neural_network.synaptic_weights.astype(float))

        # training data consisting of 4 examples--3 input values and 1 output
        training_inputs = np.array([[0, 0, 1],
                                    [1, 1, 1],
                                    [1, 0, 1],
                                    [0, 1, 1]])

        training_outputs = np.array([[0, 1, 1, 0]]).T

        # training taking place
        neural_network.train(training_inputs, training_outputs, 15000)



        weightzero = (neural_network.synaptic_weights[0])
        weightone = (neural_network.synaptic_weights[1])
        weighttwo = (neural_network.synaptic_weights[2])




        self.finalweights.text = "Ending Weights After Training: "
        '''auxstr = neural_network.synaptic_weights.tostring()
        self.inputnumber.text = auxstr'''

        self.finalweightsnumber.text = str(neural_network.synaptic_weights)


        user_input_one = int(self.inputlabelone.text, base=10)
        user_input_two = int(self.inputlabeltwo.text, base=10)
        user_input_three = int(self.inputlabelthree.text, base=10)

        self.labelfinaldata.text = "New Situation:"
        self.labelfinaldatanumber.text = str(user_input_one) + ", " + str(user_input_two)+ " ," + str(user_input_three)
        neural_output = neural_network.think(np.array([self.inputlabelone.text, self.inputlabeltwo.text, self.inputlabelthree.text]))
        self.finalnumber.text = str(neural_output.astype(float))

        colorone = neural_output.astype


        print(colorone)

        with self.canvas:
            Color(0, 0, 0, 1)
            Rectangle(size=(40, 40), pos=(370, 325))
            Color(1, 1, 1,1)
            Rectangle(size=(40, 40), pos=(370, 325))
            Color(1,.8,1,1)
            Line(points=[150, 420, 270, 458])
            Line(points=[150, 420, 270, 383])
            Line(points=[150, 420, 270, 307])
            Line(points=[150, 420, 270, 232])
            Line(points=[150, 345, 270, 458])
            Line(points=[150, 345, 270, 383])
            Line(points=[150, 345, 270, 307])
            Line(points=[150, 345, 270, 232])
            Line(points=[150, 270, 270, 458])
            Line(points=[150, 270, 270, 383])
            Line(points=[150, 270, 270, 307])
            Line(points=[150, 270, 270, 232])
            Line(points=[270, 232, 390, 345])
            Line(points=[270, 307, 390, 345])
            Line(points=[270, 383, 390, 345])
            Line(points=[270, 458, 390, 345])
            Color(0,0,0,1)
            Rectangle(size=(40, 40), pos=(130, 400), color=(0,0,0,1))
            Rectangle(size=(40, 40), pos=(130, 325), color=(0,0,0,1))
            Rectangle(size=(40, 40), pos=(130, 250), color=(0,0,0,1))
            Color(1,1,1,(abs(weightzero) / 10))
            Rectangle(size=(40, 40), pos=(130, 400))
            Color(1,1,1,(abs(weightone) / 10))
            Rectangle(size=(40, 40), pos=(130, 325))
            Color(1,1,1,(abs(weighttwo) / 10))
            Rectangle(size=(40, 40), pos=(130, 250))





class MyApp(App):
    def build(self):
        newfirst = First()
        return newfirst


if __name__ == '__main__':
    MyApp().run()