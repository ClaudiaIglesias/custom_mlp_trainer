


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
from kivy.graphics.texture import *
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

from filechooser import Filechooser, PopupLoading
from loaddataset import showdataset, readdataset, str_column_to_float
from neuralnet import initialize_network_no_weights, train_network, back_propagation, \
    initialize_network, transfer_sigmoid, transfer_sigmoid_derivative, predict, evaluate_algorithm, accuracy_metric, \
    predictfunc
from widgettexture import WidgetTexture

numoutputs = 1
n_outputs = 1
n_inputs = 2
dataset = 1
list_hidden = list()
input = list()
outputimageslist = list()
prediction = False
varerrorlayout = False

lbltimeleft = Label(text='tiempo restante estimado:', markup = True)

errorlbl = Label(text = '', markup = True)
labelcycles = (Label(text='[b] [color=D464EB]Ciclos: [/b][/color]', markup=True,
                             halign="left", color=(1, 1, 1)))
labelfolds = (Label(text='[b] [color=D464EB]Grupos: [/b][/color]' , markup=True,
                            halign="left", color=(1, 1, 1)))


inputimagelist = ''

accuracylabel =''
biaseslabels = list()
leftpanellist = list()
listoflines = list()
filename = ''
ti = TextInput(size=(40, 30), size_hint=(None, None))
ti2 = TextInput(size=(40, 30), size_hint=(None, None))
ti3 = TextInput(size=(40, 30), size_hint=(None, None))
progressbar = ProgressBar()
currentpage = 0



class LineParam():
    def __init__(self, layer, hidden, index, maximum, value, color, linewidget, canvas):
        self.layer = layer
        self.hidden = hidden
        self.index = index
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


class NeuralNetApp(App):
    class Lblleft():
        def __init__(self, myapp, type, dataerror):
            global varerrorlayout
            global list_hidden, n_outputs, n_inputs, currentpage, list_hidden, outputimageslist

            self.datestr = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            if myapp.network is None:
                return
            self.networklist = list(myapp.network)
            self.n_outputlbl = n_outputs
            self.n_inputlbl = n_inputs
            self.filename = filename
            self.type = type
            self.n_epoch = myapp.n_epoch
            self.n_folds = myapp.n_folds
            self.l_rate = myapp.l_rate
            self.dataerror = dataerror
            self.list_hidden = list(list_hidden)
            self.isimagelbl = myapp.isimage
            if myapp.sigmoidmode:
                self.transferlbl = 'sigmoid'
            else:
                self.transferlbl = 'relu'
            self.outputimageslist = list(outputimageslist)

    def auxlayoutpb(self):
        global dataset, imageinput
        global lbltimeleft, labelcycles, labelfolds, errorlbl
        global progressbar

        # calculate new maxs
        if self.valuemydata is None:
            return
        for layer in range(len(self.valuemydata.networklist)):
            mymax = 0.0
            for node in self.valuemydata.networklist[layer]:
                myweights = node['weights']
                for value in myweights:
                    mymax= max(mymax, abs(value))

            for myline in listoflines:
                if myline.layer == layer:
                    myline.value = abs(self.valuemydata.networklist[myline.layer][myline.hidden]['weights'][myline.index])
                    myline.max = mymax

        nchanges = 0
        for myline in listoflines:
            nchanges += myline.change()
        indeximageinput = int(random() * len(inputimagelist))
        if imageinput is not None:
            imageinput.update_texture(inputimagelist[indeximageinput])

        timenow = time.monotonic()
        elapse = timenow - self.timestart
        if self.valuepb == 0:
            lbltimeleft.text = 'estimando...'
        else:
            left = int((elapse / self.valuepb) * (100.0 - self.valuepb))
            #print('elapse {0}, valuep {1}, left {2}' .format(elapse, self.valuepb, left))
            if left > 3600:
                hours, seconds = divmod(left, 3600)
                minutes = int(seconds / 60)
                lbltimeleft.text = '[b] Tiempo restante estimado: [/b]' + str(hours) + ' horas ' + str(minutes) + ' minutos'
            elif left > 60:
                minutes, left = divmod(left, 60)
                lbltimeleft.text = '[b] Tiempo restante estimado: [/b]' + str(minutes) + ' minutos ' + str(left) + ' segundos'
            else:
                lbltimeleft.text = '[b] Tiempo restante estimado: [/b]' + str(left) + ' segundos'
        progressbar.value = self.valuepb

    # Error progress
    def layoutpbnext(self, deltatime):
        global labelcycles, labelfolds, errorlbl
        self.auxlayoutpb()
        errorlbl.text = '[b] [color=D464EB]Error cuadrático medio: [/b][/color]' + str("{:.4f}".format(self.mean_qerror))
        labelcycles.text = '[b] [color=D464EB]Ciclos: [/b][/color]' + str(self.currentcycle) + '/' + str(self.n_epoch)
        labelfolds.text = '[b] [color=D464EB]Grupos: [/b][/color]' + str(self.currentfold) + '/' + str(self.n_folds)

    # Train progress
    def layoutpbtrainnext(self, deltatime):
        global labelcycles, labelfolds
        self.auxlayoutpb()
        labelcycles.text = '[b] [color=D464EB]Ciclos: [/b][/color]' + str(self.currentcycle) + '/' + str(self.n_epoch)

    # Predict progress
    def layoutpbpredictnext(self, deltatime):
        global lbltimeleft, labelcycles, labelfolds
        global progressbar, imageinput
        timenow = time.monotonic()
        elapse = timenow - self.timestart
        if self.valuepredict == 0:
            lbltimeleft.text = 'estimando...'
        else:
            left = int((elapse / self.valuepredict) * (100.0 - self.valuepredict))
            if left > 3600:
                hours, seconds = divmod(left, 3600)
                minutes = int(seconds / 60)
                lbltimeleft.text = '[b] Tiempo restante estimado: [/b]' + str(hours) + ' horas ' + str(minutes) + ' minutos'
            elif left > 60:
                minutes, left = divmod(left, 60)
                lbltimeleft.text = '[b] Tiempo restante estimado: [/b]' + str(minutes) + ' minutos ' + str(left) + ' segundos'
            else:
                lbltimeleft.text = '[b] Tiempo restante estimado: [/b]' + str(left) + ' segundos'
        progressbar.value = self.valuepredict
        indeximageinput = int(random() * len(inputimagelist))
        if imageinput is not None:
            imageinput.update_texture(inputimagelist[indeximageinput])
        acc = (self.correctpredict* 100.0) / self.totalpredict
        self.labelaccuracy.text = '[b] Precisión: [/b]' + str("{:.2f}".format(acc)) + '%'
        # labelcycles.text = '[b] [color=D464EB]Ciclos: [/b][/color]' + str(self.currentcycle) + '/' + str(self.n_epoch)
        # labelfolds.text = '[b] [color=D464EB]Grupos: [/b][/color]' + str(self.n_folds)

    # When Train is pressed
    def drawinglblblackkv(self, textlayout, mydata):
        global accuracylabel, progressbar, prediction, biaseslabels
        global imageinput, inputimagelist
        global listoflines

        self.valuemydata = self.Lblleft(self, mydata.type, mydata.dataerror)

        listoflines.clear()
        prediction = True


        draw = self.parentlayout.ids.idLayoutNeuron.ids.idLayoutNeuronMainScreen.ids.idrelativelayoutneuronmainscreen
        titlelayout = self.parentlayout.ids.idLayoutNeuron.ids.idLayoutNeuronMainScreen.ids.idLabelsLayoutNeuronMainScreen

        draw.canvas.clear()

        self.leftlabels(textlayout, mydata)


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

            print('drawing level: {0}, max: {1}'.format(a, linecolorinput))
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
                            NeuralNetApp.listoflinesappend(self, 0, a, d, linecolorinput, draw.canvas, mydata.networklist, mypoints)


                    else:
                        for d in range(10):
                            mypoints = [middlex - 205, d * 50 + (middley - space / 2) + 17.5, middlex,
                                             a * 25 + (middley + 9 - spaceone / 2)]
                            NeuralNetApp.listoflinesappend(self, 0, a, d, linecolorinput, draw.canvas, mydata.networklist, mypoints)

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
                            NeuralNetApp.listoflinesappend(self, 0, a, d, linecolorinput, draw.canvas, mydata.networklist, mypoints)


                    else:
                        for d in range(10):
                            mypoints = [middlex - 205, d * 50 + (middley - 10 - space / 2) + 30, middlex,
                                             a * 25 + (middley + 9 - spaceone / 2)]
                            NeuralNetApp.listoflinesappend(self, 0, a, d, linecolorinput, draw.canvas, mydata.networklist, mypoints)
            draw.add_widget(Label(text = str(mydata.list_hidden[0]), pos = (middlex - 40 , middley+220), size_hint=(None,None), font_size=13))

            for e in range(numoutputs):
                if mydata.list_hidden[0] < 21:
                    for a in range(mydata.list_hidden[0]):
                        mypoints =[middlex+18, a * 25 + (middley  - spaceone / 2)+ 9, middlex+230,
                                         e * 50 + (middley+ 20 - spaceoutput / 2) ]
                        NeuralNetApp.listoflinesappend(self, 1, e, a, linecoloroutput, draw.canvas, mydata.networklist, mypoints)
                else:
                    for a in range(20):
                        mypoints =[middlex+18, a * 25 + (middley  - spaceone / 2)+ 9, middlex+230,
                                         e * 50 + (middley+ 20 - spaceoutput / 2) ]
                        NeuralNetApp.listoflinesappend(self, 1, e, a, linecoloroutput, draw.canvas, mydata.networklist, mypoints)

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
                            NeuralNetApp.listoflinesappend(self, 0, a, i, linecolorinput, draw.canvas, mydata.networklist,
                                                           mypoints)

                    else:
                        for i in range(10):
                            mypoints = [middlex - 205, i * 50 + (middley - space / 2) + 17.5, middlex - 80,
                                             a * 25 + (middley + 9 - spaceone / 2)]
                            NeuralNetApp.listoflinesappend(self, 0, a, i, linecolorinput, draw.canvas, mydata.networklist, mypoints)
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
                            NeuralNetApp.listoflinesappend(self, 0, a, i, linecolorinput, draw.canvas, mydata.networklist,
                                                           mypoints)

                    else:
                        for i in range(10):
                            mypoints = [middlex - 205, i * 50 + (middley - space / 2) + 17.5, middlex - 80,
                                        a * 25 + (middley + 9 - spaceone / 2)]
                            NeuralNetApp.listoflinesappend(self, 0, a, i, linecolorinput, draw.canvas, mydata.networklist,
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
                            NeuralNetApp.listoflinesappend(self, 1, d, j, linecolorinput, draw.canvas, mydata.networklist, mypoints)
                    else:
                        for j in range(20):
                            mypoints = [middlex -62, j * 25 + (middley+9 - spaceone / 2), middlex+80, d * 25 + (middley+9 - spacetwo / 2)]
                            NeuralNetApp.listoflinesappend(self, 1, d, j, linecolorinput, draw.canvas, mydata.networklist, mypoints)
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
                                NeuralNetApp.listoflinesappend(self, 1, d, j, linecolorinput, draw.canvas, mydata.networklist, mypoints)
                        else:
                            for j in range(20):
                                mypoints = [middlex -62, j * 25 + (middley+9 - spaceone / 2), middlex+80, d * 25 + (middley+9 - spacetwo / 2)]
                                NeuralNetApp.listoflinesappend(self, 1, d, j, linecolorinput, draw.canvas, mydata.networklist, mypoints)
                    else:
                        with draw.canvas:
                            Ellipse(pos=(middlex + 87, d * 25 + (middley - spacetwo / 2) + 9), size=(5, 5))
            for e in range(numoutputs):
                if mydata.list_hidden[1] < 21:
                    for d in range(mydata.list_hidden[1]):
                        mypoints = [middlex+98, d * 25 + (middley+9 - spacetwo / 2), middlex + 230, e * 50 + (middley +20 - spaceoutput / 2)]
                        NeuralNetApp.listoflinesappend(self, 2, e, d, linecoloroutput, draw.canvas, mydata.networklist, mypoints)
                else:
                    for d in range(20):
                        mypoints = [middlex + 98, d * 25 + (middley + 9 - spacetwo / 2), middlex + 230,e * 50 + (middley + 20 - spaceoutput / 2)]
                        NeuralNetApp.listoflinesappend(self, 2, e, d, linecoloroutput, draw.canvas, mydata.networklist, mypoints)
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
                            NeuralNetApp.listoflinesappend(self, 0, a, i, linecolorinput, draw.canvas, mydata.networklist,
                                                           mypoints)

                    else:
                        for i in range(10):
                            mypoints = [middlex - 205, i * 50 + (middley - space / 2) + 17.5, middlex - 130,
                                             a * 25 + (middley + 9 - spaceone / 2)]
                            NeuralNetApp.listoflinesappend(self, 0, a, i, linecolorinput, draw.canvas, mydata.networklist,
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
                            NeuralNetApp.listoflinesappend(self, 0, a, i, linecolorinput, draw.canvas, mydata.networklist,
                                                           mypoints)

                    else:
                        for i in range(10):
                            mypoints = [middlex - 205, i * 50 + (middley - space / 2) + 17.5, middlex - 130,
                                        a * 25 + (middley + 9 - spaceone / 2)]
                            NeuralNetApp.listoflinesappend(self, 0, a, i, linecolorinput, draw.canvas, mydata.networklist,
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
                            NeuralNetApp.listoflinesappend(self, 1, d, j, linecolorinput, draw.canvas, mydata.networklist, mypoints)
                    else:
                        for j in range(20):
                            mypoints = [middlex-112, j * 25 + (middley+9 - spaceone / 2), middlex, d * 25 + (middley+9 - spacetwo / 2)]
                            NeuralNetApp.listoflinesappend(self, 1, d, j, linecolorinput, draw.canvas, mydata.networklist, mypoints)
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
                            NeuralNetApp.listoflinesappend(self, 1, d, j, linecolorinput, draw.canvas, mydata.networklist,
                                                           mypoints)
                    else:
                        for j in range(20):
                            mypoints = [middlex - 112, j * 25 + (middley + 9 - spaceone / 2), middlex,
                                        d * 25 + (middley + 9 - spacetwo / 2)]
                            NeuralNetApp.listoflinesappend(self, 1, d, j, linecolorinput, draw.canvas, mydata.networklist,
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
                            NeuralNetApp.listoflinesappend(self, 2, d, l, linecolorinput, draw.canvas, mydata.networklist, mypoints)
                    else:
                        for l in range(20):
                            mypoints = [middlex+18, l * 25 + (middley+9 - spacetwo / 2), middlex+130, d * 25 + (middley+9 - spacethree / 2)]
                            NeuralNetApp.listoflinesappend(self, 2, d, l, linecolorinput, draw.canvas, mydata.networklist, mypoints)
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
                            NeuralNetApp.listoflinesappend(self, 2, d, l, linecolorinput, draw.canvas, mydata.networklist,
                                                           mypoints)
                    else:
                        for l in range(20):
                            mypoints = [middlex + 18, l * 25 + (middley + 9 - spacetwo / 2), middlex + 130,
                                        d * 25 + (middley + 9 - spacethree / 2)]
                            NeuralNetApp.listoflinesappend(self, 2, d, l, linecolorinput, draw.canvas, mydata.networklist,
                                                           mypoints)

            for e in range(numoutputs):
                if mydata.list_hidden[2] < 21:
                    for d in range(mydata.list_hidden[2]):
                        mypoints = [middlex+148, d * 25 + (middley + 9 - spacethree / 2), middlex+230, e * 50 + (middley + 20 - spaceoutput / 2)]
                        NeuralNetApp.listoflinesappend(self, 3, e, d, linecoloroutput, draw.canvas, mydata.networklist, mypoints)
                else:
                    for d in range(20):
                        mypoints = [middlex+148, d * 25 + (middley + 9 - spacethree / 2), middlex+230, e * 50 + (middley + 20 - spaceoutput / 2)]
                        NeuralNetApp.listoflinesappend(self, 3, e, d, linecoloroutput, draw.canvas, mydata.networklist, mypoints)
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
            image = WidgetTexture(row, mydata.isimagelbl)
            image.size_hint = (None, None)
            image.pos = middlex + 237 , c * 50 + (middley - 30 - spaceoutput / 2)
            draw.add_widget(image)
            c += 1

        if len(inputimagelist):
            imageinput = WidgetTexture(inputimagelist[0])
            imageinput.pos = middlex - 320 , middley - 50
            imageinput.size_hint=(None, None)
            draw.add_widget(imageinput)
        else:
            imageinput = None
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


    def listoflinesappend(self, layer, hidden, index, maximum, canvas, networklist, linepoints):
        value = abs(networklist[layer][hidden]['weights'][index])
        color = value / maximum
        with canvas:
            Color(color, color, color)
            linewidget = Line(points=linepoints)
        listoflines.append(LineParam(layer, hidden, index, maximum, value, color, linewidget, canvas))

    #When self.btnError is pressed
    #The lbls on the left
    def leftlabels(self, textlayout, leftpanelitem, *largs):
        global varerrorlayout, labelcycles, labelfolds, errorlbl
        global list_hidden, n_outputs, n_inputs, currentpage


        lbldate = Label(text=leftpanelitem.datestr, markup=True, halign="left")
        lbldate.bind(size=lbldate.setter('text_size'))
        labelfile = (Label(text='[b] Archivo: [/b][i]' + basename(leftpanelitem.filename), markup=True, halign="left"))
        labelfile.bind(size=labelfile.setter('text_size'))
        if leftpanelitem.type != 'None':
            labeltype = (Label(text='[b] [color=4BDCFF]Función: [/b][/color]' + leftpanelitem.type, markup=True,
                               halign="left", color=(1, 1, 1)))
            labeltype.bind(size=labeltype.setter('text_size'))
        labelcycles.text = '[b] [color=D464EB]Ciclos: [/b][/color]' + str(leftpanelitem.n_epoch)
        labelcycles.bind(size=labelcycles.setter('text_size'))
        labelrate = (Label(text='[b] [color=D464EB]Ratio: [/b][/color]' + str(leftpanelitem.l_rate), markup=True,
                           halign="left", color=(1, 1, 1)))
        labelrate.bind(size=labelrate.setter('text_size'))
        labeltransfer = (Label(text='[b] [color=4BDCFF]Transferencia: [/b][/color]' + str(leftpanelitem.transferlbl), markup=True,
                           halign="left", color=(1, 1, 1)))
        labeltransfer.bind(size=labeltransfer.setter('text_size'))

        if leftpanelitem.type is 'Error' or leftpanelitem.type is 'None':
            labelfolds.text = '[b] [color=D464EB]Grupos: [/b][/color]' + str(leftpanelitem.n_folds)
            labelfolds.bind(size=labelfolds.setter('text_size'))
        errorlbl.text = ''
        errorlbl.bind(size=errorlbl.setter('text_size'))
        varerrorlayout = True
        self.labelaccuracy = (Label(text='[b] Precisión: [/b]' + str("{:.2f}".format(leftpanelitem.dataerror)) + '%', markup=True,
                                   halign="left"))
        self.labelaccuracy.bind(size=self.labelaccuracy.setter('text_size'))




        if currentpage == 1:
            self.btnlesssidetext.disabled = True
        if currentpage != 1:
            self.btnlesssidetext.disabled = False

        textlayout.clear_widgets()

        textlayout.add_widget(lbldate)
        textlayout.add_widget(labelfile)
        if leftpanelitem.type != 'None':
            textlayout.add_widget(labeltype)
        textlayout.add_widget(labelcycles)
        textlayout.add_widget(labelrate)
        if leftpanelitem.type is 'Error' or leftpanelitem.type is 'None':
            textlayout.add_widget(labelfolds)
        textlayout.add_widget(errorlbl)
        textlayout.add_widget(labeltransfer)
        textlayout.add_widget(self.labelaccuracy)



        self.lblcurrentpage.text = ('[b]Página:[/b]' + ' ' + str("{:.0f}".format(currentpage)) + '/' + str("{:.0f}".format(len(leftpanellist))))

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
            self.drawinglblblackkv(textlayout, mydata)
        if type == 'btnless':
            currentpage -= 1
            mydata = leftpanellist[currentpage - 1]
            self.drawinglblblackkv(textlayout, mydata)
        NeuralNetApp.establishbtnpages(self)

        self.lblcurrentpage.text = ('[b]Página:[/b]' + ' ' + str("{:.0f}".format(currentpage)) + '/' + str(
            "{:.0f}".format(len(leftpanellist))))

    #Plain drawing small size
    def drawingtryout(self, draw, titlelayout, *largs):
        global a, b, list_hidden, n_outputs, n_inputs, accuracylabel, progressbar

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
        except Exception as e:
            print('store panel error: ' + str(e))
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
                    NeuralNetApp.network = None
                    mydataclasss = NeuralNetApp.Lblleft(self, type='', dataerror=0)
                    for key in mydatadict:
                        setattr(mydataclasss, key, mydatadict[key])
                    leftpanellist.append(mydataclasss)
                    currentpage = len(leftpanellist)
                    NeuralNetApp.establishbtnpages(self)
        except Exception as e:
            print('restore panel error: ' + str(e))
            os.remove('neuroncfg.json')

    def addclasstopanel(self, mypannelclass):
        global currentpage
        leftpanellist.append(mypannelclass)
        currentpage = len(leftpanellist)
        NeuralNetApp.establishbtnpages(self)
        NeuralNetApp.storeleftpanellist(self)
    def historylayout(self):
        self.layoutmainscreen.height = 580
        self.btnlayout.height = 50
        self.layoutinfoleft.size_hint_x = 0.4
        self.layoutinforight.size_hint_x = 0.0
        self.layout.remove_widget(self.btnTrain)
        self.layout.remove_widget(self.btnError)
        self.layout.remove_widget(self.btnUsar)
        self.layout.add_widget(self.btnUsar)
        self.layoutinfotop.remove_widget(self.btnSettings)
        self.layoutinfotop.remove_widget(self.btnBack)
        self.layoutinfotop.remove_widget(self.btnHistory)
        self.layout.remove_widget(self.btnPredict)
        self.layout.remove_widget(self.btnProcessPredict)
        self.layout.remove_widget(self.btnCancel)
        self.layout.remove_widget(self.btnnewinput)
        self.layout.remove_widget(self.layoutsettingsvertical)
        if self.settingslayoutvrb:
            self.layoutinfotop.add_widget(self.btnSettings)
        else:
            self.layoutinfotop.add_widget(self.btnBack)
        self.layoutbtninfo.remove_widget(self.btnlesssidetext)
        self.layoutbtninfo.remove_widget(self.btnplussidetext)
        self.layoutbtninfo.add_widget(self.btnlesssidetext)
        self.layoutbtninfo.add_widget(self.btnplussidetext)
        self.layoutinfodown.add_widget(Label(
            text='-----------------------------------------------------------------------------------------------------------------------------------------------',
            pos_hint={'center_x': .5, 'center_y': .5}, size_hint=(1, .1), color=(.6, .87, .89, 1)))
        self.layoutinfodown.add_widget(self.lblcurrentpage)

        self.myevent.drawinghistorylayout(self)

    def auxhistorylayout(self):
        try:
            NeuralNetApp.restoreleftpanellist(self)
            mydata = leftpanellist[-1]
            currentpage = len(leftpanellist)
            self.drawinglblblackkv(self.layoutinfomiddle, mydata)
            self.establishbtnpages()
        except:
            pass

    def settingslayout(self):
        global prediction, biaseslabels, varerrorlayout
        prediction = False
        self.settingslayoutvrb = True
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
        self.layout.remove_widget(self.btnUsar)
        self.layoutinfotop.remove_widget(self.btnHistory)
        self.layoutinfotop.add_widget(self.btnHistory)
        self.layoutinfotop.remove_widget(self.btnSettings)
        self.layoutinfotop.remove_widget(self.btnBack)
        self.layoutbtninfo.remove_widget(self.btnlesssidetext)
        self.layoutbtninfo.remove_widget(self.btnplussidetext)
        self.layout.remove_widget(self.btnProcessPredict)
        self.layout.remove_widget(self.btnCancel)
        self.layout.remove_widget(self.btnnewinput)
        self.clear(layoutneuronmainscreen, layoutlabels)
        self.layoutinfodown.clear_widgets()
        self.layout.add_widget(self.layoutsettingsvertical)
        self.layoutinfomiddle.clear_widgets()


        self.btnTrain.disabled = True
        self.btnError.disabled = True
        self.btnPredict.disabled = True
        self.onpredictlayout = False

    def btnTrainpressed(self):
            global progressbar, accuracylabel, numdata
            global  n_inputs, n_outputs, dataset, list_hidden
            layoutprogressbar = self.parentlayout.ids.mainseparatorprogressbar
            layoutprogressbar.remove_widget(progressbar)
            layoutprogressbar.add_widget(progressbar)
            seed(1)
            if self.sigmoidmode:
                self.transfermode = 'sigmoid'
            else:
                self.transfermode = 'relu'

            self.currentfold = 0
            self.currentcycle = 0
            progressbar.value = 0
            self.valuepb = 0
            self.stoptrainning = False
            self.printth = 0
            self.timestart = time.monotonic()
            self.trainclock = Clock.schedule_interval(self.layoutpbtrainnext, 1.0)
            self.network = initialize_network(n_inputs, list_hidden, n_outputs, self.transfermode)
            self.valuemydata = self.Lblleft(self, type='Entrenamiento', dataerror=0)
            self.leftlabels(self.layoutinfomiddle, self.valuemydata)
            threading.Thread(target=train_network, args=(self.network, dataset, 1, self.l_rate, self.n_epoch, n_outputs, 'train', self.myevent, self)).start()
            self.layout.remove_widget(lbltimeleft)
            self.layout.add_widget(lbltimeleft)
            #train_network(network, dataset, l_rate, n_epoch, n_outputs)
            # for row in dataset:
            #      prediction = predict(network, row)
            #      accuracy_metric(dataset,prediction)
            self.cancellayout()

    def btnErrorpressed(self):
            global progressbar, accuracylabel, numdata
            global n_inputs, n_outputs, dataset, list_hidden
            global timestart, errorlbl
            layoutprogressbar = self.parentlayout.ids.mainseparatorprogressbar
            layoutprogressbar.remove_widget(progressbar)
            layoutprogressbar.add_widget(progressbar)
            progressbar.value = 0
            self.valuepb = 0
            self.mean_qerror = 0

            if self.sigmoidmode:
                self.transfermode = 'sigmoid'
            else:
                self.transfermode = 'relu'

            self.valuepb = 0
            self.stoptrainning = False
            self.timestart = time.monotonic()
            self.trainclock = Clock.schedule_interval(self.layoutpbnext, 1.0)

            self.currentfold = 0
            self.currentcycle = 0
            progressbar.value = 0
            self.printth = 0
            self.layout.remove_widget(lbltimeleft)
            self.layout.add_widget(lbltimeleft)
            self.valuemydata = self.Lblleft(self, type='Error', dataerror=0)
            self.leftlabels(self.layoutinfomiddle, self.valuemydata)
            threading.Thread(target=evaluate_algorithm, args=(dataset, back_propagation, self.myevent, self,
                                                                   self.n_folds, self.l_rate, self.n_epoch, list_hidden)).start()
            self.cancellayout()

    def btnDetailspressed(self):
        content = Filechooser(self)
        popuploading = PopupLoading()
        content.popup = Popup(title=basename(self.filename), content=popuploading,
                           size_hint=(0.9, 0.9), auto_dismiss=False, separator_color=(.67, .57, .69))
        content.popup.open()

        popuploading.start()

        threading.Thread(target=content.filechoosershowdataset,
                         args=(dirname(self.filename), basename(self.filename), self.myevent, self.mistakeslist)).start()



        #popupfilechooser = Popup(title="Cargar archivo", content=content,
                            #size_hint=(0.9, 0.9),auto_dismiss= False, separator_color= (.67,.57,.69))
        #content.filechoosershowdataset(dirname(self.filename), basename(self.filename), self.myevent, True)
        #popupfilechooser.open()


    def btnUsarpressed(self):
        global list_hidden, n_outputs, n_inputs, currentpage, list_hidden, outputimageslist, filename
        mydata = leftpanellist[currentpage - 1]
        self.date = mydata.datestr
        self.network = list(mydata.networklist)

        n_outputs = mydata.n_outputlbl
        n_inputs = mydata.n_inputlbl
        self.filename = mydata.filename
        filename = mydata.filename
        self.n_epoch = mydata.n_epoch
        self.n_folds = mydata.n_folds
        self.l_rate = mydata.l_rate
        list_hidden = list(mydata.list_hidden)
        self.isimage = mydata.isimagelbl
        self.transfermode = mydata.transferlbl
        self.outputimageslist = list(mydata.outputimageslist)
        self.historyfile = True
        self.lblfilename.text = ('[b] Archivo: [/b]' + '[i]' + basename(filename))
        optionaloutput, optionalimage, isimage, mytexturedataset = showdataset(filename)
        if len(mytexturedataset) > 100:
            self.inputimagelist = list(mytexturedataset[10:100])
        else:
            self.inputimagelist = list(mytexturedataset[:-1])
        self.settingslayout()

    def btnokpressed(self):

        if self.filename is '':
            contentpopup = Label(text='Selecciona un archivo apretando al botón "ABRIR"', color = (1,1,1,.9))
            popuparchivo = Popup(title="Archivo no seleccionado", content=contentpopup,
                                     size_hint=(0.5, 0.3), auto_dismiss=True, separator_color=(.67, .57, .69))
            popuparchivo.open()
        else:
            self.mainlayout_func()


    def mainlayout_func(self):
        global prediction, n_inputs, n_outputs, dataset, numdata, listinputs, filename, inputimagelist, outputimageslist
        print(self.historyfile)
        inputimagelist = self.inputimagelist
        outputimageslist = self.outputimageslist
        filename = self.filename
        self.layoutmainscreen.height = 580
        self.btnlayout.height = 50
        self.layoutinfoleft.size_hint_x = 0.1
        self.layoutinforight.size_hint_x = 0.1

        if self.onpredictlayout:
            dataset = self.savedataset
            self.network = self.savenetwork
            filename = self.savedfilename
            self.lblfilename = self.parentlayout.ids.idlabelfilename
            self.lblfilename.text = ('[b] Archivo: [/b]' + '[i]' + basename(filename))

        if self.sigmoidbox.active == True:
            self.sigmoidmode = True
        else:
            self.sigmoidmode = False

        self.layouttrainscreen = self.parentlayout.ids.idLayoutNeuron.ids.idAnchorLayoutNeuron
        self.layouttrainscreen.anchor_x = 'center'

        self.layout.remove_widget(self.layoutsettingsvertical)
        self.layoutbtninfo.remove_widget(self.btnlesssidetext)
        self.layoutbtninfo.remove_widget(self.btnplussidetext)
        self.layout.remove_widget(self.btnCancel)
        self.layout.remove_widget(self.btnnewinput)
        self.layout.remove_widget(self.btnTrain)
        self.layout.remove_widget(self.btnError)
        self.layout.remove_widget(self.btnPredict)
        self.layout.remove_widget(self.btnDetails)
        self.layout.remove_widget(self.btnUsar)
        self.layout.add_widget(self.btnTrain)
        self.layout.add_widget(self.btnError)
        self.layout.add_widget(self.btnPredict)
        self.layoutinfotop.remove_widget(self.btnSettings)
        self.layoutinfotop.remove_widget(self.btnHistory)
        if self.settingslayoutvrb is False:
            self.layoutinfotop.add_widget(self.btnSettings)
            self.layoutinfotop.add_widget(self.btnHistory)
        self.layoutinfotop.remove_widget(self.btnBack)
        self.btnSettings.disabled = False
        self.btnHistory.disabled = False
        self.btnBack.disabled = False
        self.onpredictlayout = False
        self.layout.remove_widget(self.btnProcessPredict)
        layoutprogressbar = self.parentlayout.ids.mainseparatorprogressbar
        layoutprogressbar.remove_widget(progressbar)

        if prediction == False:
            if varerrorlayout == False:
                layoutprogressbar.add_widget(progressbar)
                self.myevent.drawingmainlayout(self)
                self.btnPredict.disabled = True
            else:
                self.btnPredict.disabled = True
        else:
            self.btnPredict.disabled = False
        if self.historyfile:
            self.btnPredict.disabled = False
    def predictlayout(self):
        global dataset
        self.layouttrainscreen = self.parentlayout.ids.idLayoutNeuron.ids.idAnchorLayoutNeuron
        self.layouttrainscreen.anchor_x = 'center'
        self.savedataset = dataset.copy()
        self.savenetwork = self.network.copy()
        self.savedfilename = filename
        self.layout.remove_widget(self.btnTrain)
        self.layout.remove_widget(self.btnError)
        self.layout.remove_widget(self.btnPredict)
        self.layout.remove_widget(self.btnCancel)
        self.layoutinfotop.remove_widget(self.btnSettings)
        self.layoutinfotop.remove_widget(self.btnHistory)
        self.layoutinfotop.remove_widget(self.btnBack)
        self.layout.remove_widget(self.btnnewinput)
        self.btnSettings.disabled = False
        self.btnHistory.disabled = False
        self.btnBack.disabled = False
        self.onpredictlayout = True

        self.layout.add_widget(self.btnProcessPredict)
        self.layoutinfotop.add_widget(self.btnBack)
        self.layoutinfotop.add_widget(self.btnSettings)
        self.layoutinfotop.add_widget(self.btnHistory)
        self.filechooserfunction()
# When file is chosen
    def predictdonelayout(self):
        global dataset
        self.layouttrainscreen = self.parentlayout.ids.idLayoutNeuron.ids.idAnchorLayoutNeuron
        layoutprogressbar = self.parentlayout.ids.mainseparatorprogressbar
        layoutprogressbar.add_widget(progressbar)
        progressbar.value = 0
        self.layout.remove_widget(self.btnProcessPredict)
        self.mistakeslist = list()
        self.valuemydata = self.Lblleft(self, type='Test', dataerror=0)
        self.drawinglblblackkv(self.layoutinfomiddle, self.valuemydata)
        self.leftlabels(self.layoutinfomiddle, self.valuemydata)
        threading.Thread(target=readdataset, args=(self.filename, self.myevent, self, None, None)).start()

    def predictfinishedlayout(self, accuracy):
        self.layout.remove_widget(self.btnTrain)
        self.layout.remove_widget(self.btnError)
        self.layout.remove_widget(self.btnPredict)
        self.layout.remove_widget(self.btnProcessPredict)
        self.layout.remove_widget(self.btnCancel)
        self.layoutinfotop.remove_widget(self.btnSettings)
        self.layoutinfotop.remove_widget(self.btnHistory)
        self.layoutinfotop.remove_widget(self.btnBack)
        self.layout.add_widget(self.btnDetails)
        layoutprogressbar = self.parentlayout.ids.mainseparatorprogressbar
        layoutprogressbar.remove_widget(progressbar)
        progressbar.value = 0
        self.btnSettings.disabled = False
        self.btnHistory.disabled = False
        self.btnBack.disabled = False

        self.layoutinfotop.add_widget(self.btnBack)
        self.layoutinfotop.add_widget(self.btnSettings)
        self.layoutinfotop.add_widget(self.btnHistory)
        mydata = NeuralNetApp.Lblleft(self, type='Test', dataerror=accuracy)
        NeuralNetApp.addclasstopanel(self, mydata)
        self.drawinglblblackkv(self.layoutinfomiddle, mydata)

    def trainlayout(self):
        self.layouttrainscreen = self.parentlayout.ids.idLayoutNeuron.ids.idAnchorLayoutNeuron
        self.layoutinfoleft.size_hint_x = 0.4
        self.layoutinforight.size_hint_x = 0.0
        self.layout.remove_widget(self.btnTrain)
        self.layout.remove_widget(self.btnError)
        self.layout.remove_widget(self.btnPredict)
        self.layout.remove_widget(self.btnCancel)
        self.layoutinfotop.remove_widget(self.btnSettings)
        self.layoutinfotop.remove_widget(self.btnHistory)
        self.layoutinfotop.remove_widget(self.btnBack)
        layoutprogressbar = self.parentlayout.ids.mainseparatorprogressbar
        layoutprogressbar.remove_widget(progressbar)
        progressbar.value=0
        self.btnSettings.disabled = False
        self.btnHistory.disabled = False
        self.btnBack.disabled = False
        self.onpredictlayout = False
        self.btnPredict.disabled = False

        self.layoutinfotop.add_widget(self.btnBack)
        self.layoutinfotop.add_widget(self.btnSettings)
        self.layoutinfotop.add_widget(self.btnHistory)
        mydata = NeuralNetApp.Lblleft(self, type='Entrenamiento', dataerror=0)
        NeuralNetApp.addclasstopanel(self, mydata)
        self.drawinglblblackkv(self.layoutinfomiddle, mydata)

    def cancellayout(self):
        self.layout.remove_widget(self.btnTrain)
        self.layout.remove_widget(self.btnError)
        self.layout.remove_widget(self.btnPredict)
        self.layout.remove_widget(self.btnProcessPredict)
        self.layoutinfotop.remove_widget(self.btnBack)
        self.layoutinfotop.remove_widget(self.btnSettings)
        self.layoutinfotop.remove_widget(self.btnHistory)
        self.btnSettings.disabled = True
        self.btnHistory.disabled = True
        self.btnBack.disabled = True
        self.layout.add_widget(self.btnCancel)
        self.layoutinfotop.add_widget(self.btnBack)
        self.layoutinfotop.add_widget(self.btnSettings)
        self.layoutinfotop.add_widget(self.btnHistory)

    def errorlayout(self, accuracy):
        self.layouttrainscreen = self.parentlayout.ids.idLayoutNeuron.ids.idAnchorLayoutNeuron
        self.layoutinfoleft.size_hint_x = 0.4
        self.layoutinforight.size_hint_x = 0.0
        self.layout.remove_widget(self.btnTrain)
        self.layout.remove_widget(self.btnError)
        self.layout.remove_widget(self.btnPredict)
        self.layout.remove_widget(self.btnCancel)
        self.layoutinfotop.remove_widget(self.btnSettings)
        self.layoutinfotop.remove_widget(self.btnHistory)
        self.layoutinfotop.remove_widget(self.btnBack)
        layoutprogressbar = self.parentlayout.ids.mainseparatorprogressbar
        layoutprogressbar.remove_widget(progressbar)
        progressbar.value = 0
        self.btnSettings.disabled = False
        self.btnHistory.disabled = False
        self.btnBack.disabled = False
        self.onpredictlayout = False

        self.layoutinfotop.add_widget(self.btnBack)
        self.layoutinfotop.add_widget(self.btnSettings)
        self.layoutinfotop.add_widget(self.btnHistory)
        mydata = NeuralNetApp.Lblleft(self, type='Error', dataerror=accuracy)
        self.addclasstopanel(mydata)
        self.leftlabels(self.layoutinfomiddle, mydata)

    def btnplusneuronpressed (self):
        self.btnlessneuron.disabled = False
        if list_hidden[0] >= 256:
            list_hidden[0] = 256

        else:
            list_hidden[0] = list_hidden[0] * 2
            self.drawingtryout(layoutneuronmainscreen, layoutlabels)
            self.lblhiddenlayer.text = str(list_hidden[0])
            if list_hidden[0] == 256:
                self.btnplusneuron.disabled = True
    def btnlessneuronpressed(self):
        self.btnplusneuron.disabled = False
        if list_hidden[0] <= 1:
            list_hidden[0] = 1
        else:
            list_hidden[0] = int (list_hidden[0] /2)
            self.lblhiddenlayer.text = str(list_hidden[0])
            self.drawingtryout(layoutneuronmainscreen, layoutlabels)
            if list_hidden[0] == 1:
                self.btnlessneuron.disabled = True

    def btnplusneurontwopressed(self):
        self.btnlessneurontwo.disabled = False

        if list_hidden[1] >= 256:
            list_hidden[1] = 256
        else:
            list_hidden[1] = list_hidden[1] * 2
            self.lblhiddenlayertwo.text = str(list_hidden[1])
            self.drawingtryout(layoutneuronmainscreen, layoutlabels)
            if list_hidden[1] == 256:
                self.btnplusneurontwo.disabled = True
    def btnlessneurontwopressed(self):
        self.btnplusneurontwo.disabled = False
        if list_hidden[1] <= 1:
            list_hidden[1] = 1
        else:
            list_hidden[1] = int (list_hidden[1] /2)
            self.lblhiddenlayertwo.text = str(list_hidden[1])
            self.drawingtryout(layoutneuronmainscreen, layoutlabels)
            if list_hidden[1] == 1:
                self.btnlessneurontwo.disabled = True

    def btnplusneuronthreepressed(self):
        self.btnlessneuronthree.disabled = False

        if list_hidden[2] >= 256:
            list_hidden[2] = 256
        else:
            list_hidden[2] = list_hidden[2] * 2
            self.lblhiddenlayerthree.text = str(list_hidden[2])
            self.drawingtryout(layoutneuronmainscreen, layoutlabels)
            if list_hidden[2] == 256:
                self.btnplusneuronthree.disabled = True
    def btnlessneuronthreepressed(self):
        self.btnplusneuronthree.disabled = False
        if list_hidden[2] <= 1:
            list_hidden[2] = 1
        else:
            list_hidden[2] = int (list_hidden[2] /2)
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
        self.btnlessfolds.disabled = False
        if self.n_folds >= 10:
            self.n_folds = 10
        else:
            self.n_folds += 1
            self.numfolds.text = str(self.n_folds)
            if self.n_folds == 10:
                self.btnplusfolds.disabled = True
    def btnlessfoldspressed(self):
        self.btnplusfolds.disabled = False
        if self.n_folds <= 2:
            self.n_folds = 2
        else:
            self.n_folds = self.n_folds - 1
            self.numfolds.text = str(self.n_folds)
            if self.n_folds == 2:
                self.btnlessfolds.disabled = True

    def btnplusratepressed (self):
        global l_rate
        self.btnlessrate.disabled = False
        if self.l_rate >= 1.0:
            self.l_rate = 1.0
        else:
            self.l_rate = self.l_rate + 0.1
            self.l_rate = float("{:.1f}".format(self.l_rate))
            self.numrate.text = str(self.l_rate)
            if self.l_rate == 1.0:
                self.btnplusrate.disabled = True
    def btnlessratepressed(self):
        self.btnplusrate.disabled = False
        if self.l_rate <= 0.1:
            self.l_rate = 0.1
        else:
            self.l_rate = self.l_rate - 0.1
            self.l_rate = float("{:.1f}".format(self.l_rate))
            self.numrate.text = str(self.l_rate)
            if self.l_rate == 0.1:
                self.btnlessrate.disabled = True

    def btnplusepochpressed (self):

        self.btnlessepoch.disabled = False
        if self.n_epoch >= 500:
            self.n_epoch = 500
        else:
            self.n_epoch += 10
            self.numepoch.text = str(self.n_epoch)
            if self.n_epoch == 500:
                self.btnplusepoch.disabled = True

    def filechooserfunction(self):

        content = Filechooser(self)
        popupfilechooser = Popup(title="Cargar archivo", content=content,
                            size_hint=(0.9, 0.9),auto_dismiss= False, separator_color= (.67,.57,.69))
        content.update_parent_popup(popupfilechooser)
        popupfilechooser.open()

    def btncancelpressed(self):
        self.stoptrainning = True

    def btnlessepochpressed(self):
        self.btnplusepoch.disabled = False
        if self.n_epoch <= 10:
            self.n_epoch = 10
        else:
            self.n_epoch = self.n_epoch - 10
            self.numepoch.text = str(self.n_epoch)
            if self.n_epoch == 10:
                self.btnlessepoch.disabled = True

    def build(self):
        global layoutneuronmainscreen, layoutlabels, currentpage
        class MyEventDispatcher(EventDispatcher):
            global lbltimeleft
            def __init__(self, **kwargs):
                self.register_event_type('on_evevaluatealgorithm')
                self.register_event_type('on_trainfinished')
                self.register_event_type('on_drawingmainlayout')
                self.register_event_type('on_evreaddataset')
                self.register_event_type('on_evshowfile')
                self.register_event_type('on_predictfinished')
                self.register_event_type('on_drawinghistorylayout')
                super(MyEventDispatcher, self).__init__(**kwargs)
            @mainthread
            def evevaluatealgorithm(self, myapp, scores):
                if myapp.stoptrainning:
                    self.dispatch('on_trainfinished', myapp)
                else:
                    self.dispatch('on_evevaluatealgorithm', myapp, scores)
            #Error
            def on_evevaluatealgorithm(self, myapp, scores):
                myapp.trainclock.cancel()
                layoutprogressbar = myapp.parentlayout.ids.mainseparatorprogressbar
                layoutprogressbar.remove_widget(progressbar)
                progressbar.value = 0
                print('Scores: %s' % scores)
                if len(scores):
                    accuracylabel = (sum(scores) / float(len(scores)))
                    myapp.errorlayout(accuracylabel)
                myapp.layout.remove_widget(lbltimeleft)

            @mainthread
            def trainfinished(self, myapp):
                self.dispatch('on_trainfinished', myapp)
            #Train
            def on_trainfinished(self, myapp):
                myapp.layoutprogressbar = myapp.parentlayout.ids.mainseparatorprogressbar
                myapp.layoutprogressbar.remove_widget(progressbar)
                myapp.layout.remove_widget(lbltimeleft)
                progressbar.value = 0
                myapp.trainclock.cancel()
                if myapp.stoptrainning:
                    myapp.mainlayout_func()
                else:
                    myapp.trainlayout()
                    for layer in myapp.network:
                        print(layer)

            @mainthread
            def predictfinished(self, myapp, scores):
                if myapp.stoptrainning:
                    self.dispatch('on_trainfinished', myapp)
                else:
                    self.dispatch('on_predictfinished', myapp, scores)
            # Predict
            def on_predictfinished(self, myapp, accuracy):
                myapp.layoutprogressbar = myapp.parentlayout.ids.mainseparatorprogressbar
                myapp.layoutprogressbar.remove_widget(progressbar)
                myapp.layout.remove_widget(lbltimeleft)
                progressbar.value = 0
                myapp.trainclock.cancel()
                if myapp.stoptrainning:
                    myapp.mainlayout_func()
                else:
                    myapp.predictfinishedlayout(accuracy)

            @mainthread
            def drawingmainlayout(self, myapp):
                self.dispatch('on_drawingmainlayout', myapp)

            def on_drawingmainlayout(self, myapp):
                #MyApp.drawingblank(btnOk, layoutneuronmainscreen, layoutlabels)
                global filename
                threading.Thread(target=readdataset, args=(filename, myapp.myevent, myapp, None, None)).start()

            @mainthread
            def drawinghistorylayout(self, myapp):
                self.dispatch('on_drawinghistorylayout', myapp)

            def on_drawinghistorylayout(self, myapp):
                NeuralNetApp.auxhistorylayout(myapp)


            @mainthread
            def evreaddataset(self, progressvarvalue, myisimage, mydataset, myapp):
                self.dispatch('on_evreaddataset', progressvarvalue, myisimage, mydataset, myapp)

            def on_evreaddataset(self, progressvarvalue, myisimage, mydataset, myapp):
                global dataset, n_inputs, n_outputs, lbltimeleft
                myapp.layoutprogressbar = myapp.parentlayout.ids.mainseparatorprogressbar
                #myapp.layoutprogressbar.add_widget(progressbar)
                progressbar.value = progressvarvalue
                if progressvarvalue == 100:
                    progressbar.value = 100.0
                    myapp.isimage = myisimage
                    if myapp.onpredictlayout is False:
                        dataset = mydataset
                        n_inputs = len(dataset[0]) - 1
                        n_outputs = len(set([row[-1] for row in dataset]))
                        myapp.network = initialize_network_no_weights(n_inputs, list_hidden, n_outputs)
                        myapp.drawinglblblackkv(myapp.layoutinfomiddle, myapp.Lblleft(myapp, type='None', dataerror=0))
                    myapp.layoutprogressbar = myapp.parentlayout.ids.mainseparatorprogressbar
                    myapp.layoutprogressbar.remove_widget(progressbar)
                    myapp.layout.remove_widget(lbltimeleft)
                    myapp.btnTrain.disabled = False
                    myapp.btnError.disabled = False
                    if myapp.settingslayoutvrb:
                        myapp.layoutinfotop.remove_widget(myapp.btnSettings)
                        myapp.layoutinfotop.remove_widget(myapp.btnHistory)
                        myapp.layoutinfotop.add_widget(myapp.btnSettings)
                        myapp.layoutinfotop.add_widget(myapp.btnHistory)
                        myapp.settingslayoutvrb = False
                    if myapp.onpredictlayout:
                        myapp.cancellayout()
                        layoutprogressbar = myapp.parentlayout.ids.mainseparatorprogressbar
                        layoutprogressbar.remove_widget(progressbar)
                        layoutprogressbar.add_widget(progressbar)
                        progressbar.value = 0
                        myapp.layout.remove_widget(myapp.btnCancel)
                        myapp.layout.add_widget(lbltimeleft)
                        myapp.layout.add_widget(myapp.btnCancel)
                        myapp.valuepredict = 0
                        myapp.trainclock = Clock.schedule_interval(myapp.layoutpbpredictnext, 1.0)
                        threading.Thread(target=predictfunc, args=(myapp,mydataset)).start()
            @mainthread
            def evshowfile(self, filechooser, filenameselected, details):
                self.dispatch('on_evshowfile', filechooser, filenameselected, details)

            def on_evshowfile(self, filechooser, filenameselected, details):
                try:
                    Filechooser.filechoosershowdatasetcont(filechooser, filenameselected, details)
                except Exception as e:  # work on python 3.x
                    print('filechoosershowdatasetcont: ' + str(e))


        self.filename = ''
        list_hidden.append(4)
        numlayers = str(len(list_hidden))
        self.myevent = MyEventDispatcher()

        self.n_folds = 5
        self.l_rate = 0.3
        self.n_epoch = 20

        self.onpredictlayout = False
        self.historyfile = False

        self.parentlayout = LayoutScreen()
        layoutneuronmainscreen = self.parentlayout.ids.idLayoutNeuron.ids.idLayoutNeuronMainScreen.ids.idrelativelayoutneuronmainscreen
        layoutlabels = self.parentlayout.ids.idLayoutNeuron.ids.idLayoutNeuronMainScreen.ids.idLabelsLayoutNeuronMainScreen
        self.layoutmainscreen = self.parentlayout.ids.idLayoutNeuron.ids.idLayoutNeuronMainScreen
        self.layout = self.parentlayout.ids.idLayoutMenu.ids.idBoxLayoutMenu
        self.btnlayout = self.parentlayout.ids.idLayoutMenu
        self.layoutinfoleft = self.parentlayout.ids.idboxlayoutleft
        self.layoutinforight = self.parentlayout.ids.idboxlayoutright

        self.hiddenlayeronelabelone = Label(text='[b]  Capa\noculta 1', markup=True, font_size=15, color=(1, 1, 1),
                                       size_hint=(None, None))
        self.hiddenlayeronelabeltwo = Label(text='[b]  Capa\noculta 2', markup=True, size_hint=(None, None), font_size=15,
                                       color=(1, 1, 1))
        self.hiddenlayeronelabelthree = Label(text='[b]   Capa\noculta 3', markup=True, size_hint=(None, None),
                                         font_size=15, color=(1, 1, 1))
        self.outputlabel = Label(text='Salidas', markup=True, font_size=17, size_hint=(None, None), color=(1, 1, 1))
        self.inputlabel = Label(text='Entradas', markup=True, font_size=17, size_hint=(None, None), color=(1, 1, 1))
        self.lblcurrentpage = Label(text=('[b]Páginas:[/b]' + ' ' + str("{:.0f}".format(currentpage)) + '/' + '0'), markup=True)
        self.lblfilename = ''
        self.btnplussidetext = Button(text='[b]>', markup=True, size_hint=(.2, .8),
                                 pos_hint={'center_x': .5, 'center_y': .5},
                                 background_normal='images/ButtonPlusP.png',
                                 background_down='images/ButtonDeactivated.png',
                                 background_disabled_normal='images/ButtonDeactivated.png', font_size=20)
        self.btnlesssidetext = Button(text='[b]<', markup=True, size_hint=(.2, .8),
                                 pos_hint={'center_x': 1, 'center_y': .5},
                                 background_normal='images/ButtonPlusP.png',
                                 background_down='images/ButtonDeactivated.png',
                                 background_disabled_normal='images/ButtonDeactivated.png', font_size=20)

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
        layoutsigmoidbox = BoxLayout(orientation='vertical', size_hint=(.1, 1))
        layoutsigmoidlbl = BoxLayout(orientation='vertical', size_hint=(.2, 1))
        layoutsettingsupright = BoxLayout(orientation='horizontal',spacing = 10, size_hint=(.3,1))
        self.layoutinfotop = self.parentlayout.ids.idlayouttop
        self.layoutinfomiddle = self.parentlayout.ids.idlayoutmiddle
        self.layoutinfodown = self.parentlayout.ids.idlayoutdown
        self.layoutbtninfo = self.parentlayout.ids.idlayoutlbldown
        layoutfilenameright = self.parentlayout.ids.idboxlayoutright
        self.lblfilename = self.parentlayout.ids.idlabelfilename


        self.btnPredict = Button(color=(1, 1, 1, 1), text='[b]TEST', markup = True,background_disabled_normal='images/ButtonDeactivated.png', font_size=18, size_hint=(.95, .9), pos_hint={'center_x':.5, 'center_y':.5}, background_normal='images/ButtonPlusP.png', background_down='images/ButtonDeactivated.png')
        self.btnProcessPredict = Button(color=(1, 1, 1, 1), text='[b]TEST', markup = True,background_disabled_normal='images/ButtonDeactivated.png', font_size=18, size_hint=(.95, .9), pos_hint={'center_x':.5, 'center_y':.5}, background_normal='images/ButtonPlusP.png', background_down='images/ButtonDeactivated.png')
        self.btnTrain = Button(text='[b]ENTRENAR',markup = True, background_normal='images/ButtonPlusP.png', background_down='images/ButtonDeactivated.png', size_hint=(.95,.9), background_disabled_normal='images/ButtonDeactivated.png',
                       pos_hint={'center_x':.5, 'center_y':.5}, color = (1,1,1,1), font_size=18)
        self.btnError = Button(text='[b]ERROR', markup=True, background_normal='images/ButtonPlusP.png',
                          background_down='images/ButtonDeactivated.png', size_hint=(.95, .9),
                          pos_hint={'center_x': .5, 'center_y': .5},background_disabled_normal='images/ButtonDeactivated.png', color=(1, 1, 1, 1), font_size=18)
        self.btnOk = Button(text='[b]LISTO', markup=True, background_normal='images/ButtonPlusB.png',
                          background_down='images/ButtonDeactivated.png', size_hint=(.8, .8),
                          pos_hint={'center_x': .5, 'center_y': .5}, color=(0, 0, 0, 1), font_size=18)
        self.btnOpen = Button(text='[b]ABRIR', markup=True, background_normal='images/ButtonPlusP.png',
                       background_down='images/ButtonPlusGDown.png', size_hint=(.8, .8),
                       pos_hint={'center_x': .5, 'center_y': .5}, color=(1, 1, 1, 1), font_size=18)
        self.btnBack = Button(text='[b]', markup=True, background_normal='images/ButtonBack.png',
                       background_down='images/ButtonBack.png', size_hint=(None, None), size=(60,60),
                       pos_hint={'center_x': 1, 'center_y': .5}, color=(1, 1, 1, 1),background_disabled_normal='images/ButtonBackDisabled.png')
        self.btnSettings = Button(background_normal='images/ButtonSettings.png', background_down='images/ButtonSettings.png', size_hint=(None, None), size=(60,60), pos_hint={'center_x':1, 'center_y':.5},background_disabled_normal='images/ButtonSettingsDisabled.png')
        self.btnnewinput=Button(color=(1,1,1,1), text='[b]Nueva entrada',markup = True, font_size=18, size_hint=(.95,.9),
                       pos_hint={'center_x':.5, 'center_y':.5},background_normal='images/ButtonPlusP.png', background_down='images/ButtonDeactivated.png')
        self.btnCancel = Button(color=(1,1,1,1), text='[b]CANCELAR',markup = True, font_size=18, size_hint=(.95,.9), pos_hint={'center_x':.5, 'center_y':.5},background_normal='images/ButtonPlusP.png', background_down='images/ButtonDeactivated.png')
        self.btnDetails = Button(color=(1,1,1,1), text='[b]MOSTRAR DETALLES',markup = True, font_size=18, size_hint=(.95,.9), pos_hint={'center_x':.5, 'center_y':.5},background_normal='images/ButtonPlusP.png', background_down='images/ButtonDeactivated.png')
        self.btnHistory = Button(background_normal='images/ButtonHistory.png', background_down='images/ButtonHistory.png', size_hint=(None, None), size=(60,60), pos_hint={'center_x':1, 'center_y':.5},background_disabled_normal='images/ButtonHistoryDisabled.png')
        self.btnUsar = Button(text='[b]CARGAR VALORES', markup=True, background_normal='images/ButtonPlusP.png',
                               background_down='images/ButtonDeactivated.png', size_hint=(.95, .9),
                               background_disabled_normal='images/ButtonDeactivated.png',
                               pos_hint={'center_x': .5, 'center_y': .5}, color=(1, 1, 1, 1), font_size=18)


        self.sigmoidbox = CheckBox(group='function', size_hint=(.1, 1), pos_hint={'center_x': .5, 'center_y': .5})
        lblsigmoidbox = Label(text='Sigmoid', pos_hint={'center_x': .5, 'center_y': 0})
        self.relubox = CheckBox(group='function', size_hint=(.1, 1), pos_hint={'center_x': .5, 'center_y': .5})
        lblrelubox = Label(text='Relu',  pos_hint={'center_x': .5, 'center_y': 0})
        self.sigmoidbox.active = True

        # Left Neuron Switcher
        self.btnplusneuron = Button(text = '[b]+',markup = True, size_hint=(.2,.8),pos_hint={'center_x':.5, 'center_y':.5},
                               background_normal='images/ButtonPlusP.png',
                               background_down='images/ButtonDeactivated.png',
                               background_disabled_normal='images/ButtonDeactivated.png', font_size=20)
        self.btnlessneuron = Button(text = '[b]-',markup = True,size_hint=(.2, .8),pos_hint={'center_x':.5, 'center_y':.5},
                               background_normal='images/ButtonPlusP.png',
                               background_down='images/ButtonDeactivated.png',
                               background_disabled_normal='images/ButtonDeactivated.png', font_size=20)
        self.lblhiddenlayer = Label(text=str(list_hidden[0]))
        self.lblneuron = Label(text='[b]Neuronas', markup=True)

        # Second neuron switcher
        self.btnplusneurontwo = Button(text = '[b]+',markup = True,size_hint=(.2, .8),pos_hint={'center_x':.5, 'center_y':.5},
                                  background_normal='images/ButtonPlusP.png',
                                  background_down='images/ButtonDeactivated.png',
                                  background_disabled_normal='images/ButtonDeactivated.png', font_size=20)
        self.btnlessneurontwo = Button(text = '[b]-',markup = True,size_hint=(.2, .8), pos_hint={'center_x':.5, 'center_y':.5},
                                  background_normal='images/ButtonPlusP.png',
                                  background_down='images/ButtonDeactivated.png',
                                  background_disabled_normal='images/ButtonDeactivated.png', font_size=20)
        self.lblhiddenlayertwo = Label(text='4')
        self.lblneurontwo = Label(text='[b]Neuronas', markup=True)

        # Right/Third neuron switcher
        self.btnplusneuronthree = Button(text = '[b]+',markup = True,size_hint=(.2, .8),pos_hint= {'center_x':1, 'center_y':.5},
                                    background_normal='images/ButtonPlusP.png',
                                    background_down='images/ButtonDeactivated.png',
                                    background_disabled_normal='images/ButtonDeactivated.png', font_size=20)
        self.btnlessneuronthree = Button(text = '[b]-',markup = True,size_hint=(.2, .8),pos_hint={'center_x':1, 'center_y':.5},
                                    background_normal='images/ButtonPlusP.png',
                                    background_down='images/ButtonDeactivated.png',
                                    background_disabled_normal='images/ButtonDeactivated.png', font_size=20)
        self.lblhiddenlayerthree = Label(text='4')
        self.lblneuronthree = Label(text='[b]Neuronas', markup=True)

        # Number of layers switcher
        self.btnlayerplusneuron = Button(text = '[b]+',markup = True,size_hint=(.2, .8),pos_hint={'center_x':1, 'center_y':.5},
                                    background_normal='images/ButtonPlusP.png',
                                    background_down='images/ButtonDeactivated.png',
                                    background_disabled_normal='images/ButtonDeactivated.png', font_size=20)
        self.btnlayerlessneuron = Button(text = '[b]-',markup = True,size_hint=(.2, .8),pos_hint={'center_x':1, 'center_y':.5},
                                    background_normal='images/ButtonPlusP.png',
                                    background_down='images/ButtonDeactivated.png',
                                    background_disabled_normal='images/ButtonDeactivated.png', font_size=20)
        self.lbllayer = Label(text=numlayers)
        self.lblnumlayers = Label(text='[b] Capas ocultas',markup = True, font_size=(11))
        lblblack = Widget(size_hint=(.1,.1))
        lblblacktwo = Widget(size_hint=(.1,.1))
        lblblackthree = Widget(size_hint=(.1,.1))
        canvas= Widget(size_hint=(.02,.1))
        visualize = Label(size_hint=(.1,.1))

        self.btnplusepoch = Button(text='[b]+', markup=True, size_hint=(.2, .8),
                              pos_hint={'center_x': 1, 'center_y': .5},
                              background_normal='images/ButtonPlusP.png',
                              background_down='images/ButtonDeactivated.png',
                              background_disabled_normal='images/ButtonDeactivated.png', font_size=20)
        self.btnlessepoch = Button(text='[b]-', markup=True, size_hint=(.2, .8),
                                    pos_hint={'center_x': 1, 'center_y': .5},
                                    background_normal='images/ButtonPlusP.png',
                                    background_down='images/ButtonDeactivated.png',
                                    background_disabled_normal='images/ButtonDeactivated.png', font_size=20)
        self.lblepoch = Label(text='[b] CICLOS:',markup=True, font_size=(11), pos_hint={'center_x':.5, 'center_y':.5})
        self.numepoch = Label(text=str(self.n_epoch))
        self.btnplusrate = Button(text='[b]+', markup=True, size_hint=(.2, .8),
                              pos_hint={'center_x': 1, 'center_y': .5},
                              background_normal='images/ButtonPlusP.png',
                              background_down='images/ButtonDeactivated.png',
                              background_disabled_normal='images/ButtonDeactivated.png', font_size=20)
        self.btnlessrate = Button(text='[b]-', markup=True, size_hint=(.2, .8),
                              pos_hint={'center_x': 1, 'center_y': .5},
                              background_normal='images/ButtonPlusP.png',
                              background_down='images/ButtonDeactivated.png',
                              background_disabled_normal='images/ButtonDeactivated.png', font_size=20)
        lblrate = Label(text='[b]RATIO:',markup=True, font_size=(11), pos_hint={'center_x': .5, 'center_y': .5})
        self.numrate = Label(text=str(self.l_rate))
        self.btnplusfolds = Button(text='[b]+', markup=True, size_hint=(.2, .8),
                             pos_hint={'center_x': 1, 'center_y': .5},
                             background_normal='images/ButtonPlusP.png',
                             background_down='images/ButtonDeactivated.png',
                             background_disabled_normal='images/ButtonDeactivated.png', font_size=20)
        self.btnlessfolds = Button(text='[b]-', markup=True, size_hint=(.2, .8),
                             pos_hint={'center_x': 1, 'center_y': .5},
                             background_normal='images/ButtonPlusP.png',
                             background_down='images/ButtonDeactivated.png',
                             background_disabled_normal='images/ButtonDeactivated.png', font_size=20)
        lblfolds = Label(text='[b]GRUPOS:',markup=True, font_size=(11), pos_hint={'center_x': .5, 'center_y': .5})
        self.numfolds = Label(text=str(self.n_folds))

        NeuralNetApp.restoreleftpanellist(self)

        self.btnplussidetext.disabled = True
        self.btnlesssidetext.disabled = True


        layoutsettingsup.add_widget(self.layoutsettingsupleft)
        layoutsettingsup.add_widget(lblblack)
        layoutsettingsup.add_widget(layoutsettingsupright)
        neuronlayout.add_widget(self.lblneuron)
        neuronlayout.add_widget(self.lblhiddenlayer)
        layoutsigmoidbox.add_widget(self.sigmoidbox)
        layoutsigmoidlbl.add_widget(lblsigmoidbox)
        layoutsigmoidbox.add_widget(self.relubox)
        layoutsigmoidlbl.add_widget(lblrelubox)
        self.layoutinfotop.add_widget(self.btnHistory)
        self.layoutsettingsupleft.add_widget(layoutsigmoidlbl)
        self.layoutsettingsupleft.add_widget(layoutsigmoidbox)
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



        self.neurontwolayout.add_widget(self.lblneurontwo)
        self.neurontwolayout.add_widget(self.lblhiddenlayertwo)
        self.neuronthreelayout.add_widget(self.lblneuronthree)
        self.neuronthreelayout.add_widget(self.lblhiddenlayerthree)

        # try:
        #     MyApp.restoreleftpanellist(self)
        #     mydata = leftpanellist[-1]
        #     currentpage = len(leftpanellist)
        #     #self.drawinglblblackkv(self.layoutinfomiddle, mydata)
        #     self.establishbtnpages()
        # except:
        #     pass


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
        self.btnHistory.bind(on_press=lambda x: self.historylayout())
        self.btnCancel.bind(on_press=lambda x:self.btncancelpressed())
        self.btnUsar.bind(on_press=lambda x: self.btnUsarpressed())
        self.btnOk.bind(on_press=lambda x:self.btnokpressed())
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
        self.btnDetails.bind(on_press=lambda x:self.btnDetailspressed())

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





if __name__ == '__main__':
    NeuralNetApp().run()

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



