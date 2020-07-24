from kivy.properties import NumericProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from math import sqrt
from kivy.graphics import *
from kivy.uix.gridlayout import GridLayout
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.uix.checkbox import CheckBox
import os
from os.path import dirname, join, basename, exists, expanduser
from kivy.animation import Animation
import threading
from kivy.core.window import Window

from loaddataset import showdataset
from widgettexture import WidgetTexture


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

    def __init__(self, callerclass, **kwargs):
        super(Filechooser, self).__init__(**kwargs)
        self.callerclass = callerclass

    def update_parent_popup(self, prevpopup):
        self.prevpopup = prevpopup

    def myselected(self, path, filenameselected):
        self.ids.text_input.text = basename(filenameselected)

    def filechoosershowdatasetcont(self, filenameselected, details):

        self.popup.dismiss()

        if self.mytexturedataset == None:
            containererror = BoxLayout(orientation='vertical')
            popuperror = Popup(title='DATASET:', content=containererror, separator_color=(.6,1,1, 1),
                              auto_dismiss=False, size_hint=(.5,.5))
            lblerror = Label(text='{} : Formato de archivo no válido' .format(filenameselected))
            popuperrorbtn = Button(text='Aceptar', on_press=popuperror.dismiss, background_normal='images/ButtonPlusP.png',
                              background_down='images/ButtonDeactivated.png', background_disabled_normal='images/ButtonDeactivated.png',
                              pos_hint={'center_x': .5, 'center_y': .5}, size_hint=(1,.5))
            containererror.add_widget(lblerror)
            containererror.add_widget(popuperrorbtn)
            popuperror.open()
        else:
            self.pagenum = 1
            self.pags = 0
            containerbox = BoxLayout(orientation='vertical', spacing = 10)
            popupshow = Popup(title='DATASET:', content=containerbox, separator_color=(.6,1,1, 1),
                              auto_dismiss=False)
            popupbtn = Button(text='Aceptar', on_press=popupshow.dismiss, background_normal='images/ButtonPlusP.png',
                              background_down='images/ButtonDeactivated.png', background_disabled_normal='images/ButtonDeactivated.png',
                              pos_hint={'center_x': .5, 'center_y': .5}, size_hint=(.4,1))
            self.mainlayout = GridLayout(cols=1)
            middlelayout = BoxLayout(orientation = 'horizontal')
            self.btnnextpage = Button(text='[b]>', markup=True, size_hint=(.05, .2),
                             pos_hint={'center_x': 1, 'center_y': .5},
                             background_normal='images/ButtonPlusP.png',
                             background_down='images/ButtonDeactivated.png',
                             background_disabled_normal='images/ButtonDeactivated.png', font_size=20)
            self.btnpreviouspage = Button(text='[b]<', markup=True, size_hint=(.05, .2),
                                 pos_hint={'center_x': 1, 'center_y': .5},
                                 background_normal='images/ButtonPlusP.png',
                                 background_down='images/ButtonDeactivated.png',
                                 background_disabled_normal='images/ButtonDeactivated.png', font_size=20)
            mybtnlayout = BoxLayout(orientation='horizontal', size_hint=(1,.1))
            typelayout = GridLayout( cols = 4)
            infolayout = BoxLayout(orientation='horizontal', size_hint=(1,.1))
            self.imagedatabox = CheckBox(group = 'typeobject', size_hint=(.4,1))
            lblimagedatabox = Label(text='Dataset de imágenes',halign="left")
            lblimagedatabox.bind(size=lblimagedatabox.setter('text_size'))
            self.numberdatabox = CheckBox(group = 'typeobject', size_hint=(.4,1))
            lblnumberdatabox = Label(text='Dataset de números',halign="left")
            lblnumberdatabox.bind(size=lblnumberdatabox.setter('text_size'))
            self.outputfirstbox = CheckBox(group='output', size_hint=(.4,1))
            lbloutputfirstbox = Label(text='Salidas en la primera columna',halign="left")
            lbloutputfirstbox.bind(size=lbloutputfirstbox.setter('text_size'))
            self.outputlastbox = CheckBox(group='output', size_hint=(.4,1))
            lbloutputlastbox = Label(text='Salidas en la última columna',halign="left")
            lbloutputlastbox.bind(size=lbloutputlastbox.setter('text_size'))

            self.lblpage = Label(text='', markup = True, pos_hint={'center_x': .5, 'center_y': .5},
                            size_hint=(1, .1), color=(.7,.9,.9))
            outputslast = ([row[-1] for row in self.mytexturedataset])
            setoutputslast = set(outputslast)
            self.btnnoutputs = Button(text='[b]nº salidas: [/b]' + str(len(setoutputslast)),markup = True, pos_hint={'center_x': .5, 'center_y': .5},
                            size_hint=(1, .9), background_normal='images/ButtonPlusP.png',
                             background_down='images/ButtonDeactivated.png')
            lblnelements = Label(text='[b]nº elementos: [/b]' + str(len(outputslast)), markup = True, pos_hint={'center_x': .5, 'center_y': .5},
                            size_hint=(.8, .1))

            self.lblelementsize = Label(text='', markup = True, pos_hint={'center_x': .5, 'center_y': .5},
                            size_hint=(.8, .1))




            self.outputlastbox.active = True
            self.outputchanged = False
            self.btnpreviouspage.disabled = True
            self.btnnextpage.disabled = False
            self.btnnextpage.bind(on_press=lambda x:self.numberdatasetfunc('plusbtn',False))
            self.btnpreviouspage.bind(on_press=lambda x:self.numberdatasetfunc('lessbtn',False))
            self.outputfirstbox.bind(on_press=lambda x:self.outputdatasetfunc())
            self.numberdatabox.bind(active=lambda x,y:self.resetandnumberdatasetfunc(''))
            self.btnnoutputs.bind(on_press=lambda x:self.outputoptionsfunc())

            if details is False:
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
            if details is False:
                infolayout.add_widget(self.btnnoutputs)
            infolayout.add_widget(lblnelements)
            infolayout.add_widget(self.lblelementsize)
            infolayout.add_widget(self.lblpage)
            containerbox.add_widget(mybtnlayout)
            popupshow.open()

            if details is False:
                if self.isimage is not None:
                    self.imagedatabox.active = True
                else:
                    self.numberdatabox.active = True

            self.numberdatasetfunc('', details)


    def filechoosershowdataset(self, path, filenameselected, eventp, details):
        filename = os.path.join(path, filenameselected)
        self.callerclass.filename = filename
        self.callerclass.lblfilename.text = ('[b] Archivo: [/b]' + '[i]' + filenameselected)
        self.optionaloutput, self.optionalimage, self.isimage, self.mytexturedataset = showdataset(filename)
        eventp.evshowfile(self, filenameselected, details)

    def numberdatasetfunc(self, plusless, details):
        global outputimageslist, inputimagelist

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
            self.lblelementsize.text = '[b]tamaño de los elementos: [/b]' + str(len(self.mytexturedataset[0]))
            self.lblpage.text = '[b]Página: [/b]' + str(self.pagenum) + '/' + str(self.pags)

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
            if self.isimage is 'rgb':
                points = sqrt((len(self.mytexturedataset[0]) - 1) / 3)
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
            self.lblelementsize.text = '[b]tamaño de las imágenes: [/b]' + str(int(points)) + ' x ' + str(int(points))
            if pagsaux != pagsfloat:
                self.pags += 1

            self.lblpage.text = '[b]Página: [/b]' + str(self.pagenum) + '/' + str(self.pags)
            for rowtexture in range(nitemsperpage): #in range cols * rows
                index = rowtexture+(self.pagenum-1)*nitemsperpage
                if index >= len(self.mytexturedataset):
                    break
                imagedata = WidgetTexture(self.mytexturedataset[index], self.isimage)
                self.mainlayout.add_widget(imagedata)
                if details is not False:
                    imagedata.update_color(details[rowtexture])
            outputslast = ([row[-1] for row in self.mytexturedataset])
            setoutputslast = sorted(set(outputslast))
            self.myoutputimageslist = list()
            for number in setoutputslast:
                for row in self.mytexturedataset:
                    if row[-1] == number:
                        self.myoutputimageslist.append(row)
                        break
            if len(self.mytexturedataset) > 100:
                self.callerclass.inputimagelist = list(self.mytexturedataset[10:100])
            else:
                self.callerclass.inputimagelist = list(self.mytexturedataset[:-1])

            self.callerclass.outputimageslist = self.myoutputimageslist.copy()
        self.btnactivation()

    def resetandnumberdatasetfunc(self, plusless):
        self.pagenum = 1
        self.numberdatasetfunc(plusless, False)

    def outputdatasetfunc(self):

        if self.outputfirstbox.active == True and self.outputchanged == False:
            self.optionaloutput, self.optionalimage, self.isimage, self.mytexturedataset = showdataset(filenameselected)
            self.numberdatasetfunc('', False)
            self.outputchanged = True
        elif self.outputlastbox.active == True and self.outputchanged == True:
            self.mytexturedataset = datasetoutputlastcolumn(self.mytexturedataset)
            self.numberdatasetfunc('', False)
            self.outputchanged = False
        else:
            pass

    def outputoptionsfunc(self):
        containeroutput = BoxLayout(orientation='vertical')
        outputbox = GridLayout()
        outputbox.clear_widgets()
        self.popupoutputs = Popup(title='SALIDAS:', content=containeroutput, separator_color=(.6,1,1, 1), size_hint=(.8,.4))
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
            number = 0
            for row in self.myoutputimageslist:
                boxob = BoxLayout(orientation='vertical')
                image = WidgetTexture(row, self.isimage)
                boxob.add_widget(image)
                lbloutput = Label(text=str(number))
                number += 1
                boxob.add_widget(lbloutput)
                outputbox.add_widget(boxob)
        else:
            for number in setoutputslast:
                lbloutput = Label(text=str(number))
                outputbox.add_widget(lbloutput)


        containeroutput.add_widget(outputbox)

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

        popuploading = PopupLoading()
        self.popup = Popup(title=filenameselected, content=popuploading,
                      size_hint=(0.9, 0.9), auto_dismiss=False, separator_color=(.67, .57, .69))
        self.popup.open()

        popuploading.start()

        threading.Thread(target=Filechooser.filechoosershowdataset, args=(self, path, filenameselected, self.callerclass.myevent, False)).start()
        self.prevpopup.dismiss()

    def BtnImagesFncn(self,btnimages):
        if btnimages.text == 'Dataset de números':
            btnimages.text = 'Dataset de imágenes'
        else:
            btnimages.text = 'Dataset de números'
    def BtnOutputFncn(self,btnimages):
        if btnimages.text == 'Salidas en la última columna':
            btnimages.text = 'Salidas en la primera columna'
        else:
            btnimages.text = 'Salidas en la última columna'


    def popupdismiss(self):
        self.prevpopup.dismiss()

