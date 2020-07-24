from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from math import sqrt
from kivy.graphics.texture import  *
from array import array
from kivy.graphics import *
from kivy.uix.popup import Popup


class WidgetTexture(Button):
    def __init__(self, imagebits, colorformat='luminance', predicted=0, rightvalue=0, imagebitright=None, **kwargs):
        super(WidgetTexture, self).__init__(**kwargs)

        self.predicted = predicted
        self.rightvalue = rightvalue
        self.imagebitright = imagebitright

        self.label = "Etiqueta: {}".format(imagebits[-1])
        if self.predicted != self.rightvalue:
            self.label += ", Etiqueta predicha: {}".format(predicted)
        self.cf = colorformat
        if self.cf is 'rgb':
            self.sidesquare = sqrt((len(imagebits) - 1) / 3)
        else:
            self.sidesquare = sqrt(len(imagebits) - 1)
        self.mytexture = Texture.create(size=(self.sidesquare, self.sidesquare))

        buf = list(imagebits)
        buf.pop(-1)
        arr = list()
        for value in buf:
            arr.append(int(value))

        self.bufbytes = array('B', arr)
        self.mytexture.blit_buffer(self.bufbytes, colorfmt=self.cf, bufferfmt='ubyte')
        self.mytexture.flip_vertical()

        self.background_color = (0, 0, 0, 0)
        with self.canvas:
            if self.predicted != self.rightvalue:
                Color(1,0,0)
                self.borderrect = Rectangle(pos=(self.pos[0], self.pos[1]), size=(self.sidesquare + 2, self.sidesquare + 2))
            Color(1, 1, 1)
            self.rect = Rectangle(texture=self.mytexture, pos=(self.pos[0] + 1, self.pos[1] + 1), size=(self.sidesquare, self.sidesquare))

        self.bind(pos=self.update_rect, size=self.update_rect)
        self.bind(on_press=lambda x: self.btnWidgetTexture())



    def update_rect(self, *args):
        self.rect.pos = ((self.center_x - self.rect.size[0] / 2) + 1, (self.center_y - self.rect.size[1] / 2) + 1)
        if self.predicted != self.rightvalue:
            self.borderrect.pos = (self.center_x - self.rect.size[0] / 2, self.center_y - self.rect.size[1] / 2)

    def update_texture(self, imagebits):
        buf = list(imagebits)
        buf.pop(-1)
        arr = list()
        for value in buf:
            arr.append(int(value))

        self.bufbytes = array('B', arr)
        self.mytexture.blit_buffer(self.bufbytes, colorfmt=self.cf, bufferfmt='ubyte')
        self.mytexture.flip_vertical()

    def update_rect_popup(self, *args):
        if self.predicted != self.rightvalue:
            self.boxpopupwt.rectright.pos = (self.boxpopupwt.center_x,
                                            self.boxpopupwt.center_y - self.boxpopupwt.rect.size[1] / 2)
            centered=1
        else:
            centered=2
        self.boxpopupwt.rect.pos = (self.boxpopupwt.center_x - self.boxpopupwt.rect.size[0] / centered,
                                    self.boxpopupwt.center_y - self.boxpopupwt.rect.size[1] / 2)

    def btnWidgetTexture(self):
        self.boxpopupwt = BoxLayout(orientation='vertical')
        self.boxpopupwt.background_color = (0, 0, 0, 0)
        self.mytexture.blit_buffer(self.bufbytes, colorfmt=self.cf, bufferfmt='ubyte')
        with self.boxpopupwt.canvas:
            if self.predicted != self.rightvalue:
                if self.cf is 'rgb':
                    self.sidesquare = sqrt((len(self.imagebitright) - 1) / 3)
                else:
                    self.sidesquare = sqrt(len(self.imagebitright) - 1)
                self.mytextureright = Texture.create(size=(self.sidesquare, self.sidesquare))

                buf = list(self.imagebitright)
                buf.pop(-1)
                arr = list()
                for value in buf:
                    arr.append(int(value))

                self.bufbytesright = array('B', arr)
                self.mytextureright.blit_buffer(self.bufbytesright, colorfmt=self.cf, bufferfmt='ubyte')
                self.mytextureright.flip_vertical()

                self.boxpopupwt.rectright = Rectangle(texture=self.mytextureright,
                                                 pos=(self.boxpopupwt.pos[0] + self.sidesquare * 2, self.boxpopupwt.pos[1]),
                                                 size=(self.sidesquare * 2, self.sidesquare * 2))
            Color(1, 1, 1)
            self.boxpopupwt.rect = Rectangle(texture=self.mytexture, pos=self.boxpopupwt.pos,
                                             size=(self.sidesquare * 2, self.sidesquare * 2))
        self.boxpopupwt.bind(pos=self.update_rect_popup, size=self.update_rect_popup)

        popupwt = Popup(title=self.label, content=self.boxpopupwt, separator_color=(.6, 1, 1, 1),
                        auto_dismiss=True, size_hint=(.5, .5))
        popupwt.open()
