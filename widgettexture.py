from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from math import sqrt
from kivy.graphics.texture import  *
from array import array
from kivy.graphics import *
from kivy.uix.popup import Popup


class WidgetTexture(Button):
    def __init__(self, imagebits, colorformat='luminance', **kwargs):
        super(WidgetTexture, self).__init__(**kwargs)

        self.label = "Label: {}".format(imagebits[-1])
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
            Color(1, 1, 1)
            self.rect = Rectangle(texture=self.mytexture, pos=self.pos, size=(self.sidesquare, self.sidesquare))

        self.bind(pos=self.update_rect, size=self.update_rect)
        self.bind(on_press=lambda x: self.btnWidgetTexture())

    def update_rect(self, *args):
        self.rect.pos = (self.center_x - self.rect.size[0] / 2, self.center_y - self.rect.size[1] / 2)

    def update_color(self, color):
        if color is False:
            with self.canvas:
                Color(1, 0, 0)
                self.rect = Rectangle(texture=self.mytexture, pos=self.pos, size=(self.sidesquare, self.sidesquare))
        else:
            pass

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
        self.boxpopupwt.rect.pos = (self.boxpopupwt.center_x - self.boxpopupwt.rect.size[0] / 2,
                                    self.boxpopupwt.center_y - self.boxpopupwt.rect.size[1] / 2)

    def btnWidgetTexture(self):
        self.boxpopupwt = BoxLayout(orientation='vertical')
        self.boxpopupwt.background_color = (0, 0, 0, 0)
        self.mytexture.blit_buffer(self.bufbytes, colorfmt=self.cf, bufferfmt='ubyte')
        with self.boxpopupwt.canvas:
            Color(1, 1, 1)
            self.boxpopupwt.rect = Rectangle(texture=self.mytexture, pos=self.boxpopupwt.pos,
                                             size=(self.sidesquare * 2, self.sidesquare * 2))
        self.boxpopupwt.bind(pos=self.update_rect_popup, size=self.update_rect_popup)

        popupwt = Popup(title=self.label, content=self.boxpopupwt, separator_color=(.6, 1, 1, 1),
                        auto_dismiss=True, size_hint=(.5, .5))
        popupwt.open()
