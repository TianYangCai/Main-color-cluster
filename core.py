from PIL import Image
import re
import urllib.request
import io
from Kmeans import Kmeans
from functools import partial
import math
import numpy as np


class ColorPalette():

    def __init__(self, im, k=10, show_clustering=False):
        if k > 10:
            raise ValueError("Maximum value for k is 10")
        self.k = k
        self.im = im
        self.show_clustering = show_clustering

        self.__color_palette()

    def __color_palette(self):

        #pixels = self.im.load()
        #width, height = self.im.size

        #width, height = self.im.shape[0], self.im.shape[1]

        pixel_dict = dict()

        _total_pixels = 0
        #计算每一种色彩的个数，并且创建对应的字典
        for i in range(len(self.im)):
            _total_pixels += 1
            #cpixel = pixels[col, row]
            if not isinstance(self.im[i],np.uint8):
                cpixel = tuple(self.im[i])
            else:
                cpixel = self.im[i]
            if cpixel in pixel_dict:
                pixel_dict[cpixel] = pixel_dict[cpixel] + 1
            else:
                pixel_dict[cpixel] = 1
        
        self.pixel_dict = pixel_dict
        self._total_pixels = _total_pixels
        sorted_tups = [(k, float(pixel_dict[k]/_total_pixels) * 100) for k in sorted(pixel_dict, key=pixel_dict.get, reverse=True)]

        # Getting estimated colors
        k_rgb = Kmeans(k=self.k, show_clustering=self.show_clustering).run(self.im)

        sorted_tups = self.__cross_reference(sorted_tups, k_rgb)
        sorted_tups = [e for e in sorted_tups if e[1] > 1]

        self.sorted_tups = sorted_tups
        self.dict = pixel_dict

    def __cross_reference(self, all_colors, estimated_colors):
        new_sorted_tups = dict()
        for curr_color in all_colors:
            closestColor = min(estimated_colors, key=partial(self.__color_difference, curr_color[0]))
            if closestColor in new_sorted_tups:
                new_sorted_tups[closestColor] = new_sorted_tups[closestColor] + float(self.pixel_dict[curr_color[0]]/self._total_pixels) * 100
            else:
                new_sorted_tups[closestColor] = float(self.pixel_dict[curr_color[0]]/self._total_pixels) * 100
        return [(k, new_sorted_tups[k]) for k in sorted(new_sorted_tups, key=new_sorted_tups.get, reverse=True)]

    def __color_difference(self, testColor, otherColor):
        difference = 0
        try:
            if isinstance(testColor, np.uint8):
                difference += abs(testColor - otherColor)
            else:
                difference += abs(testColor[0]-otherColor[0])
                difference += abs(testColor[1]-otherColor[1])
                difference += abs(testColor[2]-otherColor[2])
        except Exception as e:
            print("Error on color: {}\nError: {}".format(testColor, e.args))

        return difference

    def _rgb2hex(self, rgb_tuple):
        r, g, b = rgb_tuple
        return "#{:02x}{:02x}{:02x}".format(r,g,b)

    def __re_round(self, li):
        try:
            return int(round(li, 0))
        except TypeError:
            return type(li)(self.__re_round(x) for x in li)

    def get_top_colors(self, n=10, ratio=False, rounded=True, to_hex=False):
        if n > 10:
            raise ValueError("Max query is 10")

        sorted_tups = self.sorted_tups

        if rounded:
            sorted_tups = self.__re_round(sorted_tups)    

        if not ratio:
            if to_hex:
                sorted_tups = self.__re_round(sorted_tups)    
                hex_tups = [self._rgb2hex(f[0]) for f in sorted_tups[:n]]
                return hex_tups
            else:
                return [color[0] for color in sorted_tups[:n]]
        else:
            if to_hex:
                sorted_tups = self.__re_round(sorted_tups)    
                hex_tups = [(self._rgb2hex(f[0]), f[1]) for f in sorted_tups[:n]]
                return hex_tups
            else:
                return [color for color in sorted_tups[:n]]

    def get_color(self, index, ratio=False, rounded=True, to_hex=False):
        if index > 10:
            raise ValueError("Max query is 10")

        sorted_tups = self.sorted_tups

        if rounded:
            sorted_tups = self.__re_round(sorted_tups)    

        if not ratio:
            if to_hex:
                sorted_tups = self.__re_round(sorted_tups)    
                return self._rgb2hex(sorted_tups[index][0])
            else:
                return sorted_tups[index][0]
        else:
            if to_hex: 
                sorted_tups = self.__re_round(sorted_tups)    
                val = self._rgb2hex(sorted_tups[index][0])
                ratio = sorted_tups[index][1]
                return (val, ratio)
            else:
                return sorted_tups[index]
