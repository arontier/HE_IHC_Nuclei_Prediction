import numpy
from numpy import linalg
# import vigra

from optparse import OptionParser
import os
import sys
import numpy as np


class Color_Deconvolution(object):

    def __init__(self):
        self.params = {
            'image_type': 'HEDab'
        }

        return

    def log_transform(self, colorin):
        res = - 255.0 / numpy.log(256.0) * numpy.log((colorin + 1) / 256.0)
        res[res < 0] = 0.0
        res[res > 255.0] = 255.0
        return res

    def exp_transform(self, colorin):
        res = numpy.exp((255 - colorin) * numpy.log(255) / 255)
        res[res < 0] = 0.0
        res[res > 255.0] = 255.0
        return res

    def colorDeconv(self, imin):
        M_h_e_dab_meas = numpy.array([[0.650, 0.072, 0.268],
                                      [0.704, 0.990, 0.570],
                                      [0.286, 0.105, 0.776]])

        # [H,E]
        M_h_e_meas = numpy.array([[0.644211, 0.092789],
                                  [0.716556, 0.954111],
                                  [0.266844, 0.283111]])

        if self.params['image_type'] == "HE":
            # print "HE stain"
            M = M_h_e_meas
            M_inv = numpy.dot(linalg.inv(numpy.dot(M.T, M)), M.T)

        elif self.params['image_type'] == "HEDab":
            # print "HEDab stain"
            M = M_h_e_dab_meas
            M_inv = linalg.inv(M)

        else:
            # print "Unrecognized image type !! image type set to \"HE\" "
            M = numpy.diag([1, 1, 1])
            M_inv = numpy.diag([1, 1, 1])

        imDecv = numpy.dot(self.log_transform(imin.astype('float')), M_inv.T)
        imout = self.exp_transform(imDecv)

        return imout

    def colorDeconvHE(self, imin):
        """
        Does the opposite of colorDeconv
        """
        M_h_e_dab_meas = numpy.array([[0.650, 0.072, 0.268],
                                      [0.704, 0.990, 0.570],
                                      [0.286, 0.105, 0.776]])

        # [H,E]
        M_h_e_meas = numpy.array([[0.644211, 0.092789],
                                  [0.716556, 0.954111],
                                  [0.266844, 0.283111]])

        if self.params['image_type'] == "HE":
            # print "HE stain"
            M = M_h_e_meas

        elif self.params['image_type'] == "HEDab":
            # print "HEDab stain"
            M = M_h_e_dab_meas

        else:
            # print "Unrecognized image type !! image type set to \"HE\" "
            M = numpy.diag([1, 1, 1])
            M_inv = numpy.diag([1, 1, 1])

        imDecv = numpy.dot(self.log_transform(imin.astype('float')), M.T)
        imout = self.exp_transform(imDecv)


        return imout