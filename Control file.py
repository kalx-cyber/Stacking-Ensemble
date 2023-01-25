# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 14:24:07 2021

@author: Ramanzani S. Kalule
email: kramanzani@gmail.com
"""
import sys
r1 = [0,1,2,3,4,5,6]
def controller(r1):
    row=int(r1)
    if row == 0:
        print('Randomized search NN implementation machine learning') 
        import NNRandomSearch as NNR
        coc.CONCATENATE()
    elif row == 1:
        print('Randomized search multi-output machine learning') 
        import rssomult as RSMO
        vgg19.VGG19()
    elif row == 2:
        print('Randomized search single output porosity machine learning')
        import rssopor as RSSOPOR
        vgg19.VGG16()
    elif row == 3:
        print('Randomized search single output Permeability machine learning')
        import rssoper as RSSOPER
        rn50.RESNET50()
    elif row == 4:
        print('Stacking Regression single output porosity machine learning')
        import srsopor as SRSOPOR
        inc.INCEPTIONV3()
    elif row == 5:
        print('Stacking Regression single output Permeability machine learning')
        import srsoper as SRSOPER
        mnv2.MOBILENETV2()
    else:
        print('Stacking Regression multi-output machine learning')
        import srmo as SRMO
        cnn.CNN() 
for i in r1:
    controller(i)