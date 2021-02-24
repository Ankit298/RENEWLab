# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 12:23:29 2021

@author: ankit
"""

import utils
import numpy as np
filepath = 'trial_data.hdf5'

data_dim,data = utils.read_uplink_data(filepath)
print("Dimensions of received data are:")
print(data_dim)
print("Snippet of the data:")
print(data[0])

y_dim,y = utils.convert_data_dimensions(data)
print("Revised Dimensions of received data are:")
print(y_dim)
print("Snippet of the revised data:")
print(y[0])

csi_dim,csi = utils.get_channel(filepath)
print("Dimensions of csi are:")
print(csi_dim)
print("Snippet of the csi:")
print(csi[0])

channel_dim,channel,real_channel,imag_channel = utils.convert_channel_dimensions(csi)
print("Revised Dimensions of Channel are:")
print(channel_dim)
print("Snippet of the channel:")
print(channel[0])
print(real_channel[0])
print(imag_channel[0])

transmit_data_dim,transmit_data = utils.convert_transmit_dimensions(filepath,4000)
print("Dimensions of transmitted data is:")
print(transmit_data_dim)
print("Snippet of transmitted data:")
print(transmit_data[0])


