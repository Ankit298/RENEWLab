# -*- coding: utf-8 -*-
"""
Created on Sun Feb 14 12:23:29 2021

@author: ankit
"""

import utils
import numpy as np
filepath = 'trial_data.hdf5'
frames = 4000

zero_subcarrier_indices,transmit_data_dim,transmit_data = utils.convert_transmit_dimensions_single(filepath,frames)
print("Dimensions of transmitted data (1 ofdm symbol,1subcarrier) is:")
print(transmit_data_dim)
print("Snippet of transmitted data:")
print(transmit_data[0])

data_dim,data = utils.read_uplink_data(filepath)
print("Dimensions of received data are:")
print(data_dim)
print("Snippet of the data:")
print(data[0])

y_dim,y = utils.reduce_data_single(data,zero_subcarrier_indices)
print("Revised Dimensions of received data (1 ofdm symbol,1subcarrier) are:")
print(y_dim)
print("Snippet of the revised data:")
print(y[0])

csi_dim,csi = utils.get_channel(filepath)
print("Dimensions of csi are:")
print(csi_dim)
print("Snippet of the csi:")
print(csi[0])

channel_dim,channel,real_channel,imag_channel = utils.convert_channel_dimensions_single(csi)
print("Revised Dimensions of Channel (1 ofdm symbol,1 subcarrier) are:")
print(channel_dim)
print("Snippet of the channel:")
print(channel[0])
print(real_channel[0])
print(imag_channel[0])

zero_subcarrier_indices,transmit_data_dim,transmit_data = utils.convert_transmit_dimensions_all(filepath)
print("Dimensions of transmitted data (all ofdm symbols, all subcarriers) is:")
print(transmit_data_dim)
print("Snippet of transmitted data:")
print(transmit_data[0])

y_dim,y = utils.reduce_data_all(data,zero_subcarrier_indices)
print("Revised Dimensions of received data for all subcarriers are:")
print(y_dim)
print("Snippet of the revised data:")
print(y[0])

channel_dim,channel,real_channel,imag_channel = utils.convert_channel_dimensions_all(csi)
print("Revised Dimensions of Channel (all ofdm symbols, all subcarriers) are:")
print(channel_dim)
print("Snippet of the channel:")
print(channel[0])
print(real_channel[0])
print(imag_channel[0])





