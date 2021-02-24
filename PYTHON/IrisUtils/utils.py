# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 08:57:51 2021

@author: ankit
"""

# y: Receieved Data
# N: Number of received antennas
# s: Samples per symbol*2
#y_: Received Data in dimensions which can be fed to an ML model

import h5py
import numpy as np
from hdf5_lib import hdf5_lib

# Read Uplink Data
def read_uplink_data(data_path):
    with h5py.File(data_path,'r') as f:
        y = np.array(f.get('Data')['UplinkData'])
    return y.shape,y

def convert_data_dimensions(data):
    """
        Converts the received data into dimensions required for detnet.
    """
    frames,_,_,N,s = data.shape
    y_ = np.zeros((frames,2*N))
    for frame in range(frames):
        real_array = []
        imaginary_array = []
        for antenna in range(N):
            symbols = data[frame,0,0,antenna]
            complex_symbols = [] # List storing complex symbols
            for i in range(0,len(symbols)-1,2):
                complex_symbols.append(np.complex(symbols[i],symbols[i+1]))
            complex_symbols = np.array(complex_symbols)
            complex_symbols = complex_symbols[160:960] # Ignore first and last 160 symbols
            complex_symbols = complex_symbols[:80] # Pick first 80
            complex_symbols = complex_symbols[16:] # Ignore first 16 symbols in first sub-carrier
            fdomain_symbols = np.fft.fft(complex_symbols)
            real_array.append(fdomain_symbols[0].real) # Extract real-part of first sub-carrier
            imaginary_array.append(fdomain_symbols[0].imag) # Extract imaginary part of first sub-carrier
        y_[frame] = np.array(real_array + imaginary_array)
    return y_.shape,y_

def get_channel(filepath,default_frame=100, cell_i=0, ofdm_sym_i=0, ant_i =0, user_i=0, ul_sf_i=0, subcarrier_i=10, offset=-1, dn_calib_offset=0, up_calib_offset=0, n_frm_st=0, thresh=0.001, deep_inspect=False, sub_sample=1):
    """
        Most of the code in this function is copied from the verify_hdf5 function in plot_hdf5.py.
        This function calls samps2csi in order to return the channel information.
    """
    hdf5 = hdf5_lib(filepath)
    data = hdf5.data
    metadata = hdf5.metadata
    
        
    pilot_samples = hdf5.pilot_samples
    uplink_samples = hdf5.uplink_samples
    
    # Check which data we have available
    data_types_avail = []
    pilots_avail = len(pilot_samples) > 0
    ul_data_avail = len(uplink_samples) > 0
    
    if pilots_avail:
        data_types_avail.append("PILOTS")
        print("Found Pilots!")
    if ul_data_avail:
        data_types_avail.append("UL_DATA")
        print("Found Uplink Data")
    
    # Empty structure
    if not data_types_avail:
        raise Exception(' **** No pilots or uplink data found **** ')
    
    # Retrieve attributes
    symbol_length = int(metadata['SYMBOL_LEN'])
    num_pilots = int(metadata['PILOT_NUM'])
    num_cl = int(metadata['CL_NUM'])
    cp = int(metadata['CP_LEN'])
    prefix_len = int(metadata['PREFIX_LEN'])
    postfix_len = int(metadata['POSTFIX_LEN'])
    z_padding = prefix_len + postfix_len
    offset = int(prefix_len)
    fft_size = int(metadata['FFT_SIZE'])
    pilot_type = metadata['PILOT_SEQ_TYPE'].astype(str)[0]
    ofdm_pilot = np.array(metadata['OFDM_PILOT'])
    reciprocal_calib = np.array(metadata['RECIPROCAL_CALIB'])
    symbol_length_no_pad = symbol_length - z_padding
    num_pilots_per_sym = ((symbol_length_no_pad) // len(ofdm_pilot))
    
    n_ue = num_cl
    frm_plt = min(default_frame, pilot_samples.shape[0] + n_frm_st)
    
    # Verify default_frame does not exceed max number of collected frames
    ref_frame = min(default_frame - n_frm_st, pilot_samples.shape[0])
    
    print("symbol_length = {}, offset = {}, cp = {}, prefix_len = {}, postfix_len = {}, z_padding = {}, pilot_rep = {}".format(symbol_length, offset, cp, prefix_len, postfix_len, z_padding, num_pilots_per_sym))
    
    samples = pilot_samples 
    num_cl_tmp = num_pilots  # number of UEs to plot data for
    
    samps_mat = np.reshape(
            samples[::sub_sample], (samples.shape[0], samples.shape[1], num_cl_tmp, samples.shape[3], symbol_length, 2))
    samps = (samps_mat[:, :, :, :, :, 0] +
            samps_mat[:, :, :, :, :, 1]*1j)*2**-15
    
    # Correlation (Debug plot useful for checking sync)
    amps = np.mean(np.abs(samps[:, 0, user_i, ant_i, :]), axis=1)
    pilot_frames = [i for i in range(len(amps)) if amps[i] > thresh]
    if len(pilot_frames) == 0: 
        print("no valid frames where found. Decision threshold (average pilot amplitude) was %f" % thresh)
        return 
    
    # Compute CSI from IQ samples
    # Samps: #Frames, #Cell, #Users, #Antennas, #Samples
    # CSI:   #Frames, #Cell, #Users, #Pilot Rep, #Antennas, #Subcarrier
    # For correlation use a fft size of 64
    print("*verify_hdf5(): Calling samps2csi with fft_size = {}, offset = {}, bound = {}, cp = {} *".format(fft_size, offset, z_padding, cp))
    csi, _ = hdf5_lib.samps2csi(samples, num_cl_tmp, symbol_length, fft_size = fft_size, offset = offset, bound = z_padding, cp = cp, sub = sub_sample, pilot_type=pilot_type)
    return csi.shape,csi

def convert_channel_dimensions(csi):
    """
        Converts read channel information from get_channel to get dimensions required for detnet.
    """
    csi = np.array(csi)
    frames,cell,users,pilot_rep,antennas,subcarrier = csi.shape 
    csi = np.moveaxis(csi,[2,3,5],[5,2,3])
    channel = np.zeros((frames,antennas,users),dtype = complex)
    for frame in range(frames):
        channel[frame] = csi[frame,0,0,0]
    real_channel = channel.real
    imag_channel = channel.imag
    return channel.shape,channel,real_channel,imag_channel

def convert_transmit_dimensions(filepath,frames):
    """
        Converts the transmitted data into dimensions required for detnet.
        This function is specific to 2 user systems, will have to be modified to deal with
        dynamic users.
    """
    hdf5 = hdf5_lib(filepath)
    metadata = hdf5.metadata
    ofdm_data0 = np.array(metadata['OFDM_DATA_CL0'])
    ofdm_data1 = np.array(metadata['OFDM_DATA_CL1'])
    def convert(ofdm_data):
        data = ofdm_data[0]
        data = data[:64]
        return data
    symbols0 = convert(ofdm_data0)
    symbol0 = np.delete(symbols0,np.where(abs(symbols0) == 0))[0]
    symbols1 = convert(ofdm_data1)
    symbol1 = np.delete(symbols1,np.where(abs(symbols1) == 0))[0]
    transmit_symbols = np.array([symbol0.real,symbol1.real]+[symbol0.imag,symbol1.imag])
    transmit_data = np.tile(transmit_symbols,(frames,1))
    return transmit_data.shape,transmit_data





        






    

        


    
    