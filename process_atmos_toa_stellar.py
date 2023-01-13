#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 17:10:28 2019

@author: sukrit
"""
import numpy as np
import matplotlib.pyplot as plt
import pdb

#####
###Import data
#####
def make_atmos_stellarspec():
    """
    Format ATMOS TOA irradiation from Peacock+2022 for inclusion into MEAC.
    """
    
    ###Import old data
    hu_file='./hu-code-sr-uvspectraproject/Data/trappist-1_00.txt'
    hu_wav, hu_toa=np.genfromtxt(hu_file, skip_header=0, skip_footer=0,usecols=(0,1), unpack=True) #units: nm, W m**-2 nm**-1

    ###Import new data
    atmos1_file='./Peacock2022-ATMOS/TRAPPIST1_1A_PHOENIX_synthspec_Lya_original.txt'
    atmos1_wav, atmos1_toa_mw=np.genfromtxt(atmos1_file, skip_header=0, skip_footer=0,usecols=(0,1), unpack=True)# units: nm, mW m**-2 nm**-1
    atmos1_toa=atmos1_toa_mw*1.e-3 #convert mW to W

    ###Import new data
    atmos2_file='./Peacock2022-ATMOS/TRAPPIST1_1A_PHOENIX_synthspec_Lya_fill3.txt'
    atmos2_wav, atmos2_toa_mw=np.genfromtxt(atmos2_file, skip_header=0, skip_footer=0,usecols=(0,1), unpack=True)# units: nm, mW m**-2 nm**-1
    atmos2_toa=atmos2_toa_mw*1.e-3 #convert mW to W

    
    #####
    ###Produce new composite file.
    #####    
    composite_toprint1=np.zeros((len(atmos1_wav), 2))
    composite_toprint1[:,0]=atmos1_wav #wavelengths in nm
    composite_toprint1[:,1]=atmos1_toa #TOA flux W m**-2 nm**-1

    composite_toprint2=np.zeros((len(atmos2_wav), 2))
    composite_toprint2[:,0]=atmos2_wav #wavelengths in nm
    composite_toprint2[:,1]=atmos2_toa #TOA flux W m**-2 nm**-1

    #####
    ###Plot and compare to past work
    #####
    
    fig, ax1=plt.subplots(1, figsize=(8,6), sharex=True)
    
    ax1.set_title('TOA Flux')
    ax1.plot(hu_wav, hu_toa, color='black', label='MEAC (Peacock+2019)')
    ax1.plot(composite_toprint1[:,0], composite_toprint1[:,1], color='purple', label='Peacock+2022 Original', linestyle='--')
    ax1.plot(composite_toprint2[:,0], composite_toprint2[:,1], color='gold', label='Peacock+2022 Fill3', linestyle=':')

    ax1.set_yscale('log')
    ax1.set_xlim([121., 122.])
    ax1.set_ylim([1.0E-6, 1.0])
    ax1.legend(loc=0, ncol=1, borderaxespad=0., fontsize=12)    
    plt.savefig('./Peacock2022-ATMOS/plot_meac_atmos_trappist1e_toa.pdf', orientation='portrait',format='pdf')

    
    #####
    ###save data
    #####
    np.savetxt('./Peacock2022-ATMOS/trappist-1_peacock2022_orig.txt', composite_toprint1, fmt='%5.6f\t%1.6e', newline='\n')
    np.savetxt('./Peacock2022-ATMOS/trappist-1_peacock2022_fill3.txt', composite_toprint2, fmt='%5.6f\t%1.6e', newline='\n')

make_atmos_stellarspec()