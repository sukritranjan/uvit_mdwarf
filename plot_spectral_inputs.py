#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 16:15:44 2019

@author: sukrit

How many NUV photons?
"""

########################
###Import useful libraries
########################
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import pdb

hc=1.98645e-9 #value of h*c in erg*nm



###Import Hu data
solar=np.genfromtxt('./hu-code-sr-uvspectraproject/Data/solar00.txt', skip_header=0, skip_footer=0, unpack=False) #wavelength (nm), 1 AU flux (Watt m**-2 nm**-1)

###Import HIP23309 data
hip23309=np.genfromtxt('./hu-code-sr-uvspectraproject/Data/hip23309.txt', skip_header=0, skip_footer=0, unpack=False) #wavelength (nm), 1 AU-equivalent flux (Watt m**-2 nm**-1)

###Import TRAPPIST-1 data
trappist1=np.genfromtxt('./hu-code-sr-uvspectraproject/Data/trappist-1_00.txt', skip_header=0, skip_footer=0, unpack=False) #wavelength (nm), 1 AU-equivalent flux (Watt m**-2 nm**-1)

###Import GJ 876 data
gj876=np.genfromtxt('./hu-code-sr-uvspectraproject/Data/gj876_00.txt', skip_header=0, skip_footer=0, unpack=False) #wavelength (nm), 1 AU-equivalent flux (Watt m**-2 nm**-1)

###Import GJ 832 data
gj832=np.genfromtxt('./hu-code-sr-uvspectraproject/Data/gj832_00.txt', skip_header=0, skip_footer=0, unpack=False) #wavelength (nm), 1 AU-equivalent flux (Watt m**-2 nm**-1)

###Import HD85512 data
hd85512=np.genfromtxt('./hu-code-sr-uvspectraproject/Data/hd85512_00.txt', skip_header=0, skip_footer=0, unpack=False) #wavelength (nm), 1 AU-equivalent flux (Watt m**-2 nm**-1)

###Import HD40307 data
hd40307=np.genfromtxt('./hu-code-sr-uvspectraproject/Data/hd40307.txt', skip_header=0, skip_footer=0, unpack=False) #wavelength (nm), 1 AU-equivalent flux (Watt m**-2 nm**-1)


###Plot to check
fig, ax=plt.subplots(1, figsize=(8., 6.), sharex=True)
markersizeval=5.

ax.plot(hip23309[:,0], hip23309[:,1], linewidth=2, linestyle='-', marker='o', markersize=5, color='black', label='HIP23309')
ax.plot(solar[:,0], solar[:,1], linewidth=2, linestyle='-', marker='o', markersize=5, color='gold', label='Sun')

ax.plot(trappist1[:,0], trappist1[:,1], linewidth=1, linestyle='-', marker='d', markersize=3, color='darkred', label='TRAPPIST-1')
ax.plot(gj876[:,0], gj876[:,1], linewidth=1, linestyle='-', marker='d', markersize=3, color='red', label='GJ876')
ax.plot(gj832[:,0], gj832[:,1], linewidth=1, linestyle='-', marker='d', markersize=3, color='orange', label='GJ832')
ax.plot(hd85512[:,0], hd85512[:,1], linewidth=1, linestyle='-', marker='d', markersize=3, color='forestgreen', label='HD85512')
ax.plot(hd40307[:,0], hd40307[:,1], linewidth=1, linestyle='-', marker='d', markersize=3, color='blue', label='HD40307')




ax.legend(ncol=2, loc='upper left')
ax.set_xscale('linear')
ax.set_xlabel('Wavelength (nm)')
ax.set_yscale('log')
ax.set_ylabel(r'Flux (Watt m$^{-2}$ nm$^{-1}$)')
#ax.set_ylim([0., 120.])

ax.set_xscale('linear')
ax.set_xlabel('Wavelength (nm)')
ax.set_xlim([100., 400.])
ax.set_yscale('log')
plt.savefig('./plot_stellarinputs.pdf', orientation='portrait', format='pdf')


