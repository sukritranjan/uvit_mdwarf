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
from scipy import interpolate as interp
import scipy.integrate

hc=1.98645e-9 #value of h*c in erg*nm



###Import Hu data
# solar=np.genfromtxt('./hu-code-sr-uvspectraproject/Data/solar00.txt', skip_header=0, skip_footer=0, unpack=False) #wavelength (nm), 1 AU flux (Watt m**-2 nm**-1)

###Import HIP23309 data
hip23309=np.genfromtxt('./hu-code-sr-uvspectraproject/Data/hip23309.txt', skip_header=0, skip_footer=0, unpack=False) #wavelength (nm), 1 AU-equivalent flux (Watt m**-2 nm**-1)

hip23309_ext=np.genfromtxt('./hu-code-sr-uvspectraproject/Data/hip23309_ext.txt', skip_header=0, skip_footer=0, unpack=False) #wavelength (nm), 1 AU-equivalent flux (Watt m**-2 nm**-1)

#### hip23309_rebinned=np.genfromtxt('./hu-code-sr-uvspectraproject/Data/hip23309_rebinned.txt', skip_header=0, skip_footer=0, unpack=False) #wavelength (nm), 1 AU-equivalent flux (Watt m**-2 nm**-1)

# ###Import TRAPPIST-1 data
# trappist1=np.genfromtxt('./hu-code-sr-uvspectraproject/Data/trappist-1_00.txt', skip_header=0, skip_footer=0, unpack=False) #wavelength (nm), 1 AU-equivalent flux (Watt m**-2 nm**-1)

# ###Import GJ 876 data
# gj876=np.genfromtxt('./hu-code-sr-uvspectraproject/Data/gj876_00.txt', skip_header=0, skip_footer=0, unpack=False) #wavelength (nm), 1 AU-equivalent flux (Watt m**-2 nm**-1)

###Import GJ 832 data
gj832=np.genfromtxt('./hu-code-sr-uvspectraproject/Data/gj832_00.txt', skip_header=0, skip_footer=0, unpack=False) #wavelength (nm), 1 AU-equivalent flux (Watt m**-2 nm**-1)

###Import HD85512 data
hd85512=np.genfromtxt('./hu-code-sr-uvspectraproject/Data/hd85512_00.txt', skip_header=0, skip_footer=0, unpack=False) #wavelength (nm), 1 AU-equivalent flux (Watt m**-2 nm**-1)

# ###Import HD40307 data
# hd40307=np.genfromtxt('./hu-code-sr-uvspectraproject/Data/hd40307.txt', skip_header=0, skip_footer=0, unpack=False) #wavelength (nm), 1 AU-equivalent flux (Watt m**-2 nm**-1)

##############
###Rebin UVIT to coarser wavelength grid, maybe that will help with the offset issue?

##############
###Rescale GJ832, HD85512 to match HIP23309 130-170 nm
hip23309_func=interp.interp1d(hip23309[:,0], hip23309[:,1], kind='linear')
gj832_func=interp.interp1d(gj832[:,0], gj832[:,1], kind='linear')
hd85512_func=interp.interp1d(hd85512[:,0], hd85512[:,1], kind='linear')

tot_hip23309_fuv=scipy.integrate.quad(hip23309_func, 130.0, 170.0, epsabs=0., epsrel=1.e-3, limit=10000)[0]
tot_gj832_fuv=scipy.integrate.quad(gj832_func, 130.0, 170.0, epsabs=0., epsrel=1.e-3, limit=10000)[0]
tot_hd85512_fuv=scipy.integrate.quad(hd85512_func, 130.0, 170.0, epsabs=0., epsrel=1.e-3, limit=10000)[0]

rescaled_gj832=np.copy(gj832)
rescaled_gj832[:,1]=(tot_hip23309_fuv/tot_gj832_fuv)*gj832[:,1]

rescaled_hd85512=np.copy(hd85512)
rescaled_hd85512[:,1]=(tot_hip23309_fuv/tot_hd85512_fuv)*hd85512[:,1]
np.savetxt('./hu-code-sr-uvspectraproject/Data/gj832_rescaled_00.txt', rescaled_gj832, fmt='%5.6f\t%1.6e', newline='\n')
np.savetxt('./hu-code-sr-uvspectraproject/Data/hd85512_rescaled_00.txt', rescaled_hd85512, fmt='%5.6f\t%1.6e', newline='\n')


###Plot to check
fig, ax=plt.subplots(1, figsize=(8., 6.), sharex=True)
markersizeval=5.


# ax.plot(solar[:,0], solar[:,1], linewidth=2, linestyle='-', marker='o', markersize=5, color='gold', label='Sun')

# ax.plot(trappist1[:,0], trappist1[:,1], linewidth=1, linestyle='-', marker='d', markersize=3, color='darkred', label='TRAPPIST-1')
# ax.plot(gj876[:,0], gj876[:,1], linewidth=1, linestyle='-', marker='d', markersize=3, color='red', label='GJ876')
ax.plot(hip23309_ext[:,0], hip23309_ext[:,1], linewidth=2, linestyle='-', marker='.', markersize=5, color='black', label='HIP23309 (HST)')
ax.plot(hip23309[:,0], hip23309[:,1], linewidth=2, linestyle='--', marker='o', markersize=5, color='darkgrey', label='HIP23309 (UVIT)')
# ax.plot(hip23309_rebinned[:,0], hip23309_rebinned[:,1], linewidth=2, linestyle='--', marker='o', markersize=5, color='gold', label='HIP23309 (UVIT-rebinned)')

ax.plot(gj832[:,0], gj832[:,1], linewidth=1, linestyle='-', marker='d', markersize=3, color='darkred', label='GJ832')
ax.plot(rescaled_gj832[:,0], rescaled_gj832[:,1], linewidth=1, linestyle='--', marker='d', markersize=3, color='red', label='GJ832-Rescaled')

ax.plot(hd85512[:,0], hd85512[:,1], linewidth=1, linestyle='-', marker='d', markersize=3, color='forestgreen', label='HD85512')
ax.plot(rescaled_hd85512[:,0], rescaled_hd85512[:,1], linewidth=1, linestyle='--', marker='d', markersize=3, color='lightgreen', label='HD85512-Rescaled')

# ax.plot(hd40307[:,0], hd40307[:,1], linewidth=1, linestyle='-', marker='d', markersize=3, color='blue', label='HD40307')





ax.legend(ncol=2, loc='best', fontsize=14)
ax.set_xscale('linear')
#ax.set_ylim([0., 120.])

ax.set_xscale('linear')
ax.set_xlabel('Wavelength (nm)', fontsize=16)
ax.set_xlim([115., 315.])
ax.set_yscale('log')
ax.set_ylim([1.0E-6, 1.0E1])
ax.set_ylabel(r'Flux (Watt m$^{-2}$ nm$^{-1}$)', fontsize=16)
ax.yaxis.set_tick_params(labelsize=15)
ax.xaxis.set_tick_params(labelsize=15)
plt.savefig('./plot_stellarinputs.pdf', orientation='portrait', format='pdf')


