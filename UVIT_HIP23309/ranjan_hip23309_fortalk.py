#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 16:42:13 2022

@author: sukrit
"""

########################
###Control switches for plotting
########################
compare_hst_uvit_hip23309_data=False #Compare FUV UVIT measurements of HIP 23309 to FUV HST measurements of HIP23309
nuv_inter_orbit_stability=False #Plot stability of UVT NUV measurement orbit-by-orbit?
compare_hip23309_otherms_fuv=False #compare FUV UVIT measurements of HIP 23309 to measurements of other stars
compare_hip23309_otherms_nuv=False #compare FUV UVIT measurements of HIP 23309 to measurements of other stars
compare_hip23309_otherms=True #compare all-UV UVIT measurements of HIP 23309 to measurements of other stars

########################
###Import useful libraries
########################
from astropy.io import fits
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import scipy.integrate
from scipy import interpolate as interp
import coronagraph as cg
from scipy.ndimage import gaussian_filter1d


########################
###Define useful constants, all in CGS (via http://www.astro.wisc.edu/~dolan/constants.html)
########################

#Unit conversions
km2m=1.e3 #1 km in m
km2cm=1.e5 #1 km in cm
cm2km=1.e-5 #1 cm in km
amu2g=1.66054e-24 #1 amu in g
bar2atm=0.9869 #1 bar in atm
Pa2bar=1.e-5 #1 Pascal in bar
bar2Pa=1.e5 #1 bar in Pascal
deg2rad=np.pi/180.
bar2barye=1.e6 #1 Bar in Barye (the cgs unit of pressure)
barye2bar=1.e-6 #1 Barye in Bar
micron2m=1.e-6 #1 micron in m
micron2cm=1.e-4 #1 micron in cm
metricton2kg=1000. #1 metric ton in kg

#Fundamental constants
c=2.997924e10 #speed of light, cm/s
h=6.6260755e-27 #planck constant, erg/s
k=1.380658e-16 #boltzmann constant, erg/K
sigma=5.67051e-5 #Stefan-Boltzmann constant, erg/(cm^2 K^4 s)
R_earth=6371.*km2m#radius of earth in m
R_sun=69.63e9 #radius of sun in cm
AU=1.496e13#1AU in cm
pc=3.086e18 #1 pc in cm


########################
###Import processed data
########################

#######
###Stellar parameters
#######
d_hip23309=26.8*pc #Distance ot HIP 23309 in cm; Malo et al. 2014

d_gj832=4.9*pc #Distance to gj832 in cm; France et al. 2016 Table 1
d_gj667c=6.9*pc #Distance to gj667c in cm; France et al. 2016 Table 1
d_hd85512=11.2*pc #Distance to hd85512 in cm; France et al. 2016 Table 1
d_hd40307=12.9*pc #Distance to hd85512 in cm; France et al. 2016 Table 1


#######
###MUSCLES Data
######
# gj832 = fits.getdata('./hlsp_muscles_hst_stis_gj832_g230l_v22_component-spec.fits',1) #accesss via keywords WAVELENGTH (A), FLUX (erg/s/cm2/A) and ERROR (erg/s/cm2/A) ###M1.5V star
gj832 = fits.getdata('./LIT_DATA/hlsp_muscles_multi_multi_gj832_broadband_v22_adapt-const-res-sed.fits',1) #accesss via keywords WAVELENGTH (A), FLUX (erg/s/cm2/A) and ERROR (erg/s/cm2/A) ###M1.5V star
gj667c = fits.getdata('./LIT_DATA/hlsp_muscles_multi_multi_gj667c_broadband_v22_adapt-const-res-sed.fits',1) #accesss via keywords WAVELENGTH (A), FLUX (erg/s/cm2/A) and ERROR (erg/s/cm2/A) ###M1.5V star
hd85512 = fits.getdata('./LIT_DATA/hlsp_muscles_multi_multi_hd85512_broadband_v22_adapt-const-res-sed.fits',1) #accesss via keywords WAVELENGTH (A), FLUX (erg/s/cm2/A) and ERROR (erg/s/cm2/A) ###K6 star
hd40307 = fits.getdata('./LIT_DATA/hlsp_muscles_multi_multi_hd40307_broadband_v22_adapt-const-res-sed.fits',1) #accesss via keywords WAVELENGTH (A), FLUX (erg/s/cm2/A) and ERROR (erg/s/cm2/A) ###K2.5V star



#######
###Pineda HIP 23309 synthetic spectrum, somehow based on AU Mic
######
pineda_hip23309_aumic_wav, pineda_hip23309_aumic_flux=np.genfromtxt('./LIT_DATA/pineda_HIP23309_full_scaled.dat', skip_header=1, skip_footer=0,usecols=(0,1), unpack=True) #A, erg/s/cm2/nm #This is the version from AU Mic data, synthesized by Sebastian.

pineda_hip23309_aumic_luminosity=(pineda_hip23309_aumic_flux*4.0*np.pi*d_hip23309**2.0)*1.0E-1 #get luminosity, convert to  erg/s/cm2/A


#######
###FUSE HIP23309 data
######
hip_hst = pd.read_csv('./HIP23309_HST_FUVl.txt', names=['wavelength', 'flux', 'err']) #also A, erg/s/cm2/A, erg/s/cm2/A

#######
###UVIT HIP23309 data
######
HIP23309_FUV = pd.read_csv('./HIP23309_FUV_T01_207T01_9000001720_FUV_Grating1m2_crossdisp_50pixels_spec.dat', delimiter='\s+', names=['lambda', 'flux', 'flux_err']) #FUV data integrated across all 7 orbits. A, erg/s/cm2/A, erg/s/cm2/A


HIP23309_NUV = pd.read_csv('./HIP23309_NUV_T01_207T01_9000001720_NUV_Gratingm1_cross_disp_50pixels_spec.dat', delimiter='\s+', names=['lambda', 'flux', 'flux_err'])  #NUV data integrated across all 7 orbits. A, erg/s/cm2/A, erg/s/cm2/A


###Orbit-by-orbit NUV data. Units: A, erg/s/cm2/A, erg/s/cm2/A
HIP23309_NUV1 = pd.read_csv('./HIP23309_NUV_4168_T01_207T01_9000001720_NUV_Gratingm1_cross_disp_50pixels_spec.dat', delimiter='\s+', names=['lambda', 'flux', 'flux_err']) # start 10:16:44.111110888, end 10:47:18.081329640  2017-11-24, exposure 1778.054

HIP23309_NUV2 = pd.read_csv('./HIP23309_NUV_1019_T01_207T01_9000001720_NUV_Gratingm1_cross_disp_50pixels_spec.dat', delimiter='\s+', names=['lambda', 'flux', 'flux_err']) # start 11:54:11.160188880, end 12:25:06.162442784, exposure 1798.212

HIP23309_NUV3 = pd.read_csv('./HIP23309_NUV_7868_T01_207T01_9000001720_NUV_Gratingm1_cross_disp_50pixels_spec.dat', delimiter='\s+', names=['lambda', 'flux', 'flux_err']) # start 13:31:38.205353824, end 14:02:33.208607728, exposure 1798.101

HIP23309_NUV4 = pd.read_csv('./HIP23309_NUV_4205_T01_207T01_9000001720_NUV_Gratingm1_cross_disp_50pixels_spec.dat', delimiter='\s+', names=['lambda', 'flux', 'flux_err']) # start 15:09:04.742312768, end 15:39:59.219568656, exposure 1797.295

HIP23309_NUV5 = pd.read_csv('./HIP23309_NUV_1054_T01_207T01_9000001720_NUV_Gratingm1_cross_disp_50pixels_spec.dat', delimiter='\s+', names=['lambda', 'flux', 'flux_err']) # start 16:46:31.791574528, end 17:17:26.272828416, exposure 1797.37

HIP23309_NUV6 = pd.read_csv('./HIP23309_NUV_7904_T01_207T01_9000001720_NUV_Gratingm1_cross_disp_50pixels_spec.dat', delimiter='\s+', names=['lambda', 'flux', 'flux_err']) # start 18:23:58.840836224, end 18:54:53.322090112, exposure 1798.28

HIP23309_NUV7 = pd.read_csv('./HIP23309_NUV_6463_T01_207T01_9000001720_NUV_Gratingm1_cross_disp_50pixels_spec.dat', delimiter='\s+', names=['lambda', 'flux', 'flux_err']) # start 20:03:37.602328448, end 20:25:26.148615520, exposure 1260.055


########################
###Plot data quality checks
#Objective of these checks is to determine just how robust our measurements are.
########################

#######
###Compare against HST Data: FUV
######

if compare_hst_uvit_hip23309_data:
    ###Rebin HST data
    
    ##Get abscissa to rebin to.
    #Initialize
    uvit_wav_centers=HIP23309_FUV['lambda'].to_numpy() #centers of all UVIT data.
    uvit_wav_left=np.zeros(np.shape(uvit_wav_centers))
    uvit_wav_right=np.zeros(np.shape(uvit_wav_centers))
    
    #Get abscissa
    uvit_wav_left[1:]=0.5*(uvit_wav_centers[0:-1] + uvit_wav_centers[1:])
    uvit_wav_right[:-1]=0.5*(uvit_wav_centers[0:-1] + uvit_wav_centers[1:])
    uvit_wav_left[0]=uvit_wav_centers[0]-(uvit_wav_right[0]-uvit_wav_centers[0])
    uvit_wav_right[-1]=uvit_wav_centers[-1]+(uvit_wav_centers[-1]-uvit_wav_left[-1])
    
    #Downselect
    fuv_min=1300. #UVIT FUV not trustworthy below ~1300 A (Prasanta email 2/18/222)
    fuv_max=((hip_hst['wavelength'].to_numpy())[-1]) #Terminate where the HST data runs out. 
    inds=np.where((uvit_wav_left>=fuv_min) & (uvit_wav_right<=fuv_max))
    
    uvit_wav_left_cut=uvit_wav_left[inds]
    uvit_wav_right_cut=uvit_wav_right[inds]
    uvit_wav_centers_cut=uvit_wav_centers[inds]
    uvit_wav_deltas_cut=uvit_wav_right_cut-uvit_wav_left_cut
    
    hip23309_hst_smoothed=gaussian_filter1d(hip_hst['flux'], 14.63/0.6/2.385 )#FWHM of 14.63, /2.385 to get sigma, each STIS wavelength resolution element is 0.6

    hip23309_hst_smoothed_uvitbinned = cg.downbin_spec(hip23309_hst_smoothed,hip_hst['wavelength'], uvit_wav_centers_cut, dlam=uvit_wav_deltas_cut) #For some reason error-weighting doesn't work
    
    ###Rebin UVIT data to eliminate negative fluxes
    rebinfactor=4
    num_uvit=len(HIP23309_FUV['lambda'].to_numpy())
    
    num_uvit_rebinned=int(np.floor(num_uvit/rebinfactor))
    flux_uvit_rebinned=np.zeros(num_uvit_rebinned)
    fluxerr_uvit_rebinned=np.zeros(num_uvit_rebinned)
    wavs_uvit_rebinned=np.zeros(num_uvit_rebinned)
    
    for ind in range(0, num_uvit_rebinned):
        wavs_uvit_rebinned[ind]=(1.0/rebinfactor)*np.sum(HIP23309_FUV['lambda'].to_numpy()[(ind*rebinfactor):((ind+1)*rebinfactor)])
        flux_uvit_rebinned[ind]=(1.0/rebinfactor)*np.sum(HIP23309_FUV['flux'].to_numpy()[(ind*rebinfactor):((ind+1)*rebinfactor)])
        fluxerr_uvit_rebinned[ind]=(1.0/rebinfactor)*np.sqrt(np.sum(HIP23309_FUV['flux_err'].to_numpy()[(ind*rebinfactor):((ind+1)*rebinfactor)]**2.0))
    
   ###Calculate "chi-square". Approximate HST data as perfect (no error)
    chisquare=np.sum((hip23309_hst_smoothed_uvitbinned-HIP23309_FUV['flux'].to_numpy()[inds])**2.0/(HIP23309_FUV['flux_err'].to_numpy()[inds])**2.0)
    dof=len(hip23309_hst_smoothed_uvitbinned)
    rchisquare=chisquare/dof
    
    print('"Chi-square": {0:3.1f}, dof:{1:3f}"reduced "chi square"": {2:1.2f}'.format(chisquare, dof, rchisquare))
   
    ###Calculate integrated flux in HST, UVIT overlap (1300-1700 A)
    total_fuv_uvit=np.sum(uvit_wav_deltas_cut*HIP23309_FUV['flux'].to_numpy()[inds])
    total_fuv_uvit_err=np.sqrt(np.sum((uvit_wav_deltas_cut*HIP23309_FUV['flux_err'].to_numpy()[inds])**2.0))
    print('"UVIT 130-170 nm flux: {0:1.2e} \pm {1:1.2e}'.format(total_fuv_uvit, total_fuv_uvit_err))  

    
    inds2=np.where((hip_hst['wavelength'].to_numpy()>=fuv_min) & (hip_hst['wavelength'].to_numpy()<=fuv_max))
    total_fuv_hst=np.sum(0.6*hip_hst['flux'].to_numpy()[inds2])
    total_fuv_hst_err=np.sqrt(np.sum((0.6*hip_hst['err'].to_numpy()[inds2])**2.0))
    print('"HST 130-170 nm flux: {0:1.2e} \pm {1:1.2e}'.format(total_fuv_hst, total_fuv_hst_err))  

    print("(UVIT-HST)/error: {0:2.2f}".format((total_fuv_uvit-total_fuv_hst)/np.sqrt(total_fuv_uvit_err**2.0+total_fuv_hst_err**2.0)))  
    ###Plot
    markersize=3
    
    # pdb.set_trace()
    
    fig1, ax2=plt.subplots(1, figsize=(8,5), sharex=True)
    ax2.errorbar(hip_hst['wavelength'], hip_hst['flux']*4.0*np.pi*d_hip23309**2.0, yerr=hip_hst['err']*4.0*np.pi*d_hip23309**2.0, color='purple',marker='o', markersize=markersize, label='HST')
    ax2.errorbar(HIP23309_FUV['lambda'], HIP23309_FUV['flux']*4.0*np.pi*d_hip23309**2.0, yerr=HIP23309_FUV['flux_err']*4.0*np.pi*d_hip23309**2.0, color='black', marker='o', markersize=markersize, label='UVIT')
    ax2.errorbar(wavs_uvit_rebinned, flux_uvit_rebinned*4.0*np.pi*d_hip23309**2.0, yerr=fluxerr_uvit_rebinned*4.0*np.pi*d_hip23309**2.0, color='green', marker='o', markersize=markersize, label='UVIT (binned)')

    # ax2.errorbar(uvit_wav_centers_cut, hip23309_hst_rebinned*4.0*np.pi*d_hip23309**2.0, color='blue', marker='o', markersize=markersize, label='HST rebinned to UVIT')
    # ax2.errorbar(wl, hip23309_hst_rebinned*4.0*np.pi*d_hip23309**2.0, yerr=hip23309_hst_rebinned_err*4.0*np.pi*d_hip23309**2.0, color='blue', marker='o', markersize=markersize, label='HST rebinned to UVIT')
    # ax2.errorbar(hip_hst['wavelength'], hip23309_hst_rebinned*4.0*np.pi*d_hip23309**2.0, yerr=hip23309_hst_rebinned*0.0, color='blue', marker='o', markersize=markersize, label='HST rebinned to UVIT')
    ax2.errorbar(uvit_wav_centers_cut, hip23309_hst_smoothed_uvitbinned*4.0*np.pi*d_hip23309**2.0, yerr=hip23309_hst_smoothed_uvitbinned*0.0, color='blue', marker='o', markersize=markersize, label='HST rebinned to UVIT')

    ax2.legend(bbox_to_anchor=[-0.14, 1.03, 2., .152], loc=3, ncol=4, borderaxespad=0., fontsize=14)

    ax2.set_yscale('log')
    ax2.set_ylim([1.0e25, 1.0e28])
    ax2.set_xlim([1300.,1700.])
    ax2.set_xlabel('Wavelength (A)', fontsize=18)
    ax2.set_ylabel('Luminosity (erg/s/A)', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig('./Plots/hst_uvit_hip23309_comparison_FORTALK.pdf', orientation='portrait',format='pdf')

#######
###Compare Inter-orbit Stability: NUV
######

if nuv_inter_orbit_stability:
    fig1, ax2=plt.subplots(1, figsize=(8,5.5), sharex=True)
    
    markersizebyorbit=3
    markersizeintegrated=3
    

    
    ax2.errorbar(HIP23309_NUV1['lambda'], HIP23309_NUV1['flux']*4.0*np.pi*d_hip23309**2.0, yerr=HIP23309_NUV1['flux_err']*4.0*np.pi*d_hip23309**2.0, color='blue', marker='o', markersize=markersizebyorbit, label='Orbit 1')
    ax2.errorbar(HIP23309_NUV2['lambda'], HIP23309_NUV2['flux']*4.0*np.pi*d_hip23309**2.0, yerr=HIP23309_NUV2['flux_err']*4.0*np.pi*d_hip23309**2.0, color='orange', marker='o', markersize=markersizebyorbit, label='Orbit 2')
    ax2.errorbar(HIP23309_NUV3['lambda'], HIP23309_NUV3['flux']*4.0*np.pi*d_hip23309**2.0, yerr=HIP23309_NUV3['flux_err']*4.0*np.pi*d_hip23309**2.0, color='green', marker='o', markersize=markersizebyorbit, label='Orbit 3')
    ax2.errorbar(HIP23309_NUV4['lambda'], HIP23309_NUV4['flux']*4.0*np.pi*d_hip23309**2.0, yerr=HIP23309_NUV4['flux_err']*4.0*np.pi*d_hip23309**2.0, color='red', marker='o', markersize=markersizebyorbit, label='Orbit 4')
    ax2.errorbar(HIP23309_NUV5['lambda'], HIP23309_NUV5['flux']*4.0*np.pi*d_hip23309**2.0, yerr=HIP23309_NUV5['flux_err']*4.0*np.pi*d_hip23309**2.0, color='purple', marker='o', markersize=markersizebyorbit, label='Orbit 5')
    ax2.errorbar(HIP23309_NUV6['lambda'], HIP23309_NUV6['flux']*4.0*np.pi*d_hip23309**2.0, yerr=HIP23309_NUV6['flux_err']*4.0*np.pi*d_hip23309**2.0, color='brown', marker='o', markersize=markersizebyorbit, label='Orbit 6')
    ax2.errorbar(HIP23309_NUV7['lambda'], HIP23309_NUV7['flux']*4.0*np.pi*d_hip23309**2.0, yerr=HIP23309_NUV7['flux_err']*4.0*np.pi*d_hip23309**2.0, color='pink', marker='o', markersize=markersizebyorbit, label='Orbit 7')
    ax2.errorbar(HIP23309_NUV['lambda'], HIP23309_NUV['flux']*4.0*np.pi*d_hip23309**2.0, yerr=HIP23309_NUV['flux_err']*4.0*np.pi*d_hip23309**2.0, color='black', marker='o', markersize=markersizeintegrated, label='Combined Orbits 1-7')
    
    ax2.set_yscale('log')
    ax2.set_ylim([3.0e25, 3.0e27])
    ax2.legend(bbox_to_anchor=[-0.12, 1.02, 2., .152], loc=3, ncol=4, borderaxespad=0., fontsize=14)    
    ax2.set_xlim([2300.,2900.])
    ax2.set_xlabel('Wavelength (A)', fontsize=16)
    ax2.set_ylabel('Luminosity (erg/s/A)', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig('./Plots/uvit_nuv_orbit_by_orbit_FORTALK.pdf', orientation='portrait',format='pdf')

#######
###Compare against other M-dwarfs: FUV
######
markersize=3

if compare_hip23309_otherms_fuv:
    fig1, (ax1, ax2)=plt.subplots(2, figsize=(8,10), sharex=True)
    ax1.errorbar(hip_hst['wavelength'], hip_hst['flux']*4.0*np.pi*d_hip23309**2.0, yerr=hip_hst['err']*4.0*np.pi*d_hip23309**2.0, color='purple', label='HIP23309 (FUMES)',marker='o', markersize=markersize)
    ax1.errorbar(gj832['WAVELENGTH'], gj832['FLUX']*4.0*np.pi*d_gj832**2.0, yerr=gj832['ERROR']*4.0*np.pi*d_gj832**2.0, color='red', label='GJ832 (MUSCLES)',marker='o', markersize=markersize)
    ax1.errorbar(gj667c['WAVELENGTH'], gj667c['FLUX']*4.0*np.pi*d_gj667c**2.0, yerr=gj667c['ERROR']*4.0*np.pi*d_gj667c**2.0, color='orange', label='GJ667c (MUSCLES)',marker='o', markersize=markersize)
    ax1.errorbar(hd85512['WAVELENGTH'], hd85512['FLUX']*4.0*np.pi*d_hd85512**2.0, yerr=hd85512['ERROR']*4.0*np.pi*d_hd85512**2.0, color='blue', label='HD85512 (MUSCLES)',marker='o', markersize=markersize)
    ax1.errorbar(hd40307['WAVELENGTH'], hd40307['FLUX']*4.0*np.pi*d_hd40307**2.0, yerr=hd40307['ERROR']*4.0*np.pi*d_hd40307**2.0, color='magenta', label='HD40307 (MUSCLES)',marker='o', markersize=markersize)
    ax1.errorbar(HIP23309_FUV['lambda'], HIP23309_FUV['flux']*4.0*np.pi*d_hip23309**2.0, yerr=HIP23309_FUV['flux_err']*4.0*np.pi*d_hip23309**2.0, color='black', label='HIP23309 (UVIT)',marker='o', markersize=markersize)
    # ax1.errorbar(HIP23309_NUV['lambda'], HIP23309_NUV['flux']*4.0*np.pi*d_hip23309**2.0, yerr=HIP23309_NUV['flux_err']*4.0*np.pi*d_hip23309**2.0, color='black',marker='o', markersize=markersize)    
    
    ax1.set_yscale('linear')
    ax1.set_ylim([-1E27, 4.e27])
    ax1.legend(loc=0)
    ax1.set_xlabel('Wavelength (A)')
    ax1.set_ylabel('Luminosity (erg/s/A)')
    
    ax2.plot(hip_hst['wavelength'], hip_hst['flux']*4.0*np.pi*d_hip23309**2.0, color='purple', label='HIP23309 (FUMES)',marker='o', markersize=markersize)
    ax2.plot(gj832['WAVELENGTH'], gj832['FLUX']*4.0*np.pi*d_gj832**2.0, color='red', label='GJ832 (MUSCLES)',marker='o', markersize=markersize)
    ax2.plot(gj667c['WAVELENGTH'], gj667c['FLUX']*4.0*np.pi*d_gj667c**2.0, color='orange', label='GJ667c (MUSCLES)',marker='o', markersize=markersize)
    ax2.plot(hd85512['WAVELENGTH'], hd85512['FLUX']*4.0*np.pi*d_hd85512**2.0, color='blue', label='HD85512 (MUSCLES)',marker='o', markersize=markersize)
    ax2.plot(hd40307['WAVELENGTH'], hd40307['FLUX']*4.0*np.pi*d_hd40307**2.0, color='magenta', label='HD40307 (MUSCLES)',marker='o', markersize=markersize)
    ax2.plot(HIP23309_FUV['lambda'], HIP23309_FUV['flux']*4.0*np.pi*d_hip23309**2.0, color='black', label='HIP23309 (UVIT)',marker='o', markersize=markersize)
    # ax2.plot(HIP23309_NUV['lambda'], HIP23309_NUV['flux']*4.0*np.pi*d_hip23309**2.0, color='black',marker='o', markersize=markersize)
    ax2.set_yscale('log')
    ax2.set_ylim([1.e21, 1.e28])
    
    ax2.set_xlim([1200.,1800.])
    ax2.set_xlabel('Wavelength (A)')
    ax2.set_ylabel('Luminosity (erg/s/A)')
    plt.savefig('./Plots/compare_hip23309_otherstars_fuv.pdf', orientation='portrait',format='pdf')


#######
###Compare against other M-dwarfs: NUV
######
if compare_hip23309_otherms_nuv:
    fig1, (ax1, ax2)=plt.subplots(2, figsize=(8,10), sharex=True)
    ax1.plot(pineda_hip23309_aumic_wav, pineda_hip23309_aumic_luminosity, color='pink', label='AU-MIC (Pineda)',marker='o', markersize=markersize)
    ax1.errorbar(gj832['WAVELENGTH'], gj832['FLUX']*4.0*np.pi*d_gj832**2.0, yerr=gj832['ERROR']*4.0*np.pi*d_gj832**2.0, color='red', label='GJ832 (MUSCLES)',marker='o', markersize=markersize)
    ax1.errorbar(gj667c['WAVELENGTH'], gj667c['FLUX']*4.0*np.pi*d_gj667c**2.0, yerr=gj667c['ERROR']*4.0*np.pi*d_gj667c**2.0, color='orange', label='GJ667c (MUSCLES)',marker='o', markersize=markersize)
    ax1.errorbar(hd85512['WAVELENGTH'], hd85512['FLUX']*4.0*np.pi*d_hd85512**2.0, yerr=hd85512['ERROR']*4.0*np.pi*d_hd85512**2.0, color='blue', label='HD85512 (MUSCLES)',marker='o', markersize=markersize)
    ax1.plot(hd40307['WAVELENGTH'], hd40307['FLUX']*4.0*np.pi*d_hd40307**2.0, color='magenta', label='HD40307 (MUSCLES)',marker='o', markersize=markersize)
    # ax1.errorbar(HIP23309_FUV['lambda'], HIP23309_FUV['flux']*4.0*np.pi*d_hip23309**2.0, yerr=HIP23309_FUV['flux_err']*4.0*np.pi*d_hip23309**2.0, color='black', label='HIP23309 (UVIT)',marker='o', markersize=markersize)
    ax1.errorbar(HIP23309_NUV['lambda'], HIP23309_NUV['flux']*4.0*np.pi*d_hip23309**2.0, yerr=HIP23309_NUV['flux_err']*4.0*np.pi*d_hip23309**2.0, color='black',marker='o', markersize=markersize)   
    
    ax1.set_yscale('linear')
    ax1.set_ylim([-1E27, 3E27])
    ax1.legend(loc=0)
    ax1.set_xlabel('Wavelength (A)')
    ax1.set_ylabel('Luminosity (erg/s/A)')
    
    ax2.plot(pineda_hip23309_aumic_wav, pineda_hip23309_aumic_luminosity, color='pink', label='AU-MIC (Pineda)',marker='o', markersize=markersize)
    ax2.plot(gj832['WAVELENGTH'], gj832['FLUX']*4.0*np.pi*d_gj832**2.0, color='red', label='GJ832 (MUSCLES)',marker='o', markersize=markersize)
    ax2.plot(gj667c['WAVELENGTH'], gj667c['FLUX']*4.0*np.pi*d_gj667c**2.0, color='orange', label='GJ667c (MUSCLES)',marker='o', markersize=markersize)
    ax2.plot(hd85512['WAVELENGTH'], hd85512['FLUX']*4.0*np.pi*d_hd85512**2.0, color='blue', label='HD85512 (MUSCLES)',marker='o', markersize=markersize)
    ax2.plot(hd40307['WAVELENGTH'], hd40307['FLUX']*4.0*np.pi*d_hd40307**2.0, color='magenta', label='HD40307 (MUSCLES)',marker='o', markersize=markersize)
    # ax2.plot(HIP23309_FUV['lambda'], HIP23309_FUV['flux']*4.0*np.pi*d_hip23309**2.0, color='black', label='HIP23309 (UVIT)',marker='o', markersize=markersize)
    ax2.plot(HIP23309_NUV['lambda'], HIP23309_NUV['flux']*4.0*np.pi*d_hip23309**2.0, color='black',marker='o', markersize=markersize)    
    ax2.set_yscale('log')
    ax2.set_ylim([3.e21, 3.e27])
    
    ax2.set_xlim([2000.,3000.])
    ax2.set_xlabel('Wavelength (A)')
    ax2.set_ylabel('Luminosity (erg/s/A)')
    plt.savefig('./Plots/compare_hip23309_otherstars_nuv.pdf', orientation='portrait',format='pdf')

#######
###Compare against other M-dwarfs: total
######
if compare_hip23309_otherms:
    fig1, ax2=plt.subplots(1, figsize=(9,6), sharex=True)
    # ax1.plot(pineda_hip23309_aumic_wav, pineda_hip23309_aumic_luminosity, color='pink', label='AU-MIC (Pineda)',marker='o', markersize=markersize)
    # ax1.errorbar(gj832['WAVELENGTH'], gj832['FLUX']*4.0*np.pi*d_gj832**2.0, yerr=gj832['ERROR']*4.0*np.pi*d_gj832**2.0, color='red', label='GJ832 (MUSCLES)',marker='o', markersize=markersize)
    # ax1.errorbar(gj667c['WAVELENGTH'], gj667c['FLUX']*4.0*np.pi*d_gj667c**2.0, yerr=gj667c['ERROR']*4.0*np.pi*d_gj667c**2.0, color='orange', label='GJ667c (MUSCLES)',marker='o', markersize=markersize)
    # ax1.errorbar(hd85512['WAVELENGTH'], hd85512['FLUX']*4.0*np.pi*d_hd85512**2.0, yerr=hd85512['ERROR']*4.0*np.pi*d_hd85512**2.0, color='blue', label='HD85512 (MUSCLES)',marker='o', markersize=markersize)
    # ax1.plot(hd40307['WAVELENGTH'], hd40307['FLUX']*4.0*np.pi*d_hd40307**2.0, color='magenta', label='HD40307 (MUSCLES)',marker='o', markersize=markersize)
    # ax1.errorbar(HIP23309_FUV['lambda'], HIP23309_FUV['flux']*4.0*np.pi*d_hip23309**2.0, yerr=HIP23309_FUV['flux_err']*4.0*np.pi*d_hip23309**2.0, color='black', label='HIP23309 (UVIT)',marker='o', markersize=markersize)
    # ax1.errorbar(HIP23309_NUV['lambda'], HIP23309_NUV['flux']*4.0*np.pi*d_hip23309**2.0, yerr=HIP23309_NUV['flux_err']*4.0*np.pi*d_hip23309**2.0, color='black',marker='o', markersize=markersize)   
    
    # ax1.set_yscale('linear')
    # ax1.set_ylim([-1E27, 1E28])
    # ax1.set_xlabel('Wavelength (A)')
    # ax1.set_ylabel('Luminosity (erg/s/A)')
    
    # ax2.plot(pineda_hip23309_aumic_wav, pineda_hip23309_aumic_luminosity, color='pink', label='AU-MIC (Pineda)',marker='o', markersize=markersize)
    ax2.plot(gj832['WAVELENGTH'], gj832['FLUX']*4.0*np.pi*d_gj832**2.0, color='red', label='GJ832 (MUSCLES)',marker='o', markersize=markersize)
    ax2.plot(gj667c['WAVELENGTH'], gj667c['FLUX']*4.0*np.pi*d_gj667c**2.0, color='orange', label='GJ667c (MUSCLES)',marker='o', markersize=markersize)
    ax2.plot(hd85512['WAVELENGTH'], hd85512['FLUX']*4.0*np.pi*d_hd85512**2.0, color='blue', label='HD85512 (MUSCLES)',marker='o', markersize=markersize)
    ax2.plot(hd40307['WAVELENGTH'], hd40307['FLUX']*4.0*np.pi*d_hd40307**2.0, color='magenta', label='HD40307 (MUSCLES)',marker='o', markersize=markersize)
    ax2.plot(HIP23309_FUV['lambda'], HIP23309_FUV['flux']*4.0*np.pi*d_hip23309**2.0, color='black', label='HIP23309 (UVIT)',marker='o', markersize=markersize)
    ax2.plot(HIP23309_NUV['lambda'], HIP23309_NUV['flux']*4.0*np.pi*d_hip23309**2.0, color='black',marker='o', markersize=markersize)    
    ax2.set_yscale('log')
    ax2.set_ylim([1.e21, 1.e28])
    # ax2.legend(loc=0)
    ax2.legend(bbox_to_anchor=[-0.14, 1.03, 2., .152], loc=3, ncol=3, borderaxespad=0., fontsize=14)
   
    ax2.set_xlim([1200.,3000.])
    ax2.set_xlabel('Wavelength (A)', fontsize=16)
    ax2.set_ylabel('Luminosity (erg/s/A)', fontsize=16)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.savefig('./Plots/compare_hip23309_otherstars_FORTALK.pdf', orientation='portrait',format='pdf')
