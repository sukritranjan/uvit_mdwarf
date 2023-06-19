#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 16:42:13 2022

@author: sukrit
"""

########################
###Control switches for plotting
########################
compare_hst_uvit_hip23309_data=True #Compare FUV UVIT measurements of HIP 23309 to FUV HST measurements of HIP23309 #Needs to be true ALWAYS to run other bits...
nuv_inter_orbit_stability=False #Plot stability of UVT NUV measurement orbit-by-orbit?

compare_hip23309_otherms_intrinsicflux_fuv=True #compare FUV UVIT measurements of HIP 23309 to measurements of other stars in intrinsic flux
compare_hip23309_otherms_intrinsicflux_nuv=True #compare NUV UVIT measurements of HIP 23309 to measurements of other stars in intrinsic flux

make_hip23309_spectrum=False #Make synthetic spectrum of HIP23309 for use in theoretical models.

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
nm2cm=1.0E-7 #1 nm in cm
A2cm=1.0E-8 #1 A in cm
A2nm=1.0E-1 #1 A in nm
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
d_hip23309=26.8*pc #Distance to HIP 23309 in cm; Malo et al. 2014
d_gj832=5.0*pc #Distance to gj832 in cm; Youngblood et al. 2016 Table 2
d_gj667c=6.8*pc #Distance to gj667c in cm; Youngblood et al. 2016 Table 2
d_hd85512=11.2*pc #Distance to hd85512 in cm; Youngblood et al. 2016 Table 2
d_hd40307=13.0*pc #Distance to hd85512 in cm; Youngblood et al. 2016 Table 2
d_aumic=9.79*pc #Distance to AU Mic in cm; Plavchan+2020 Table 1
d_adleo=4.9*pc #Distance to AD Leo in cm; Segura+2005.

r_hip23309=0.93*R_sun #Radius of HIP 23309 in cm; Malo et al. 2014
r_gj832=0.46*R_sun #Radius of  gj832 in cm; Youngblood et al. 2016 Table 2
r_gj667c=0.46*R_sun #Radius of  gj667c in cm; Youngblood et al. 2016 Table 2
r_hd85512=0.7*R_sun #Radius of  hd85512 in cm; Youngblood et al. 2016 Table 2
r_hd40307=0.83*R_sun #Radius of  hd85512 in cm; Youngblood et al. 2016 Table 2
r_aumic=0.75*R_sun #Radius of AU Mic in cm; Plavchan+2020 Table 1
r_adleo=0.42*R_sun #Radius of AD Leo in cm; Kossakowski et al. 2022 Table 1

T_eff_hip23309=3886.0 #Pineda et al. 2021

#######
###MUSCLES Data
######
gj832 = fits.getdata('./LIT_DATA/hlsp_muscles_multi_multi_gj832_broadband_v22_adapt-const-res-sed.fits',1) #accesss via keywords WAVELENGTH (A), FLUX (erg/s/cm2/A) and ERROR (erg/s/cm2/A) ###M1.5V star
gj667c = fits.getdata('./LIT_DATA/hlsp_muscles_multi_multi_gj667c_broadband_v22_adapt-const-res-sed.fits',1) #accesss via keywords WAVELENGTH (A), FLUX (erg/s/cm2/A) and ERROR (erg/s/cm2/A) ###M1.5V star
hd85512 = fits.getdata('./LIT_DATA/hlsp_muscles_multi_multi_hd85512_broadband_v22_adapt-const-res-sed.fits',1) #accesss via keywords WAVELENGTH (A), FLUX (erg/s/cm2/A) and ERROR (erg/s/cm2/A) ###K6 star
hd40307 = fits.getdata('./LIT_DATA/hlsp_muscles_multi_multi_hd40307_broadband_v22_adapt-const-res-sed.fits',1) #accesss via keywords WAVELENGTH (A), FLUX (erg/s/cm2/A) and ERROR (erg/s/cm2/A) ###K2.5V star



#######
###AU Mic
######

aumic = fits.getdata('./LIT_DATA/h_hd197481_e140m-1425_020x020_51062_spc.fits',1) #accesss via keywords WAVE (A), FLUX (erg/s/cm2/A) and ERROR (erg/s/cm2/A) ###M1.5V star

aumic_wav_lowres=np.arange(1200, 1701, step=1)
aumic_flux_lowres, aumic_fluxerr_lowres=cg.downbin_spec_err(aumic['FLUX'][0], aumic['ERROR'][0], aumic['WAVE'][0], aumic_wav_lowres, dlam=np.ones(np.shape(aumic_wav_lowres)))


#######
###AD Leo
######

adleo_wav_um, adleo_flux_units=np.genfromtxt('./LIT_DATA/adleo_dat.txt', skip_header=175, skip_footer=1,usecols=(0,1), unpack=True) #um, Watt/cm2/um; fluxes are at Earth-star distance
adleo_wav=adleo_wav_um*1.0E4 #convert um to A
adleo_flux=adleo_flux_units *1.0E3 #convert Watt/cm2/um to erg/s/cm2/A

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
    fuv_min=1300. #UVIT FUV not trustworthy below ~1300 A (Prasanta email 2/18/2022)
    fuv_max=((hip_hst['wavelength'].to_numpy())[-1]) #Terminate where the HST data runs out. 
    inds=np.where((uvit_wav_left>=fuv_min) & (uvit_wav_right<=fuv_max))
    
    uvit_wav_left_cut=uvit_wav_left[inds]
    uvit_wav_right_cut=uvit_wav_right[inds]
    uvit_wav_centers_cut=uvit_wav_centers[inds]
    uvit_wav_deltas_cut=uvit_wav_right_cut-uvit_wav_left_cut
    
    hip23309_hst_smoothed=gaussian_filter1d(hip_hst['flux'], 14.63/0.6/2.385 )#FWHM of 14.63, /2.385 to get sigma, each STIS wavelength resolution element is 0.6

    hip23309_hst_smoothed_uvitbinned = cg.downbin_spec(hip23309_hst_smoothed,hip_hst['wavelength'], uvit_wav_centers_cut, dlam=uvit_wav_deltas_cut) #For some reason error-weighting doesn't work
    
    ###Rebin UVIT data to eliminate negative fluxes
    rebinfactor=5
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
    print('UVIT {0:1f}-{1:1f} nm flux: {2:1.2e} \pm {3:1.2e}'.format(fuv_min/10, fuv_max/10, total_fuv_uvit, total_fuv_uvit_err))  

    
    inds2=np.where((hip_hst['wavelength'].to_numpy()>=fuv_min) & (hip_hst['wavelength'].to_numpy()<=fuv_max))
    total_fuv_hst=np.sum(0.6*hip_hst['flux'].to_numpy()[inds2])
    total_fuv_hst_err=np.sqrt(np.sum((0.6*hip_hst['err'].to_numpy()[inds2])**2.0))
    print('"HST {0:1f}-{1:1f} nm flux: {2:1.2e} \pm {3:1.2e}'.format(fuv_min/10, fuv_max/10, total_fuv_hst, total_fuv_hst_err))  

    print("(UVIT-HST)/error: {0:2.2f}".format((total_fuv_uvit-total_fuv_hst)/np.sqrt(total_fuv_uvit_err**2.0+total_fuv_hst_err**2.0)))  
    
    ###Plot
    markersize=3
            
    fig1, (ax1,ax2)=plt.subplots(2, figsize=(8,8), sharex=True)
    ax1.errorbar(hip_hst['wavelength'], hip_hst['flux']*1E14, yerr=hip_hst['err']*1E14, color='palevioletred', marker='o', markersize=markersize, label='HST', alpha=0.7)
    ax1.errorbar(HIP23309_FUV['lambda'], HIP23309_FUV['flux']*1E14, yerr=HIP23309_FUV['flux_err']*1E14, color='black', marker='o', markersize=markersize, label='UVIT')
    ax1.errorbar(wavs_uvit_rebinned, flux_uvit_rebinned*1E14, yerr=fluxerr_uvit_rebinned*1E14, color='green', marker='o', markersize=markersize, zorder=10, label='UVIT (binned 5x)')
    ax1.plot(uvit_wav_centers_cut, hip23309_hst_smoothed_uvitbinned*1E14, color='darkorange', marker='o', markersize=markersize, label='HST degraded to UVIT')
    ax1.set_yscale('linear')
    # ax1.set_ylim([-1E27, 4.0E27])
    ax1.set_ylim([-2E-1, 2.0])
    # ax1.legend(bbox_to_anchor=[-0.02, 1.08, 2., .152], loc=3, ncol=4, borderaxespad=0., fontsize=15)
    ax1.legend(loc='best', ncol=2, borderaxespad=0., fontsize=15)
    ax1.set_ylabel(r'Flux (10$^{-14}$ erg s$^{-1}$ cm$^{-2}$ A$^{-1}$)', fontsize=15)
    ax1.set_xlim([1300.,1700.])
    ax1.set_xlabel(r'Wavelength ($\AA$)', fontsize=15)
    ax1.yaxis.set_tick_params(labelsize=15)
    ax1.xaxis.set_tick_params(labelsize=15)
    
    
    ax2.plot(hip_hst['wavelength'], hip_hst['flux'], color='palevioletred', marker='o', markersize=markersize, label='HST', alpha=0.7)
    ax2.plot(HIP23309_FUV['lambda'], HIP23309_FUV['flux'], color='black', marker='o', markersize=markersize, label='UVIT')
    ax2.plot(wavs_uvit_rebinned, flux_uvit_rebinned, color='green', marker='o', markersize=markersize, zorder=10, label='UVIT (binned 5x)')
    ax2.plot(uvit_wav_centers_cut, hip23309_hst_smoothed_uvitbinned, color='darkorange', marker='o', markersize=markersize, label='HST degraded to UVIT')
    ax2.set_yscale('log')
    # ax1.set_ylim([-1E27, 4.0E27])
    ax2.set_ylim([1.0e-16, 1.0e-13])
    # ax1.legend(bbox_to_anchor=[-0.02, 1.08, 2., .152], loc=3, ncol=4, borderaxespad=0., fontsize=15)
    # ax2.legend(loc='best', ncol=2, borderaxespad=0., fontsize=15)
    ax2.set_ylabel(r'Flux (erg s$^{-1}$ cm$^{-2}$ A$^{-1}$)', fontsize=15)
    ax2.set_xlim([1300.,1700.])
    ax2.set_xlabel(r'Wavelength ($\AA$)', fontsize=15)
    ax2.yaxis.set_tick_params(labelsize=15)
    ax2.xaxis.set_tick_params(labelsize=15)
       
    
    plt.savefig('./Plots/hst_uvit_hip23309_comparison.pdf', orientation='portrait',format='pdf')

#######
###Compare Inter-orbit Stability: NUV
######

if nuv_inter_orbit_stability:
    
    ###Calculate flux in NUV orbit-by-orbit
    #Form abscissa files
    nuv_uvit_wav_centers=HIP23309_NUV1['lambda'].to_numpy() #centers of all UVIT data.
    nuv_uvit_wav_left=np.zeros(np.shape(nuv_uvit_wav_centers))
    nuv_uvit_wav_right=np.zeros(np.shape(nuv_uvit_wav_centers))
    
    #Get abscissa
    nuv_uvit_wav_left[1:]=0.5*(nuv_uvit_wav_centers[0:-1] + nuv_uvit_wav_centers[1:])
    nuv_uvit_wav_right[:-1]=0.5*(nuv_uvit_wav_centers[0:-1] + nuv_uvit_wav_centers[1:])
    nuv_uvit_wav_left[0]=nuv_uvit_wav_centers[0]-(nuv_uvit_wav_right[0]-nuv_uvit_wav_centers[0])
    nuv_uvit_wav_right[-1]=nuv_uvit_wav_centers[-1]+(nuv_uvit_wav_centers[-1]-nuv_uvit_wav_left[-1])
    
    nuv_uvit_wav_deltas=nuv_uvit_wav_right-nuv_uvit_wav_left
    
    #Downselect
    nuv_min=2200. # What is citation for this being the reliable range?
    nuv_max=2900. # What is citation for this being the reliable range?
    inds=np.where((nuv_uvit_wav_left>=nuv_min) & (nuv_uvit_wav_right<=nuv_max))
    nuv_data_list=[HIP23309_NUV, HIP23309_NUV1, HIP23309_NUV2, HIP23309_NUV3, HIP23309_NUV4, HIP23309_NUV5, HIP23309_NUV6, HIP23309_NUV7]
    total_nuv_uvit=np.zeros(len(nuv_data_list))
    total_nuv_uvit_err=np.zeros(len(nuv_data_list))
    for ind in range(0, len(nuv_data_list)):
        nuv_data=nuv_data_list[ind]
        total_nuv_uvit[ind]=np.sum(nuv_uvit_wav_deltas[inds]*nuv_data['flux'].to_numpy()[inds])
        total_nuv_uvit_err[ind]=np.sqrt(np.sum((nuv_uvit_wav_deltas[inds]*nuv_data['flux_err'].to_numpy()[inds])**2.0))
        print('UVIT 220-290 nm flux, {0:1.1f}: {1:1.2e} \pm {2:1.2e}'.format(ind, total_nuv_uvit[ind], total_nuv_uvit_err[ind])) 
    
    ###Plot flux variation
    fig0, ax1=plt.subplots(1, figsize=(8,6))
    orbitnums=np.array([1,2,3,4,5,6,7])
    ax1.errorbar(orbitnums, total_nuv_uvit[1:]*1E12, yerr=total_nuv_uvit_err[1:]*1E12, linestyle='none', markersize=5, color='red', marker='o', capsize=20) #plot orbit-by-orbit.
    ax1.errorbar(np.mean(orbitnums), total_nuv_uvit[0]*1E12, yerr=total_nuv_uvit_err[0]*1E12, linestyle='none', markersize=10, color='black',  marker='s',capsize=220) #plot orbit-by-orbit.
    ax1.set_xlabel('Orbit', fontsize=20)
    ax1.set_ylabel(r'NUV Flux (10$^{-12}$ erg s$^{-1}$ cm$^{-2}$)', fontsize=20)
    ax1.yaxis.set_tick_params(labelsize=16)
    ax1.xaxis.set_tick_params(labelsize=16)
    plt.savefig('./Plots/uvit_nuv_orbit_by_orbit_band.pdf', orientation='portrait',format='pdf')
    
    print((total_nuv_uvit[-1]-total_nuv_uvit[0])/total_nuv_uvit[-1]) #max degree of variation from median.
    print((total_nuv_uvit[-1]-total_nuv_uvit[0])/np.sqrt(total_nuv_uvit_err[-1]**2.0 + total_nuv_uvit_err[0]**2.0)) #max degree of variation from median.
    
    ###Calculate "reduced chi-squares" to see how well data agree.
    
    
    for ind in range(1, len(nuv_data_list)):
        nuv_data=nuv_data_list[ind]
        ###Calculate "chi-square". Approximate HST data as perfect (no error)
        chisquare=np.sum((nuv_data['flux'].to_numpy()[inds]-HIP23309_NUV['flux'].to_numpy()[inds])**2.0/np.sqrt(nuv_data['flux_err'].to_numpy()[inds]**2.0+HIP23309_NUV['flux_err'].to_numpy()[inds]**2.0)**2.0)
        dof=len(np.squeeze(inds))
        rchisquare=chisquare/dof
         
        print('"Chi-square": {0:3.1f}, dof:{1:3f}"reduced "chi square"": {2:1.2f}'.format(chisquare, dof, rchisquare))
    
    ###Plot spectral variation in orbit-by-orbit flux
    # fig2, (ax1, ax2)=plt.subplots(2, figsize=(8,10), sharex=True)
    
    # markersizebyorbit=3
    # markersizeintegrated=3
    
    # ax1.errorbar(HIP23309_NUV1['lambda'], HIP23309_NUV1['flux']*1E14, yerr=HIP23309_NUV1['flux_err']*1E14, color='blue', marker='o', markersize=markersizebyorbit, label='Orbit 1')
    # ax1.errorbar(HIP23309_NUV2['lambda'], HIP23309_NUV2['flux']*1E14, yerr=HIP23309_NUV2['flux_err']*1E14, color='orange', marker='o', markersize=markersizebyorbit, label='Orbit 2')
    # ax1.errorbar(HIP23309_NUV3['lambda'], HIP23309_NUV3['flux']*1E14, yerr=HIP23309_NUV3['flux_err']*1E14, color='green', marker='o', markersize=markersizebyorbit, label='Orbit 3')
    # ax1.errorbar(HIP23309_NUV4['lambda'], HIP23309_NUV4['flux']*1E14, yerr=HIP23309_NUV4['flux_err']*1E14, color='red', marker='o', markersize=markersizebyorbit, label='Orbit 4')
    # ax1.errorbar(HIP23309_NUV5['lambda'], HIP23309_NUV5['flux']*1E14, yerr=HIP23309_NUV5['flux_err']*1E14, color='purple', marker='o', markersize=markersizebyorbit, label='Orbit 5')
    # ax1.errorbar(HIP23309_NUV6['lambda'], HIP23309_NUV6['flux']*1E14, yerr=HIP23309_NUV6['flux_err']*1E14, color='brown', marker='o', markersize=markersizebyorbit, label='Orbit 6')
    # ax1.errorbar(HIP23309_NUV7['lambda'], HIP23309_NUV7['flux']*1E14, yerr=HIP23309_NUV7['flux_err']*1E14, color='pink', marker='o', markersize=markersizebyorbit, label='Orbit 7')
    # ax1.errorbar(HIP23309_NUV['lambda'], HIP23309_NUV['flux']*1E14, yerr=HIP23309_NUV['flux_err']*1E14, color='black', marker='o', markersize=markersizeintegrated, label='Combined')    
    
    # ax1.set_yscale('linear')
    # ax1.set_ylim([0, 2.0])
    # ax1.legend(bbox_to_anchor=[-0.07, 1.07, 2., .152], loc=3, ncol=4, borderaxespad=0., fontsize=15)
    # # ax1.set_xlabel('Wavelength (A)')
    # ax1.set_ylabel(r'Flux (10$^{-14}$ erg s$^{-1}$ cm$^{-2}$ A$^{-1}$)', fontsize=16)
    
    # ax2.errorbar(HIP23309_NUV1['lambda'], HIP23309_NUV1['flux'], yerr=HIP23309_NUV1['flux_err'], color='blue', marker='o', markersize=markersizebyorbit, label='Orbit 1')
    # ax2.errorbar(HIP23309_NUV2['lambda'], HIP23309_NUV2['flux'], yerr=HIP23309_NUV2['flux_err'], color='orange', marker='o', markersize=markersizebyorbit, label='Orbit 2')
    # ax2.errorbar(HIP23309_NUV3['lambda'], HIP23309_NUV3['flux'], yerr=HIP23309_NUV3['flux_err'], color='green', marker='o', markersize=markersizebyorbit, label='Orbit 3')
    # ax2.errorbar(HIP23309_NUV4['lambda'], HIP23309_NUV4['flux'], yerr=HIP23309_NUV4['flux_err'], color='red', marker='o', markersize=markersizebyorbit, label='Orbit 4')
    # ax2.errorbar(HIP23309_NUV5['lambda'], HIP23309_NUV5['flux'], yerr=HIP23309_NUV5['flux_err'], color='purple', marker='o', markersize=markersizebyorbit, label='Orbit 5')
    # ax2.errorbar(HIP23309_NUV6['lambda'], HIP23309_NUV6['flux'], yerr=HIP23309_NUV6['flux_err'], color='brown', marker='o', markersize=markersizebyorbit, label='Orbit 6')
    # ax2.errorbar(HIP23309_NUV7['lambda'], HIP23309_NUV7['flux'], yerr=HIP23309_NUV7['flux_err'], color='pink', marker='o', markersize=markersizebyorbit, label='Orbit 7')
    # ax2.errorbar(HIP23309_NUV['lambda'], HIP23309_NUV['flux'], yerr=HIP23309_NUV['flux_err'], color='black', marker='o', markersize=markersizeintegrated, label='Combined')
    
    # ax2.set_yscale('log')
    # ax2.set_ylim([2.0e-16, 2.0e-14])
    
    # ax2.set_xlim([2200.,2900.])
    # ax2.set_xlabel('Wavelength (A)', fontsize=16)
    # ax2.set_ylabel(r'Flux (erg s$^{-1}$ cm$^{-2}$ A$^{-1}$)', fontsize=16)
    # fig2.subplots_adjust(wspace=0.01)
    # ax1.yaxis.set_tick_params(labelsize=15)
    # ax2.yaxis.set_tick_params(labelsize=15)
    # ax2.xaxis.set_tick_params(labelsize=15)
    # plt.savefig('./Plots/uvit_nuv_orbit_by_orbit.pdf', orientation='portrait',format='pdf')
    
    fig2, ax1=plt.subplots(1, figsize=(8,6.0), sharex=True)
    
    markersizebyorbit=3
    markersizeintegrated=3
    
    ax1.errorbar(HIP23309_NUV1['lambda'], HIP23309_NUV1['flux']*1E14, yerr=HIP23309_NUV1['flux_err']*1E14, color='blue', marker='o', markersize=markersizebyorbit, label='Orbit 1')
    ax1.errorbar(HIP23309_NUV2['lambda'], HIP23309_NUV2['flux']*1E14, yerr=HIP23309_NUV2['flux_err']*1E14, color='orange', marker='o', markersize=markersizebyorbit, label='Orbit 2')
    ax1.errorbar(HIP23309_NUV3['lambda'], HIP23309_NUV3['flux']*1E14, yerr=HIP23309_NUV3['flux_err']*1E14, color='green', marker='o', markersize=markersizebyorbit, label='Orbit 3')
    ax1.errorbar(HIP23309_NUV4['lambda'], HIP23309_NUV4['flux']*1E14, yerr=HIP23309_NUV4['flux_err']*1E14, color='red', marker='o', markersize=markersizebyorbit, label='Orbit 4')
    ax1.errorbar(HIP23309_NUV5['lambda'], HIP23309_NUV5['flux']*1E14, yerr=HIP23309_NUV5['flux_err']*1E14, color='purple', marker='o', markersize=markersizebyorbit, label='Orbit 5')
    ax1.errorbar(HIP23309_NUV6['lambda'], HIP23309_NUV6['flux']*1E14, yerr=HIP23309_NUV6['flux_err']*1E14, color='brown', marker='o', markersize=markersizebyorbit, label='Orbit 6')
    ax1.errorbar(HIP23309_NUV7['lambda'], HIP23309_NUV7['flux']*1E14, yerr=HIP23309_NUV7['flux_err']*1E14, color='pink', marker='o', markersize=markersizebyorbit, label='Orbit 7')
    ax1.errorbar(HIP23309_NUV['lambda'], HIP23309_NUV['flux']*1E14, yerr=HIP23309_NUV['flux_err']*1E14, color='black', marker='o', markersize=markersizeintegrated, label='Combined')    
    
    ax1.set_yscale('linear')
    ax1.set_ylim([0, 2.0])
    # ax1.legend(bbox_to_anchor=[-0.07, 1.02, 2., .152], loc=3, ncol=4, borderaxespad=0., fontsize=15)
    ax1.legend(loc='best', ncol=2, borderaxespad=0., fontsize=15)    
    ax1.set_xlim([2200.,2900.])
    ax1.set_xlabel('Wavelength ($\AA$)', fontsize=20)
    ax1.set_ylabel(r'Flux (10$^{-14}$ erg s$^{-1}$ cm$^{-2}$ A$^{-1}$)', fontsize=20)
    ax1.yaxis.set_tick_params(labelsize=15)
    ax1.xaxis.set_tick_params(labelsize=15)
    plt.savefig('./Plots/uvit_nuv_orbit_by_orbit.pdf', orientation='portrait',format='pdf')



#######################################
###Intrinsic Flux
#######################################    
#Factor to multiply to convert flux at detector to desired quantity
convert_detected_flux_hip23309=4.0*np.pi*d_hip23309**2.0/(4.0*np.pi*r_hip23309**2.0)
convert_detected_flux_gj832=4.0*np.pi*d_gj832**2.0/(4.0*np.pi*r_gj832**2.0)
convert_detected_flux_gj667c=4.0*np.pi*d_gj667c**2.0/(4.0*np.pi*r_gj667c**2.0)
convert_detected_flux_hd85512=4.0*np.pi*d_hd85512**2.0/(4.0*np.pi*r_hd85512**2.0)
convert_detected_flux_hd40307=4.0*np.pi*d_hd40307**2.0/(4.0*np.pi*r_hd40307**2.0)
convert_detected_flux_aumic=4.0*np.pi*d_aumic**2.0/(4.0*np.pi*r_aumic**2.0)
convert_detected_flux_adleo=4.0*np.pi*d_adleo**2.0/(4.0*np.pi*r_adleo**2.0)


#######
###Compare against other M-dwarfs: FUV
######


markersize=3

if compare_hip23309_otherms_intrinsicflux_fuv:
    # fig1, (ax1, ax2)=plt.subplots(2, figsize=(8,10), sharex=True)
    # ax1.errorbar(gj832['WAVELENGTH'], gj832['FLUX']*convert_detected_flux_gj832, yerr=gj832['ERROR']*convert_detected_flux_gj832, color='red', label='GJ832 (MUSCLES)',marker='o', markersize=markersize)
    # ax1.errorbar(gj667c['WAVELENGTH'], gj667c['FLUX']*convert_detected_flux_gj667c, yerr=gj667c['ERROR']*convert_detected_flux_gj667c, color='orange', label='GJ667c (MUSCLES)',marker='o', markersize=markersize)
    # ax1.errorbar(hd85512['WAVELENGTH'], hd85512['FLUX']*convert_detected_flux_hd85512, yerr=hd85512['ERROR']*convert_detected_flux_hd85512, color='blue', label='HD85512 (MUSCLES)',marker='o', markersize=markersize)
    # ax1.errorbar(hd40307['WAVELENGTH'], hd40307['FLUX']*convert_detected_flux_hd40307, yerr=hd40307['ERROR']*convert_detected_flux_hd40307, color='magenta', label='HD40307 (MUSCLES)',marker='o', markersize=markersize)
    # ax1.errorbar(aumic_wav_lowres, aumic_flux_lowres*convert_detected_flux_aumic, yerr=aumic_fluxerr_lowres*convert_detected_flux_aumic, color='green', label='AU Mic',marker='o', markersize=markersize)
    # ax1.errorbar(hip_hst['wavelength'], hip_hst['flux']*convert_detected_flux_hip23309, yerr=hip_hst['err']*convert_detected_flux_hip23309, color='purple', label='HIP23309 (FUMES)',marker='o', markersize=markersize)
    # ax1.errorbar(HIP23309_FUV['lambda'], HIP23309_FUV['flux']*convert_detected_flux_hip23309, yerr=HIP23309_FUV['flux_err']*convert_detected_flux_hip23309, color='black', label='HIP23309 (UVIT)',marker='o', markersize=markersize)
    
    # ax1.set_yscale('linear')
    # ax1.set_ylim([-1E3, 1.e5])
    # ax1.legend(loc=0)
    # ax1.set_xlabel('Wavelength (A)')
    # ax1.set_ylabel('Intrinsic Flux (erg/s/A/cm2)')
    
    # ax2.plot(gj832['WAVELENGTH'], gj832['FLUX']*convert_detected_flux_gj832, color='red', label='GJ832 (MUSCLES)',marker='o', markersize=markersize)
    # ax2.plot(gj667c['WAVELENGTH'], gj667c['FLUX']*convert_detected_flux_gj667c, color='orange', label='GJ667c (MUSCLES)',marker='o', markersize=markersize)
    # ax2.plot(hd85512['WAVELENGTH'], hd85512['FLUX']*convert_detected_flux_hd85512, color='blue', label='HD85512 (MUSCLES)',marker='o', markersize=markersize)
    # ax2.plot(hd40307['WAVELENGTH'], hd40307['FLUX']*convert_detected_flux_hd40307, color='magenta', label='HD40307 (MUSCLES)',marker='o', markersize=markersize)
    # ax2.plot(aumic_wav_lowres, aumic_flux_lowres*convert_detected_flux_aumic, color='green', label='AU Mic',marker='o', markersize=markersize)
    # ax2.plot(hip_hst['wavelength'], hip_hst['flux']*convert_detected_flux_hip23309, color='purple', label='HIP23309 (FUMES)',marker='o', markersize=markersize)
    # ax2.plot(HIP23309_FUV['lambda'], HIP23309_FUV['flux']*convert_detected_flux_hip23309, color='black', label='HIP23309 (UVIT)',marker='o', markersize=markersize)
    # ax2.set_yscale('log')
    # ax2.set_ylim([1.e-1, 1.e6])
    
    # ax2.set_xlim([1200.,1800.])
    # ax2.set_xlabel('Wavelength (A)')
    # ax2.set_ylabel('Intrinsic Flux (erg/s/A/cm2)')
    # plt.savefig('./Plots/compare_hip23309_otherstars_intrinsicflux_fuv.pdf', orientation='portrait',format='pdf')
    fig1, ax1=plt.subplots(1, figsize=(10,7), sharex=True)
    ax1.plot(gj667c['WAVELENGTH'], gj667c['FLUX']*convert_detected_flux_gj667c, color='red', label='GJ667c',marker='o', markersize=markersize)
    ax1.plot(gj832['WAVELENGTH'], gj832['FLUX']*convert_detected_flux_gj832, color='orange', label='GJ832',marker='o', markersize=markersize)
    ax1.plot(hd85512['WAVELENGTH'], hd85512['FLUX']*convert_detected_flux_hd85512, color='blue', label='HD85512',marker='o', markersize=markersize)
    ax1.plot(hd40307['WAVELENGTH'], hd40307['FLUX']*convert_detected_flux_hd40307, color='magenta', label='HD40307',marker='o', markersize=markersize)
    ax1.plot(aumic_wav_lowres, aumic_flux_lowres*convert_detected_flux_aumic, color='green', label='AU Mic',marker='o', markersize=markersize)
    ax1.plot(adleo_wav, adleo_flux*convert_detected_flux_adleo, color='grey', label='AD Leo',marker='o', markersize=markersize)

    ax1.plot(hip_hst['wavelength'], hip_hst['flux']*convert_detected_flux_hip23309, color='purple', label='HIP 23309 (HST)',marker='o', markersize=markersize)
    # ax1.plot(HIP23309_FUV['lambda'], HIP23309_FUV['flux']*convert_detected_flux_hip23309, color='black', label='HIP 23309 (UVIT)',marker='o', markersize=markersize)
    ax1.errorbar(wavs_uvit_rebinned, flux_uvit_rebinned*convert_detected_flux_hip23309, yerr=fluxerr_uvit_rebinned, color='black', marker='o', markersize=markersize, zorder=10, label='HIP 23309 (UVIT, binned 5x)')

    

    
    ax1.set_yscale('log')
    ax1.set_ylim([1.e-1, 1.e6])
    ax1.legend(loc=0, ncol=3, fontsize=14)
    # ax1.legend(bbox_to_anchor=[-0.15, 1.03, 2., .152], loc=3, ncol=3, borderaxespad=0., fontsize=15)

    ax1.set_xlabel('Wavelength ($\AA$)', fontsize=20)
    ax1.set_ylabel(r'Flux at Stellar Surface (erg s$^{-1}$ $\AA^{-1}$ cm$^{-2}$)', fontsize=20)
    ax1.set_xlim([1290.,1710.])
    ax1.yaxis.set_tick_params(labelsize=15)
    ax1.xaxis.set_tick_params(labelsize=15)   
    plt.savefig('./Plots/compare_hip23309_otherstars_intrinsicflux_fuv.pdf', orientation='portrait',format='pdf')


#######
###Compare against other M-dwarfs: NUV
######
if compare_hip23309_otherms_intrinsicflux_nuv:
    # fig1, (ax1, ax2)=plt.subplots(2, figsize=(8,10), sharex=True)
    # ax1.errorbar(gj832['WAVELENGTH'], gj832['FLUX']*convert_detected_flux_gj832, yerr=gj832['ERROR']*convert_detected_flux_gj832, color='red', label='GJ832 (MUSCLES)',marker='o', markersize=markersize)
    # ax1.errorbar(gj667c['WAVELENGTH'], gj667c['FLUX']*convert_detected_flux_gj667c, yerr=gj667c['ERROR']*convert_detected_flux_gj667c, color='orange', label='GJ667c (MUSCLES)',marker='o', markersize=markersize)
    # ax1.errorbar(hd85512['WAVELENGTH'], hd85512['FLUX']*convert_detected_flux_hd85512, yerr=hd85512['ERROR']*convert_detected_flux_hd85512, color='blue', label='HD85512 (MUSCLES)',marker='o', markersize=markersize)
    # ax1.errorbar(hd40307['WAVELENGTH'], hd40307['FLUX']*convert_detected_flux_hd40307, yerr=hd40307['ERROR']*convert_detected_flux_hd40307, color='magenta', label='HD40307 (MUSCLES)',marker='o', markersize=markersize)
    # ax1.errorbar(HIP23309_NUV['lambda'], HIP23309_NUV['flux']*convert_detected_flux_hip23309, yerr=HIP23309_NUV['flux_err']*convert_detected_flux_hip23309, color='black',marker='o', markersize=markersize)   
    
    # ax1.set_yscale('linear')
    # ax1.set_ylim([-1E4, 3E5])
    # ax1.legend(loc=0)
    # ax1.set_xlabel('Wavelength (A)')
    # ax1.set_ylabel('Intrinsic Flux (erg/s/A/cm2)')
    
    # ax2.plot(gj832['WAVELENGTH'], gj832['FLUX']*convert_detected_flux_gj832, color='red', label='GJ832 (MUSCLES)',marker='o', markersize=markersize)
    # ax2.plot(gj667c['WAVELENGTH'], gj667c['FLUX']*convert_detected_flux_gj667c, color='orange', label='GJ667c (MUSCLES)',marker='o', markersize=markersize)
    # ax2.plot(hd85512['WAVELENGTH'], hd85512['FLUX']*convert_detected_flux_hd85512, color='blue', label='HD85512 (MUSCLES)',marker='o', markersize=markersize)
    # ax2.plot(hd40307['WAVELENGTH'], hd40307['FLUX']*convert_detected_flux_hd40307, color='magenta', label='HD40307 (MUSCLES)',marker='o', markersize=markersize)
    # ax2.plot(HIP23309_NUV['lambda'], HIP23309_NUV['flux']*convert_detected_flux_hip23309, color='black',marker='o', markersize=markersize)    
    # ax2.set_yscale('log')
    # ax2.set_ylim([1.e-1, 1.e6])
    
    # ax2.set_xlim([2000.,3000.])
    # ax2.set_xlabel('Wavelength (A)')
    # ax2.set_ylabel('Intrinsic Flux (erg/s/A/cm2)')
    # plt.savefig('./Plots/compare_hip23309_otherstars_intrinsicflux_nuv.pdf', orientation='portrait',format='pdf')
    
    fig1, ax1=plt.subplots(1, figsize=(10,7), sharex=True)
    ax1.plot(gj667c['WAVELENGTH'], gj667c['FLUX']*convert_detected_flux_gj667c, color='red', label='GJ667c',marker='o', markersize=markersize)
    ax1.plot(gj832['WAVELENGTH'], gj832['FLUX']*convert_detected_flux_gj832, color='orange', label='GJ832',marker='o', markersize=markersize)
    ax1.plot(hd85512['WAVELENGTH'], hd85512['FLUX']*convert_detected_flux_hd85512, color='blue', label='HD85512',marker='o', markersize=markersize)
    ax1.plot(hd40307['WAVELENGTH'], hd40307['FLUX']*convert_detected_flux_hd40307, color='magenta', label='HD40307',marker='o', markersize=markersize)
    ax1.plot(adleo_wav, adleo_flux*convert_detected_flux_adleo, color='grey', label='AD Leo',marker='o', markersize=markersize)
    ax1.plot(HIP23309_NUV['lambda'], HIP23309_NUV['flux']*convert_detected_flux_hip23309, color='black',marker='o', markersize=markersize, label='HIP 23309 (UVIT)')      
    
    ax1.set_yscale('log')
    ax1.set_ylim([1E-1, 1E6])
    ax1.legend(loc='upper left', ncol=2, fontsize=14)
    # ax1.legend(bbox_to_anchor=[-0.15, 1.03, 2., .152], loc=3, ncol=3, borderaxespad=0., fontsize=15)
    ax1.set_xlabel('Wavelength ($\AA$)', fontsize=20)
    ax1.set_ylabel(r'Flux at Stellar Surface (erg s$^{-1}$ $\AA^{-1}$ cm$^{-2}$)', fontsize=20)
    ax1.set_xlim([2000.,3000.])
    ax1.yaxis.set_tick_params(labelsize=15)
    ax1.xaxis.set_tick_params(labelsize=15)   
    plt.savefig('./Plots/compare_hip23309_otherstars_intrinsicflux_nuv.pdf', orientation='portrait',format='pdf')
    
    # def get_median_flux(wavelengths, fluxs):
    #     """
    #     """
    #     inds=np.where((wavelengths>=2450.0) & (wavelengths<=2550.0))
    #     median_flux=np.median(fluxs[inds])
    #     return median_flux
    
    # fig1, ax1=plt.subplots(1, figsize=(8,7), sharex=True)
    # ax1.plot(gj667c['WAVELENGTH'], (gj667c['FLUX']/get_median_flux(gj667c['WAVELENGTH'],gj667c['FLUX'])), color='red', label='GJ667c',marker='o', markersize=markersize)
    # ax1.plot(gj832['WAVELENGTH'], (gj832['FLUX']/get_median_flux(gj832['WAVELENGTH'],gj832['FLUX'] )), color='orange', label='GJ832',marker='o', markersize=markersize)
    # ax1.plot(hd85512['WAVELENGTH'], (hd85512['FLUX']/get_median_flux(hd85512['WAVELENGTH'],hd85512['FLUX'])), color='blue', label='HD85512',marker='o', markersize=markersize)
    # ax1.plot(hd40307['WAVELENGTH'], (hd40307['FLUX']/get_median_flux(hd40307['WAVELENGTH'],hd40307['FLUX'])), color='magenta', label='HD40307',marker='o', markersize=markersize)
    # ax1.plot(adleo_wav, (adleo_flux/get_median_flux(adleo_wav,adleo_flux)), color='grey', label='AD Leo',marker='o', markersize=markersize)

    # ax1.plot(HIP23309_NUV['lambda'].to_numpy(), (HIP23309_NUV['flux'].to_numpy()/get_median_flux(HIP23309_NUV['lambda'].to_numpy(), HIP23309_NUV['flux'].to_numpy())), color='black',marker='o', markersize=markersize, label='HIP 23309 (UVIT)')      
    
    # ax1.set_yscale('log')
    # ax1.set_ylim([1E-1, 1E3])
    # ax1.legend(loc='upper left', ncol=2, fontsize=14)
    # # ax1.legend(bbox_to_anchor=[-0.15, 1.03, 2., .152], loc=3, ncol=3, borderaxespad=0., fontsize=15)
    # ax1.set_xlabel('Wavelength ($\AA$)', fontsize=20)
    # ax1.set_ylabel(r'Normalized Flux', fontsize=20)
    # ax1.set_xlim([2000.,3000.])
    # ax1.yaxis.set_tick_params(labelsize=15)
    # ax1.xaxis.set_tick_params(labelsize=15)   
    # plt.savefig('./Plots/compare_hip23309_otherstars_shape_nuv.pdf', orientation='portrait',format='pdf')


if make_hip23309_spectrum:
    ####
    gaia_wav_A, gaia_flux_otherunits=np.genfromtxt('./LIT_DATA/HIP23309_corrected_Gaia_spectra.csv', skip_header=2, skip_footer=0, unpack=True, usecols=(0,1), delimiter=',') #units: A, erg/s/cm2/A
    gaia_wav=gaia_wav_A*0.1 #convert A to nm
    gaia_flux=gaia_flux_otherunits*0.01 #convert erg/cm2/A/s to W/m**-1/nm

    ####
    #Blackbody?
    nuv_max=np.ceil(np.max(HIP23309_NUV['lambda'])*A2nm) # max wavelength covered by the actual data, converted to nm. 
    model_wav=np.arange(100.0, 1001.0, step=1.0) # wavelengths to be filled in by model, in nm
    
    #Estimate flux assuming blackbody, drawing on eqn 2.43 of Catling & Kasting 2017
    model_wav_cm=model_wav*nm2cm #convert wavelengths from nm to cm for cgs calculation
    model_flux=(r_hip23309/d_hip23309)**2.0 * np.pi * (2.0*h*c**2.0/model_wav_cm**5.0)*(1.0/(np.exp(h*c/(model_wav_cm*k*T_eff_hip23309))-1))*A2cm #erg cm**-2 s**-1 cm**-1, converted to erg cm**-2 s**-1 A**-1

    
    ####Remove UVIT FUV <130 nm = 1300 A as unreliable.
    fuv_uvit_inds=np.where((wavs_uvit_rebinned >=1290.0) & (wavs_uvit_rebinned<=1710.0))
    nuv_uvit_inds=np.where((HIP23309_NUV['lambda']>=2000) & (HIP23309_NUV['lambda']<=2950))

    ####Make and save file for purely UVIT data.
    hip23309_spec_wav=np.concatenate((wavs_uvit_rebinned[fuv_uvit_inds]*A2nm, HIP23309_NUV['lambda'].to_numpy()[nuv_uvit_inds]*A2nm, gaia_wav))
    hip23309_spec_flux=np.concatenate((flux_uvit_rebinned[fuv_uvit_inds]*0.01, HIP23309_NUV['flux'].to_numpy()[nuv_uvit_inds]*0.01, gaia_flux))
    #Factor of 0.01 to convert erg s**-1 cm**-2 A**-1 to W m**-2 nm**-1
    savedata=np.column_stack((hip23309_spec_wav, hip23309_spec_flux))
    np.savetxt('../uv-prebiochem/Raw_Data/Mdwarf_Spectra/Steady-State/UVIT/HIP23309spec.txt', savedata, delimiter='\t', newline='\n', fmt='%3.6f %1.6e') #Print checkfile for reaction rates.

    ####Make and save file for UVIT+HST Data
    hip23309_hstextended_spec_wav=np.concatenate((hip_hst['wavelength'].to_numpy()*A2nm, HIP23309_NUV['lambda'].to_numpy()[nuv_uvit_inds]*A2nm, gaia_wav))
    hip23309_hstextended_spec_flux=np.concatenate((hip_hst['flux'].to_numpy()*0.01, HIP23309_NUV['flux'].to_numpy()[nuv_uvit_inds]*0.01, gaia_flux))
    #Factor of 0.01 to convert erg s**-1 cm**-2 A**-1 to W m**-2 nm**-1
    savedata2=np.column_stack((hip23309_hstextended_spec_wav, hip23309_hstextended_spec_flux))
    np.savetxt('../uv-prebiochem/Raw_Data/Mdwarf_Spectra/Steady-State/UVIT/HIP23309spec_hstextended.txt', savedata2, delimiter='\t', newline='\n', fmt='%3.6f %1.6e') #Print checkfile for reaction rates.
    
    ####Make and save file for UVIT data rebinned
    rebinned_wavs=np.arange(1300.0, 1740.0, step=40.0)
    uvit_bindownmore = cg.downbin_spec(flux_uvit_rebinned,wavs_uvit_rebinned, rebinned_wavs, dlam=np.ones(np.shape(rebinned_wavs))*40.0) #For some reason error-weighting doesn't work

    hip23309_rebinmore_spec_wav=np.concatenate((rebinned_wavs*A2nm, HIP23309_NUV['lambda'].to_numpy()[nuv_uvit_inds]*A2nm, gaia_wav))
    hip23309_rebinmore_spec_flux=np.concatenate((uvit_bindownmore*0.01, HIP23309_NUV['flux'].to_numpy()[nuv_uvit_inds]*0.01, gaia_flux))
    #Factor of 0.01 to convert erg s**-1 cm**-2 A**-1 to W m**-2 nm**-1
    savedata=np.column_stack((hip23309_rebinmore_spec_wav, hip23309_rebinmore_spec_flux))
    np.savetxt('../uv-prebiochem/Raw_Data/Mdwarf_Spectra/Steady-State/UVIT/HIP23309spec_rebinned.txt', savedata, delimiter='\t', newline='\n', fmt='%3.6f %1.6e') #Print checkfile for reaction rates.
    
    fig1, ax=plt.subplots(1, figsize=(8,6))
    ax.plot(hip23309_spec_wav, hip23309_spec_flux, color='black')
    ax.plot(hip23309_hstextended_spec_wav, hip23309_hstextended_spec_flux, color='green', linestyle='--')
    ax.plot(gaia_wav, gaia_flux, color='red')
    ax.plot(model_wav, model_flux*0.01, color='blue')
    ax.set_xlim([200,350])

    ax.set_yscale('log')
    ax.set_ylim(bottom=1.0E-18)
    
    


