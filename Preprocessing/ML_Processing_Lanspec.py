import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sp
import scipy.stats as sps
from scipy.optimize import curve_fit
import glob
from scipy.special import gamma, factorial
from scipy.signal import find_peaks
import os

########
# Inputs:
R = 30           # resistance (ohm)
M = 1            # X parameter
D = 0.000127     # Probe Diameter [m]
L = 0.005128     # Probe Height [m]
########
# Langmuri Probe Data 
T_e = []; n_e = []; n_i = []
p_10 = []; p_11 = []; p_12 = []
p_13 = []; p_14 = []; p_15 = []; p_16 = []
pow_ = []; pup_ = []; pdn_ = []
flow_ = []
# Spectroscopic Data
p_696_57_ = []; p_706_73_ = []; p_738_45_ = []; p_750_42_ = []
p_751_52_ = []; p_763_55_ = []; p_772_44_ = []; p_794_90_ = []
p_800_70_ = []; p_801_57_ = []; p_810_43_ = []; p_811_58_ = []
p_826_49_ = []; p_840_90_ = []; p_842_53_ = []; p_852_21_ = []
p_912_33_ = []; p_922_48_ = []; p_965_80_ = []
max_param = []
# Create a folder on a desktop and name it "ML" and choose it as path_directory
# Drop in Langmuir Probe and Spectroscopy Data into this folder for preprocessing
# Path Directory
path_dir = r'/Users/scienceman/Desktop/ML/'
def Langmuir_Data():
    result = []
    for root, dirs, files in os.walk(path_dir):
        for dd in files:
            if dd[0] == '1':
                result.append(os.path.join(root,dd))
    return result
def Spectroscopic_Data():
    result = []
    for root, dirs, files in os.walk(path_dir):
        for d in files:
            if d[0] == 's':
                result.append(os.path.join(root,d))  
    return result
# Calibration File for Spectroscopic Measurements
# Include path directory '/Users/scienceman/Desktop/Deep_Learning/Spectro_Lang_Data/ML_Data/'
pcaldir = r'path_directory'
thor_dir = pcaldir + 'SLS201L_Spectrum_Calibration.csv'
cal_dir = pcaldir + 'lamp.csv'
calib = pd.read_csv(cal_dir)
# Calibration File on Thors lab for SLS201L Red Lamp
thor = pd.read_csv(thor_dir)
# Wavelength (nm)
wave_thor = np.array(thor.iloc[:,0]).astype(float)
# Intensity (a.u.)
int_thor = np.array(thor.iloc[:,1]).astype(float)
wave_interp = np.linspace(340,1000,1000000)
# Interpoated Calibration File on Thor's Lab
int_thorp = np.interp(wave_interp,wave_thor, int_thor)
calib = pd.read_csv(cal_dir) # Calibration File taken in the lab with the red lamp
wave_cal = np.array(calib.iloc[:,0]).astype(float)
# Intensity of measurement with red lamp
int_cal = np.array(calib.iloc[:,1]).astype(float)
# Smoothed Intensity of the measurement with red lamp 
int_cal_sav = sp.savgol_filter(int_cal,4001, 2)
# Interpolated Smoothed Intensity of the red lamp 340-1000 nm range
int_calp = np.interp(wave_interp, wave_cal, int_cal_sav)
# Calibration File
calib_file = int_thorp / int_calp
######
for s in Spectroscopic_Data():
    for f in Langmuir_Data():
        if str(f[32:]) == str(s[34:]):
            # Langmuir Probe Data Analysis 
            print('___________________________________')
            print(f[32:])
            ## Importing the CSV files
            T = pd.read_csv(f, delimiter= ',')
            # Column #1: Time (ms), Column #3: Avg(A), Column #4: Avg(B)
            # time (s)
            limit = np.argmax(np.array(T.iloc[1:,4]).astype(float)) - 200
            time = np.array(T.iloc[1:limit,0]).astype(float)
            # Average Channel A
            avg_A = - np.array(T.iloc[1:limit,3]).astype(float)
            # Average Channel B
            if T.iloc[0,4] == '(mV)':
                avg_B = np.array(T.iloc[1:limit,4]).astype(float) / 1000
            else:
                avg_B = np.array(T.iloc[1:limit,4]).astype(float)
            ##### 
            # Savitzky-Golay filter: polyorder: 2, window length 5001 of avg_B
            avg_Bs = sp.savgol_filter(avg_B,5001, 2)
            #####
            # Linear fit avg_A
            slope, intercept, r_value, p_value, std_err = sps.linregress(time, avg_A)
            avg_Al = (slope * time) + intercept
            #####
            real_V = (10 * avg_Al) - (M * avg_Bs)
            #####
            real_I = (M * avg_Bs) / R
            Secd_real_I = np.gradient(np.gradient(real_I, real_V), real_V)
            ##### 
            def smooth(y, box_pts):
                box = np.ones(box_pts)/box_pts
                y_smooth = np.convolve(y, box, mode='same')
                return y_smooth
            #### smoothed Second Derivative of real Current
            smsd_real_I = smooth(Secd_real_I, 1001)
            #### 
            b = np.argmax(smsd_real_I)
            #plt.plot(real_V[b:], abs(np.log10(Secd_real_I[b:])))
            peak_proof = (-np.log10(abs(smsd_real_I[b:])))
            mean_peak = np.mean(peak_proof)
            std_peak = np.std(peak_proof)
            peaks = (find_peaks(peak_proof, height= mean_peak + 1*std_peak))
            ##### Plasma Voltage
            plasma_V = real_V[b:][peaks[0][0]]
            #print('Plasma Potential',plasma_V)
            #plt.plot(real_V, smsd_real_I)
            ##### Energy  (eV)
            E = plasma_V - real_V
            #####
            e = 1.6021765E-19    # Elementary Electron Charge [C]
            me = 9.10938E-31     # Mass of an Electron
            ##### EEDF (eV^-1)
            ####
            EEDF = e * (np.sqrt((8 * me)) / ((e**3)*np.pi*D*L))* smsd_real_I * np.sqrt(e*E)
            NaN = np.argwhere(np.isnan(EEDF))[0][0]
            N = np.argwhere(EEDF[:NaN] < 0)[::-1][0][0]
            E_range = E[:NaN][N:][::-1]+1E-200
            EEDF_range = EEDF[:NaN][N:][::-1]
            ####### EEPF (eV^-1.5)
            EEPF = EEDF_range / np.sqrt(E_range)
            #print(np.max(EEPF))
            max_param.append(np.max(EEPF))
            #print(np.max(EEPF))
            ####### EEDF * E
            E_EEDF = EEDF_range * E_range
            ######
            # Electron Density (1/m3)
            ne = np.trapz(EEDF_range, E_range)
            # Electron Temperature (eV)
            Te = (0.66666666666) * (1/ ne) *  np.trapz(E_EEDF, E_range)
            ######
            def func(x, a, b):
                return a + b * (x ** 0.5)  
            for ii in np.arange(start=1E4, stop=1E6, step=1E3):
                edge = (np.argwhere(real_I < 0)[::-1][0][0] - int(ii))
                popt, pcov = curve_fit(func, E[:edge], real_I[:edge])
                fit = popt[0] + (popt[1]) * (E[:edge] ** 0.5) 
                edge_I = real_I[:edge][::-1][0]
                edge_fit = fit[-1]
                if abs((edge_I - edge_fit)/ edge_I) > 0.001:
                    continue
                else:
                    edge = edge
                    popt, pcov = curve_fit(func, E[:edge], real_I[:edge])
                    b = popt[1]
                    #print(b)
                    #plt.plot(E[:edge], real_I[:edge])
                    #plt.plot(E[:edge], fit)
                    break
                break
            b = abs(popt[1])
            #mi = 6.6335209E-26      # Mass of Ar [kg]
            mi = 3*1.6735575E-27   # Mass of H3+ [kg]
            #mi = 4*1.6735575E-27   # Mass of He [kg]
            ni = (b * np.sqrt(mi)) / (2 * e * (D/2) * L * np.sqrt(2*e))
            #print('Electron Density (1/m3)' , ne)
            #print('Ion Density (1/m3)' , ni)
            #print(ni/ne)
            #print('Electron Temperature (eV)', Te)
            #print('Ion/Electron Density Ratio', ni/ne)
            EEPF_norm = EEPF / np.trapz(EEPF,E_range)
            E_test = np.linspace(1,18,10000)
            EEPF_test = np.interp(E_test, E_range, (EEPF))
            cut_test = np.argwhere(EEPF_test <= 2E13)[0][0]
            ks_E = E_test[:cut_test]
            ks_EEPF = EEPF_test[:cut_test]
            ks_EEDF = ks_EEPF * np.sqrt(ks_E)
            ks_ne = np.trapz(ks_EEDF,ks_E)
            ks_norm_EEDF = ks_EEDF / ks_ne
            ks_norm_EEPF = (ks_EEDF / ks_ne) / np.sqrt(ks_E)
            ks_fit = ((ks_EEDF / ks_ne)) / ks_E
            # Break energy into two segments
            E1 = np.linspace(1.0,6,1000)
            E2 = np.linspace(5,ks_E[-1],1000)
            ks_fit1 = np.interp(E1, ks_E, ks_fit)
            ks_fit2 = np.interp(E2, ks_E, ks_fit)
            plt.plot(E1, ks_fit1)
            plt.plot(E2, ks_fit2)
            plt.grid(True)
            fig = plt.gcf()
            plt.yscale('log')
            fig.set_size_inches(10,10)
            ax1 = plt.subplot(2,2,1)
            '''
            ax1.plot(E1, ks_fit1, linewidth=1)
            ax1.plot(E2, ks_fit2, linewdith=1)
            #ax1.ylim(1e-4, 5e-1)
            #plt.xlim(-1,25)
            plt.ylim(1e-4, 10e-1)
            plt.xlabel('Energy (eV)', fontsize=12)
            plt.ylabel('EEPF $(eV^{-3/2})$', fontsize=12)
            plt.grid(True)
            fig = plt.gcf()
            plt.yscale('log')
            fig.set_size_inches(10,10)
            '''
            #def lm_fit(x,a,b,c,d,e):
             #   return a*np.sqrt(x*e)+f*np.sqrt(h*x) + (c**-1.5)*d*np.exp(-(x/c)**b)
            def bifit(x,a,b,c):
                return (c**-1.5)*a*np.exp(-(x/c)**b)
            
            def funcc(x, a,b,c,d,e,f):
                it1 = (c**-1.5)*a*np.exp(-(x/c)**b)
                it2 = (d**-1.5)*e*np.exp(-(x/d)**f)
                return (it1 + it2)
            def func_ks(x, a,b,c,d,e):
                it1 = (c**-1.5)*a*np.exp(-(x/c)**b-(x/d))
                return (it1)
            
            p1, pcov1 = curve_fit(bifit,E1,ks_fit1)
            p2, pcov2 = curve_fit(bifit,E2,ks_fit2)
            print('EEDF_range/ne: a, b, c', p1)
            print('EEDF_range/ne: c, d, f', p2)
            fit1 = bifit(E1, p1[0], p1[1], p1[2])
            fit2 = bifit(E2, p2[0], p2[1], p2[2])
            ax2 = plt.subplot(222)
            ax2.plot(E1, fit1)
            ax2.plot(E2, fit2, label='%s' % f[32:])
            #ax2.plot(LM_E,LM_fit)
            #ax2.plot(LM_E,fit1)
            #plt.plot(LM_E,fit1)
            #ax2.plot(LM_E, fit1)
            #plt.legend()
            #plt.plot(E_range, np.fft.fft(EEPF), label='%s' % f[32:])
            #plt.plot(real_V, smsd_real_I)
            #plt.ylim(1e-4, 10e-1)
            #plt.xlim(-1,25)
            plt.xlabel('Energy (eV)', fontsize=12)
            plt.ylabel('EEPF $eV^{-3/2}$', fontsize=11)
            plt.ylim(1e-4, 10e-1)
            plt.grid(True)
            fig = plt.gcf()
            plt.yscale('log')
            fig.set_size_inches(10,10)
            plt.grid(True)
            n_i.append(ni)
            n_e.append(ne)
            T_e.append(Te)
            # pressure Upstream (Torr)
            pup_.append(float(f[-12:-8]))
            # Pressure Downstream (mTorr)
            pdn_.append(float(f[-22:-18]))
            # Plasma Power (W)
            pow_.append(float(f[-27:-24]))
            # Argon Flow Rate
            flow_.append(float((f[-35:-32])))   
            # Fitting Parameters
            p_10.append(p1[0])
            p_11.append(p1[1])
            p_12.append(p1[2])
            p_13.append(p2[0])
            p_14.append(p2[1])
            p_15.append(p2[2])
            # Spectroscopic Data Analysis 
            # Importing the the csv files for the spectroscopic measurements
            Em = pd.read_csv(s, delimiter = ',')
            wave = np.array(Em.iloc[:,0]).astype(float)
            spect = np.array(Em.iloc[:,1]).astype(float)
            spect_p = np.interp(wave_interp,wave,spect)
            # Offseting the baseline to zero
            spect_pp = spect_p - spect_p[0]
            # Calibrating the emission spectrum with the calib file
            calib_spect = spect_pp * calib_file
            # Area of the emission spectrum 
            area_spect =  np.trapz(calib_spect,wave_interp)
            # Normalized emission spectrum
            norm_spect = calib_spect / area_spect
            # All normalized peak intensities
            g = 0.01
            p_696_57_.append(np.max(find_peaks(norm_spect[540000:540600], height= g)[1]['peak_heights']))
            p_706_73_.append(np.max(find_peaks(norm_spect[555258:556108], height= g)[1]['peak_heights']))
            p_738_45_.append(np.max(find_peaks(norm_spect[603412:604102], height= g)[1]['peak_heights']))
            p_750_42_.append(np.max(find_peaks(norm_spect[621581:622200], height= g)[1]['peak_heights']))
            p_751_52_.append(np.max(find_peaks(norm_spect[623115:623915], height= g)[1]['peak_heights']))
            p_763_55_.append(np.max(find_peaks(norm_spect[641456:642106], height= g)[1]['peak_heights']))
            p_772_44_.append(np.max(find_peaks(norm_spect[655008:655648], height= g)[1]['peak_heights']))
            p_794_90_.append(np.max(find_peaks(norm_spect[689001:689621], height= g)[1]['peak_heights']))
            p_800_70_.append(np.max(find_peaks(norm_spect[697800:698411], height= g)[1]['peak_heights']))
            p_801_57_.append(np.max(find_peaks(norm_spect[699029:699729], height= g)[1]['peak_heights']))
            p_810_43_.append(np.max(find_peaks(norm_spect[712454:713104], height= g)[1]['peak_heights']))
            p_811_58_.append(np.max(find_peaks(norm_spect[714216:714916], height= g)[1]['peak_heights']))
            p_826_49_.append(np.max(find_peaks(norm_spect[737000:737517], height= g)[1]['peak_heights']))
            p_840_90_.append(np.max(find_peaks(norm_spect[758613:759200], height= g)[1]['peak_heights']))
            p_842_53_.append(np.max(find_peaks(norm_spect[761122:761822], height= g)[1]['peak_heights']))
            p_852_21_.append(np.max(find_peaks(norm_spect[775800:776480], height= g)[1]['peak_heights']))
            p_912_33_.append(np.max(find_peaks(norm_spect[867000:867579], height= g)[1]['peak_heights']))
            p_922_48_.append(np.max(find_peaks(norm_spect[882255:882955], height= g)[1]['peak_heights']))
            p_965_80_.append(np.max(find_peaks(norm_spect[948000:948595], height= g)[1]['peak_heights']))
            #print(p_696_57_,p_706_73_, p_738_45_,p_750_42_,p_751_52_,p_763_55_,p_772_44_,p_794_90_,p_800_70_,
            #      p_801_57_,p_810_43_,p_811_58_, p_826_49_,p_840_90_,p_842_53_,p_852_21_, p_912_33_,p_922_48_,
            #      p_965_80_)
            '''
            ax3 = plt.subplot(2,1,2)
            plt.grid(True)
            fig.set_size_inches(10,10)
            plt.rcParams.update({'font.size' : 12})  
            plt.xlabel('Wavelength (nm)', fontweight='bold')
            plt.ylabel('Intensity (a.u.)', fontweight='bold')
            ax3.plot(wave_interp,norm_spect)
            '''
def zero_list(n):
    list_zeros = [0] * n
    return list_zeros
def two_list(n):
    list_zeros = [14] * n
    return list_zeros

# Saving all langmuir probe and spectroscopic processed outputs

#np.savetxt('Norm_EEDF_LM_best.txt', np.column_stack([E_range, (EEPF)]), delimiter= ',', header= 'Energy (eV), EEDF (eV-1)')

np.savetxt('Square_14He.csv', np.column_stack([p_696_57_,p_706_73_,p_738_45_,p_750_42_,p_751_52_,p_763_55_,
                                                     p_772_44_,p_794_90_,p_800_70_,p_801_57_,p_810_43_,p_811_58_,
                                                     p_826_49_,p_840_90_,p_842_53_,p_852_21_,p_912_33_,p_922_48_,
                                                     p_965_80_,n_i, n_e, T_e, pup_, pdn_, pow_,p_10,p_11,p_12,
                                                     p_13,p_14,p_15, zero_list(len(p_811_58_)),two_list(len(p_811_58_)),flow_ ]), delimiter=',', header=
           'p_696_57, p_706_73,p_738_45,p_750_42, p_751_52, p_763_55, p_772_44, p_794_90, p_800_70, p_801_57, p_810_43,p_811_58'+
           ', p_826_49, p_840_90, p_842_53, p_852_21, p_912_33, p_922_48,p_965_80, Ion Density (1/m3), Electron Density (1/m3),'+
           'Electron Temperature (eV),Pressure Upstream (Torr),Pressure Downstream (mTorr), Plasma Power (W),'+
           'fit10, fit11, fit12, fit13, fit14, fit15,H2_Flow_Rate (sccm), He_Flow_Rate,Argon Flow Rate (sccm)')

'''
np.savetxt('ML_Inputs_Outputs_`10H2_Protocol.csv', np.column_stack([n_i, n_e, T_e, pup_, pdn_, pow_,p_10,p_11,p_12, p_20,p_21,p_22,p_30,p_31,p_32,
                                                          two_list(len(n_i)),zero_list(len(n_i)),flow_]), delimiter=',',
           header='Ion Density (1/m3), Electron Density (1/m3),Electron Temperature (eV),Pressure Upstream (Torr),Pressure Downstream (mTorr)'+
           ', Plasma Power (W),fit10, fit11, fit12, fit20, fit21, fit22, fit30, fit31, fit32,H2_Flow_Rate (sccm), He_Flow_Rate,Argon Flow Rate (sccm)')


'''


