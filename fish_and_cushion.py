# -*- coding: utf-8 -*-
'''
    Fish_and_cushion.py

    Automated FCS and FCCS analysis

    Main Autor:
    
    Alex Koch 2021 (alexander.koch@manchester.ac.uk, alexkoch22@protonmail.com)

    Contributors: Hannah Perkins, James Bagnall 

    This script does *not* analyse flow cytometry standard FCS files.

'''

import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from math import *
import os
import itertools
import datetime
import re # For regular expressions

# For testing stationarity
from statsmodels.tsa.stattools import adfuller, kpss

# Fitting
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution
import warnings

import plotly
import plotly.graph_objs as go # Graphs
import plotly.express as px
import colorlover as cl
import plotly.io as pio

# Import library modules
from fcsfiles import *

# User interface
import tkinter
from tkinter import filedialog

# Import models
from models import *
from simulated_file_reader import *

def reindex(arr, indices):
    new_array = []
    for i in range(len(indices)):
        new_array.append(arr[indices[i]])
    return new_array

def colour_picker(channel, opacity):
    if channel == 'Auto-correlation 1':
        return 'rgba(51, 204, 51, ' +  str(opacity) + ')'
    elif channel == 'Auto-correlation 2':
        return 'rgba(204, 0, 0, ' + str(opacity) + ')'
    elif channel == 'Cross-correlation 1 versus 2':
        return 'rgba(0, 153, 204, ' + str(opacity) + ')'
    elif channel == 'Cross-correlation 2 versus 1':
        return 'rgba(255, 153, 0, ' + str(opacity) + ')'

def calc_diffusion(tau_D, tau_D_error, alpha=1, alpha_error=0, r=0.22):
    D = r**2 / (4 * tau_D**alpha)
    #D_error = D * tau_D_error / tau_D
    D_error = np.sqrt((alpha**2 * (r**4 / 16) * tau_D**(-2*(alpha+1)) * tau_D_error**2) + D**2 * (np.log(tau_D))**2 * alpha_error**2)
    return D, D_error

def stationarity_test(count_rate_array):
    '''

    Parameters:

        count_rate_array | Numpy array of ints shape = [2, *Length of count data*] | The number of counts over time

    Returns:

        stationary | Bool | True is if the counts over time is stationary as defined by both the ADF and KPSS test
    '''
    if count_rate_array.shape[0] == 0:
        # Cross-correlation data, therefore no count rate array
        return "", "", "", "", ""

    counts = count_rate_array[:, 1]
    # ADF test
    adf_result = adfuller(counts)
    if adf_result[1] > 0.05:
        # Not stationary
        ADFStat = False
    else:
        # Stationary
        ADFStat = True
    
    # KPSS test
    statistic, p_value, n_lags, critical_values = kpss(counts, regression='c', lags = "auto")
    if p_value < 0.05:
        # Not Stationary
        KPSSStat = False
    else:
        # Stationary
        KPSSStat = True

    stationary = ADFStat and KPSSStat
    KPSS_pvalue = 1 - p_value # Convert the KPSS pvalue to be the same interpretation as the ADF test
    return ADFStat, adf_result[1], KPSSStat, KPSS_pvalue, stationary

def score_signal(count_rate_array):
    if count_rate_array.shape[0] == 0:
        # Cross-correlation data, therefore no count rate array
        return "", "", ""
    with np.errstate(over='ignore'): # ignore overflow warnings in exponential function
        exp_decay = lambda t, I_0, lam: I_0 * np.exp(-lam * t)
        t = count_rate_array[:,0]
        I = count_rate_array[:,1]
        try:
            fit_params, cov_mat = curve_fit(exp_decay, t, I)
            fraction_bleached = 1 - exp_decay(count_rate_array[-1,0], *fit_params) / exp_decay(0, *fit_params)
            decay_rate = fit_params[1]
            fit_errors = np.sqrt(np.diag(cov_mat))

        except:
            fraction_bleached = 'N/A'
            decay_rate = 'N/A'
        mean_intensity = np.mean(count_rate_array[:,1])
        

    return decay_rate, fraction_bleached, mean_intensity

def rename_channel(channel):
    if channel == "Auto-correlation detector Meta1":
        channel = "Auto-correlation 1"
    if channel == "Auto-correlation detector Meta2":
        channel = "Auto-correlation 2"
    if channel == "Cross-correlation detector Meta1 versus detector Meta2":
        channel = "Cross-correlation 1 versus 2"
    if channel == "Cross-correlation detector Meta2 versus detector Meta1":
        channel = "Cross-correlation 2 versus 1"
    return channel

#Function to add units to times
def unitFix(variable):
    milliUnitFinder = re.compile('ms')
    milli = milliUnitFinder.findall(variable)
    microUnitFinder = re.compile('µ')
    micro = microUnitFinder.findall(variable)
    alt_microUnitFinder = re.compile('u')
    alt_micro = alt_microUnitFinder.findall(variable)
    nanoUnitFinder = re.compile('n')
    nano = nanoUnitFinder.findall(variable)
    if len(milli) == 1:
        variable = 1e-03*float(variable[0:-2])
    elif len(micro) == 1 or len(alt_micro) == 1:
        variable = 1e-06*float(variable[0:-2])
    elif len(nano) == 1:
        variable = 1e-09*float(variable[0:-2])
    else:
        variable = float(variable[0:-2])
    return variable

def background_correction(corr, average_signal, background_signal):
    beta = (average_signal / (average_signal - background_signal))**2
    corr *= beta
    return corr

#def photo_bleaching_correction(tau, fraction_bleached, measurement_time):
def photo_bleaching_correction(tau, decay_rate):
    #alpha = -np.log(float(fraction_bleached)) / measurement_time
    #return (1 + alpha * tau / 2 + (alpha * tau)**2 / 6)**(-1)
    return np.exp(-decay_rate * tau)

def correlator_error(tau, average_intensity, max_time, correlator_bin_time, beta, d_guess, wxy, molecular_brightness, error_function='None'):
    '''
    Calculates the variance of the correlation due to the averaging effects of the multi-tau correlator algorithm as set out by Saffarian and Elson 2001 (equation 31)

    Parameters:

        tau                  | Numpy array of floats shape=[*number of lag times*]  | Array of lag times 
        average_intensity    | Float                                                | Average number of counts per second recorded by the detector
        max_time             | Float                                                | The maximum lag time 
        correlator_bin_time  | Float                                                | The smallest lag time used by the multi-tau correlator algorithm of which all other lag times are multiples of
        beta                 |                                                      |
        d_guess              | Float                                                | An estimated diffusion rate for calculating the 'particle' noise in the 'saffarian_full' error function
        wxy                  | Float                                                | The radius of the confocal volume
        molecular_brightness | Float                                                | The maximum number of photons emitted per second by a molecule placed in the centre of the observation volume
        error_function       | String                                               | One of: 'None', 'saffarian_reduced', 'saffarian_simple' or 'saffarian_full'

    Returns:

        sigma | Numpy array of floats shape=[*number of lag times*]
    '''
    alpha = 1
    sigma = []
    if error_function == 'None' or average_intensity == '':
        sigma = np.zeros(len(tau))
    elif error_function == 'saffarian_reduced':
        for k in range(len(tau)):
            sigma.append(np.sqrt(1/ (tau[k] * average_intensity**2)))
    elif error_function == 'saffarian_simple' or error_function == 'saffarian_full':
        # Saffarian and Elson 2001 equation 31 
        # First calculate the binned time overwhich each point the correlation is calculated over

        # Very small changes in the recorded lag time mean that if the sampling time is found using the lag time, then the subsequent
        # values calculated using the sampling time will be decimals when they are supposed to be integers. Therefore, using the 
        # correlator time and the multi-tau correlator architecture to calculate the sampling times rather than using the lag time.

        sampling_time = np.array(correlator_bin_time, dtype = np.float64)
        for ii in range(1, len(tau)):
            if ii < 14:
                sampling_time = np.append(sampling_time, correlator_bin_time)
            else:
                factor = (ii)//8
                sampling_time = np.append(sampling_time, np.power(2, factor - 1) * correlator_bin_time) 
        #N is the number of measurements over a sampling period taken during the experiment. This can't be a fraction of a
        #measurement, so taking the floor of the value
        N = np.floor(max_time / sampling_time)
        # v is an integer where v*delta_t (the sampling time) gives the lag time. N-v then gives the number of possible pairs 
        v = np.round(tau / sampling_time)
        # Calculate the average number of counts in each dwell (lag time) by multiplying the average fluorescence (a frequency) 
        # over the whole trace with each lag time
        mean_fluorescence = average_intensity * sampling_time 
        # Calculate standard deviation as defined in equation 31 from Saffarian and Elson 2001 
        # for k in range(len(tau)):
        sigma = np.sqrt(1 / ((N - v) * mean_fluorescence**2))
        #If the error function is the full version, add in the extra requirements
        if error_function == 'saffarian_full':
            #Following Palo et al. doi 10.1016/S0006-3495(00)76523-4 to find the apparent brightness, q_App
            #Define the constant gamma1 for a 3D Gaussian beam
            gamma1 = 1/(2*np.sqrt(2))
            #The brightness in counts of the fluorophore, q
            q = molecular_brightness * sampling_time
            #Factor, t, as defined by Palo et al. doi 10.1016/S0006-3495(00)76523-4
            t = d_guess * sampling_time / (wxy ** 2)
            #To make the calculation easier, set up 2 factors, A and C
            A = (1-beta) ** 0.5
            C = ((1 + beta * t) ** 0.5) - 1
            #Calculate the apparent brightness
            Btsq = beta * t**2
            prefactor = 4 * q / (Btsq * A)
            term1 = beta * (1 + t) *np.arctanh((A * C) / (beta + C))
            term2 = A * C
            q_App = prefactor * (term1 - term2)
            sigma = np.sqrt(((1 + (gamma1 * q_App))**2) / ((N - v) * mean_fluorescence**2))

    else:
        print('ERROR: Correlator error function incorrectly defined')
        sigma = np.zeros(len(tau))

    return np.array(sigma)

def model_selection(indices, data_points, sum_squared_errors, models, test='AIC'):
    '''
    Selects the best scoring (lowest score) model from a set of candidate correlation models that have been fit to the correlation data
    by applying either the Akaike Information Criterion (AIC) or Bayesian Information Criterion (BIC) to each model fit. 

    AIC and BIC implementation from https://machinelearningmastery.com/probabilistic-model-selection-measures/

    Parameters:

        indices            | Array of ints          | Array of indices that map to the correct location in the models and fits arrays 
        data_points        | Int                    | Number of data points 
        sum_squared_errors | Array of floats        | The sum of the squared residual errors from the curve fitting
        models             | Array of model classes | The candidate correlation models 
        test               | String                 | One of: AIC, BIC
    
    Returns:

        indices            | Array of ints          | Array of indices that map the new arrangement, with the best scoring models first 

    '''
    assert len(sum_squared_errors) == len(models)
    selection_scores = []
    #indices = range(len(models)) #[[ind] for ind in range(len(models))]
    if test == 'AIC':
        # Preferred model is the one with the minimum AIC value
        for m in range(len(sum_squared_errors)):
            number_of_parameters = len(models[m].args)
            mean_squared_error = sum_squared_errors[m] / data_points
            aic = data_points * np.log(mean_squared_error) + 2 * number_of_parameters
            #print("AIC = %s" %aic)
            selection_scores.append(aic)

    if test == 'BIC':
        # Preferred model is the one with the minimum BIC value, 
        # the penalty term for the number of parameters is larger than that of AIC
        for m in range(len(sum_squared_errors)):
            number_of_parameters = len(models[m].args)
            mean_squared_error = sum_squared_errors[m] / data_points
            bic = data_points * np.log(mean_squared_error) + number_of_parameters * np.log(data_points)
            selection_scores.append(bic)
    
    # Sort according to the lowest score 
    zipped_list = zip(indices, selection_scores)
    zipped_list = list(zipped_list)
    sorted_list = sorted(zipped_list, key=lambda x: x[1])
    #print("sorted list = %s" %sorted_list)
    indices, selection_scores = [], []
    for el in sorted_list:
        indices.append(el[0])
        selection_scores.append(el[1])
        
    #print("indices = %s, selection_scores = %s" %(indices, selection_scores))
    return list(indices), list(selection_scores)

def fit_curve(tau, corr, model, sigma, structural_parameter, use_bounds = True, reject_fits_equal_to_bounds=False):
    '''

    Parameters:

        tau                  | Array of floats  | Array of lag times 
        corr                 | Array of floats  | Array of correlation values
        model                | Model class      | Class that wraps the correlation function to be fit, see models.py
        sigma                | Array of floats  | Array of standard deviations around each point in the correlation array
        structural_parameter | Float            | Parameter used in the correlation model that describes the shape of the confocal volume
        use_bounds           | Bool             | Whether physical bounds should be during the final round of curve fitting
    
    Returns:

        fit                  | Array of floats  | Array of fitted correlation values
        fit_params           | Array of floats  | The fitted parameters of the correlation model
        fit_errors           | Array of floats  | The standard deviation in the fitted parameters         
        sum_squared_errors   | Float            | The sum of the squared residual errors from the curve fitting
        fit_Rsquared         | Float            | The R squared value of the fitted curve
        valid_fit            | Bool             | True if the curve was fit 
        chi2                 | Float            | The reduced Chi squared value, should sit between 0.5 and 2 for a good fit

    '''
    # To fix the issue of generating incorrect fits we need to cast the inputs as 64 bit floats
    # See: https://github.com/scipy/scipy/issues/7117
    tau = tau.astype(dtype=np.float64)
    corr = corr.astype(dtype=np.float64)
    if sigma is not None:
        # Only cast sigma if it is defined
        sigma = sigma.astype(dtype=np.float64) 
    bounds = (-np.inf, np.inf)
    if use_bounds == True:
        bounds = model.bounds
    
    # Set the structural parameter for use in the model, this is not a functional way of programming but it gets around the 
    # limitations of the scipy curve fit function which will fit all arguments of the function given to it
    model.a = structural_parameter

    def sum_of_squared_errors(parameterTuple):
        #HANNAH added this function
        warnings.filterwarnings("ignore") # do not print warnings by genetic algorithm
        val = model.correlation(tau, *parameterTuple)
        return np.sum((corr - val) ** 2.0)

    def generate_Initial_Parameters():
        #HANNAH added this function
        #Add in a loop here for the number of parameters for each model
        bounds = model.bounds
        boundsReshape = tuple(zip(*bounds))
        parameterBounds = []
        for i in range(len(boundsReshape)):
            parameterBounds.append(boundsReshape[i]) # search bounds for ith variable
        #print(parameterBounds)
        # "seed" the numpy random number generator for repeatable results
        result = differential_evolution(sum_of_squared_errors, parameterBounds, seed=3)
        return result.x
    #HANNAH added the use of generate_Initial_Parameters
    initial_params = generate_Initial_Parameters()
    #initial_params = model.guess_values
    valid_fit = False
    reason_for_rejection = ''
    try:
        fit_params, cov_mat = curve_fit(model.correlation, tau, corr, p0=initial_params, bounds=bounds, sigma=sigma, method='trf', **{'loss':'arctan'})
        if reject_fits_equal_to_bounds == True:
            for i in range(len(fit_params)):
                if fit_params[i] == bounds[0][i] or fit_params[i] == bounds[1][i]:
                    print('Hit bound')
                    raise Exception("Fit parameter " + model.args[i] + " equal to bounds") 

        fit_errors = np.sqrt(np.diag(cov_mat))
        # manually calculate R-squared goodness of fit
        fit_residual = corr - model.correlation(tau, *fit_params)
        sum_squared_errors = np.sum(fit_residual**2)
        fit_Rsquared = 1 - np.var(fit_residual) / np.var(corr)
        fit = model.correlation(tau, *fit_params)
        # calculate the reduced Chi squared value (Chi squared divided by the number of degrees of freedom). It should be between 0.5 and 2
        chi2 = (1/(len(corr)-len(model.args))) * np.sum(((corr - fit)/(sigma))**2)
        valid_fit = True

    except Exception as error:
        print("ERROR Couldn't fit: %s" %error)
        valid_fit = False
        fit_params = None
        fit_errors = None
        sum_squared_errors = None
        fit_Rsquared = None
        fit = None
        chi2 = None
        reason_for_rejection = error

    return fit, fit_params, fit_errors, sum_squared_errors, fit_Rsquared, valid_fit, reason_for_rejection, chi2

def analyse_fcs_files(fcs_files, image_directory, analysis_directory, excel_file, settings):
    #model = lookup_model(settings["auto-correlation-model"])
    background_signal = 0
    if settings["background_correction"] == "True":
        # Load up background FCS file
        #print("Load background file")
        background_signal = float(input("Input average background signal (Hz): "))
    else:
        print("Initialise background counts to zero")
    combined_traces_figures = dict()
    def gen_cols(args):
        for i in args:
            yield i
            yield str(i) + " std"
    diffusion_cols = ["D", "D std"]

    df_best_fit = pd.DataFrame()
    df_best_cross = pd.DataFrame()
    model_dictionary = dict()
    
    interactions = False
    #interaction_data = {'x_A': [], 'x_B': [], 'y_A': [], 'y_B': [], 'Complex': [], 'Free x Free': [], 'file': [], 'Kd_A': [], 'Kd_A_std': [], 'Kd_B': [], 'Kd_B_std': [], 'Kd_line': [], 'Kd_line_std': [], 'data_points': []}
    interaction_data = {'x_A': [], 'x_B': [], 'y_A': [], 'y_B': [], 'file': [], 'Kd_A': [], 'Kd_A_std': [], 'Kd_B': [], 'Kd_B_std': [], 'data_points': []}

    # Go through each file
    for fcs_file in tqdm(fcs_files):
        fcs = None

        try:
            experimental_data = False
            if fcs_file.endswith(".fcs"): 
                fcs = ConfoCor3Fcs(fcs_file)
                experimental_data = True
            elif fcs_file.endswith(".sim"):
                fcs = SimulatedFCS(fcs_file)
                experimental_data = False

        except Exception as e:
            print('Error: ' + str(e) + '\n Cannot open file {:}'.format(fcs_file))
            continue

        channels = fcs['FcsData']['FcsEntry'][0]['FcsDataSet']['Acquisition']['AcquisitionSettings']['Channels']
        if channels == 4:
            interactions = True
            
        positions = int(fcs['FcsData']['FcsEntry'][0]['FcsDataSet']['Acquisition']['AcquisitionSettings']['SamplePositions']['SamplePositionSettings']['Positions'])
        if positions == 0:
            positions = 1
        #print("Positions = %s" %int(positions))
        
        '''
        Determine which channels have been used. Add the relevant models to file
        '''
        multiple_models = False
        all_model_arguments = []
        for channel_index in range(channels):
            channel = fcs['FcsData']['FcsEntry'][channel_index]['FcsDataSet']['Channel']
            if channel == "Auto-correlation detector Meta1":
                #models_.append(gen_cols(lookup_model(settings["auto_correlation_1_model"]).args))
                if len(settings["auto_correlation_1_model"]) > 1:
                    multiple_models = True
                for model in settings["auto_correlation_1_model"]:
                    arguments = gen_cols(lookup_model(model).args)
                    all_model_arguments.append(arguments)
                    if model not in model_dictionary:
                        model_dictionary[model] = {'arguments': list(set().union(*arguments))}
            elif channel == "Auto-correlation detector Meta2":
                #models_.append(gen_cols(lookup_model(settings["auto_correlation_2_model"]).args))
                if len(settings["auto_correlation_2_model"]) > 1:
                    multiple_models = True
                for model in settings["auto_correlation_2_model"]:
                    arguments = gen_cols(lookup_model(model).args)
                    all_model_arguments.append(arguments)
                    if model not in model_dictionary:
                        model_dictionary[model] = {'arguments': list(set().union(*arguments))}
            elif channel == "Cross-correlation detector Meta1 versus detector Meta2" or channel == "Cross-correlation detector Meta2 versus detector Meta1":
                #models_.append(gen_cols(lookup_model(settings["cross_correlation_model"]).args))
                if len(settings["cross_correlation_model"]) > 1:
                    multiple_models = True
                for model in settings["cross_correlation_model"]:
                    arguments = gen_cols(lookup_model(model).args)
                    all_model_arguments.append(arguments)
                    if model not in model_dictionary:
                        model_dictionary[model] = {'arguments': list(set().union(*arguments))}
            else:
                print("something went bad when assigning channels :(")

        
        model_cols = list(set().union(*all_model_arguments)) 
        if multiple_models:
            common_cols = ["File", "Position", "Channel", "Model", "Model selection test", "Model selection value", "Bleaching Half life / s", "Bleaching %", "Mean intensity / Hz", "ADF p-value", "KPSS p-value", "Stationary", "Valid fit", "Reason for rejection", "R^2", "rChi^2", 'c / nM', 'CPM / kHz', 'Volume / fL']
        else:
            common_cols = ["File", "Position", "Channel", "Model", "Bleaching Half life / s", "Bleaching %", "Mean intensity / Hz", "ADF p-value", "KPSS p-value", "Stationary", "Valid fit", "Reason for rejection", "R^2", "rChi^2", 'c / nM', 'CPM / kHz', 'Volume / fL']
        columns = common_cols + diffusion_cols + model_cols 
        best_fit_col_num = len(columns)

        #for model in range()
        df_fit = pd.DataFrame(columns=columns)

        # 15/4/21 
        #if multiple_models == True:
        for key, values in model_dictionary.items():
            columns = common_cols + diffusion_cols + model_cols # values['arguments']
            #print('columns:')
            #print(columns)
            if 'dataframe' not in model_dictionary[key]:
                model_dictionary[key]['dataframe'] = pd.DataFrame(columns=columns)
        
        if settings["save_individual_intensity_plots"] == "True":
            all_intensity_traces_figures = [] 
        
        average_adjustment = 0
        if settings['average_auto_correlation'] == 'True':
            average_auto_correlations = {}
            average_adjustment = 1 

        #if settings["save_individual_plots"] == "True":
        all_traces_figures = [] 
        #print("Creating empty figures...")
        for i in tqdm(range(positions + average_adjustment)):
            f = go.Figure()
            f.update_layout(template="plotly_white")
            f.update(layout_showlegend=True)
            f.update_xaxes(title_text='Log lag time τ / log10(τ)')
            f.update_yaxes(title_text='G(τ)')
            all_traces_figures.append(f)

        # Go through each entry in the file
        for i in tqdm(range(len(fcs['FcsData']['FcsEntry']))):
            position = fcs['FcsData']['FcsEntry'][i]['FcsDataSet']['Position']
            channel = fcs['FcsData']['FcsEntry'][i]['FcsDataSet']['Channel']
            channel = rename_channel(channel)
            try:
                if channel == 'Auto-correlation 1':
                    wavelength = fcs['FcsData']['FcsEntry'][i]['FcsDataSet']['Acquisition']['AcquisitionSettings']['BeamPath']['BeamPathSettings']['Attenuator'][0]['Wavelength']
                elif channel == 'Auto-correlation 2':
                    wavelength = fcs['FcsData']['FcsEntry'][i]['FcsDataSet']['Acquisition']['AcquisitionSettings']['BeamPath']['BeamPathSettings']['Attenuator'][1]['Wavelength']
                else:
                    wavelength = fcs['FcsData']['FcsEntry'][i]['FcsDataSet']['Acquisition']['AcquisitionSettings']['BeamPath']['BeamPathSettings']['Attenuator'][0]['Wavelength']
            except:
                print("Error: Only one wavelength saved")
                wavelength = fcs['FcsData']['FcsEntry'][i]['FcsDataSet']['Acquisition']['AcquisitionSettings']['BeamPath']['BeamPathSettings']['Attenuator'][0]['Wavelength']
            wavelength = unitFix(wavelength) # This gives a wavelength in m
            NA = settings['numerical_aperture']
            n = settings['refractive_index']
            # Calculate the shape parameters of the confocal volume
            if experimental_data == True:
                wxy = 0.61 * wavelength / NA
                wz = 2 * n * wavelength / (NA ** 2)
            else:
                wxy = float(settings['wxy_overide'])
                wz = float(settings['wz_overide'])
            
            #print("wxy = %s" %wxy)
            V_eff = 1E3 * 2**(3/2) * np.pi**1.5 * wxy**2 * wz # Effective volume of the confocal (observation) volume in litres as defined in https://www.picoquant.com/images/uploads/page/files/7351/appnote_quantfcs.pdf
            #print("V_eff = %s" %V_eff)
            # Convert to micrometres 
            wxy *= 1E6
            wz *= 1E6
            structural_parameter = wz / wxy
            beta = ( wxy / wz )**2

            models = []
            if channel == "Auto-correlation 1":
                for model in settings["auto_correlation_1_model"]:
                    models.append(lookup_model(model))
            elif channel == "Auto-correlation 2":
                for model in settings["auto_correlation_2_model"]:
                    models.append(lookup_model(model))
            elif channel == "Cross-correlation 1 versus 2" or channel == "Cross-correlation 2 versus 1":
                for model in settings["cross_correlation_model"]:
                    models.append(lookup_model(model))
            #print("Models = %s" %[model.args for model in models])
            
            count_rate_array = fcs['FcsData']['FcsEntry'][i]['FcsDataSet']['CountRateArray']
            decay_rate, fraction_bleached, mean_intensity = score_signal(count_rate_array)
            ADFStat, ADF_pvalue, KPSSStat, KPSS_pvalue, stationary = stationarity_test(count_rate_array)
            corr_array = fcs['FcsData']['FcsEntry'][i]['FcsDataSet']['CorrelationArray']
            # Slice the correlation array using the specified time domains
            corr_array = corr_array[settings['time_domain']['bottom_channel']:settings['time_domain']['top_channel']]
            # Minus 1 off correlation array to bring in line with standard correlation calculations
            corr_array[:,1] = [el-1 for el in corr_array[:,1]]
            # Correct for background
            if settings['background_correction'] == 'True':
                corr_array[:,1] = background_correction(corr_array[:,1], mean_intensity, background_signal)
            
            # Correct for photo-bleaching
            if channel == "Auto-correlation 1" or channel == "Auto-correlation 2":
                if settings['photobleaching_correction'] == 'True':
                    corr_array[:,1] *= photo_bleaching_correction(corr_array[:,0], decay_rate)

            if settings['average_auto_correlation'] == 'True':
                if channel in average_auto_correlations:
                    average_auto_correlations[channel]['auto_correlation'][:, 1] += corr_array[:, 1]
                    average_auto_correlations[channel]['number_of_auto_correlations'] += 1
                else:
                    average_auto_correlations[channel] = {'auto_correlation': corr_array, 'number_of_auto_correlations': 1}

            # Set the uncertainty on each point of the correlation curve
            sigma = None
            if settings["use_errors_in_fitting"] == "True":
                # Retrieve correlator settings for use in calculating errors
                max_time = fcs['FcsData']['FcsEntry'][i]['FcsDataSet']['Acquisition']['AcquisitionSettings']['MeasurementTime']
                max_time = unitFix(max_time)
                correlator_bin_time = fcs['FcsData']['FcsEntry'][i]['FcsDataSet']['Acquisition']['AcquisitionSettings']['CorrelatorBinning']
                #Change from microseconds to seconds
                correlator_bin_time = unitFix(correlator_bin_time)

                lowest_signal_intensity = 0
                # For cross-correlations use the intensity from the channel with the lowest signal 
                if channel == "Cross-correlation 2 versus 1":
                    if df_fit.loc[i-1]['Mean intensity / Hz'] > df_fit.loc[i-2]['Mean intensity / Hz']:
                        lowest_signal_intensity = float(df_fit.loc[i-1]['Mean intensity / Hz'])
                    else:
                        lowest_signal_intensity = df_fit.loc[i-2]['Mean intensity / Hz']
                    sigma = correlator_error(corr_array[:,0], lowest_signal_intensity, max_time, correlator_bin_time, beta, settings["diffusion_guess"], wxy, settings["molecular_brightness"], error_function=settings["correlator_error_function"])
                elif channel == "Cross-correlation 1 versus 2":
                    if df_fit.loc[i-2]['Mean intensity / Hz'] > df_fit.loc[i-3]['Mean intensity / Hz']:
                        lowest_signal_intensity = df_fit.loc[i-2]['Mean intensity / Hz']
                    else:
                        lowest_signal_intensity = df_fit.loc[i-3]['Mean intensity / Hz']
                    sigma = correlator_error(corr_array[:,0], lowest_signal_intensity, max_time, correlator_bin_time, beta, settings["diffusion_guess"], wxy, settings["molecular_brightness"], error_function=settings["correlator_error_function"])
                else:
                    sigma = correlator_error(corr_array[:,0], mean_intensity, max_time, correlator_bin_time, beta, settings["diffusion_guess"], wxy, settings["molecular_brightness"], error_function=settings["correlator_error_function"])
            
            #print("len(all_traces_figures) = %s" %len(all_traces_figures))
            if settings["save_individual_plots"] == "True":
                if sigma is not None:
                    #print("corr_array[:,0].shape = %s" %corr_array[:,0].shape)
                    all_traces_figures[position].add_trace(go.Scatter(x=np.log10(corr_array[:,0]), 
                                                                      y=corr_array[:,1] - sigma,
                                                                      mode='lines',
                                                                      legendgroup = "Group",
                                                                      line = {'width': 0, 'smoothing': 0.7},
                                                                      showlegend=False))

                    all_traces_figures[position].add_trace(go.Scatter(x=np.log10(corr_array[:,0]), 
                                                                      y=corr_array[:,1],
                                                                      mode='lines',
                                                                      legendgroup = "Group",
                                                                      showlegend=True,
                                                                      fillcolor = colour_picker(channel, 0.3),
                                                                      fill = 'tonexty',
                                                                      line = {'color': colour_picker(channel, 1), 'smoothing': 0.7},
                                                                      #name = str(channel)))
                                                                      name = fcs._filename[:-4] + ": " + str(position+1) + " " + str(channel)))

                    all_traces_figures[position].add_trace(go.Scatter(x=np.log10(corr_array[:,0]), 
                                                                      y=corr_array[:,1] + sigma,
                                                                      mode='lines',
                                                                      fillcolor = colour_picker(channel, 0.3),
                                                                      fill = 'tonexty',
                                                                      line = {'width': 0, 'smoothing': 0.7},
                                                                      legendgroup = "Group",
                                                                      showlegend=False))
                else:
                    all_traces_figures[position].add_trace(go.Scatter(x=np.log10(corr_array[:,0]), 
                                                                      y=corr_array[:,1],
                                                                      mode='lines',
                                                                      legendgroup = "Group",
                                                                      showlegend=True,
                                                                      line = {'color': colour_picker(channel, 1), 'smoothing': 0.7},
                                                                      name = str(channel)))
                                                                      #name = fcs._filename[:-4] + ": " + str(position+1) + " " + str(channel)))
            
            if settings["save_individual_intensity_plots"] == "True":
                if fcs['FcsData']['FcsEntry'][i]['FcsDataSet']['CountRateArray'].shape[0] != 0:
                    intensity_fig = go.Figure()
                    intensity_fig.update_layout(template="plotly_white")
                    intensity_fig.update(layout_showlegend=True)
                    intensity_fig.update_xaxes(title_text='Time / s')
                    intensity_fig.update_yaxes(title_text='Intensity / kHz')
                    intensity_fig.add_trace(go.Scatter(x=fcs['FcsData']['FcsEntry'][i]['FcsDataSet']['CountRateArray'][:, 0], 
                                                       y=fcs['FcsData']['FcsEntry'][i]['FcsDataSet']['CountRateArray'][:, 1],
                                                       mode='lines',
                                                       showlegend=True,
                                                       name= fcs._filename[:-4] + ": " + str(position+1) + " " + str(channel)))
                    all_intensity_traces_figures.append(intensity_fig)

            if combined_traces_figures.get(channel, False) is False:
                f = go.Figure()
                f.update_layout(template="plotly_white", title=channel)
                f.update(layout_showlegend=False)
                f.update_xaxes(title_text='Log lag time τ / log10(τ)')
                f.update_yaxes(title_text='G(τ)')
                combined_traces_figures[channel] = f
            
            if sigma is not None:
                combined_traces_figures[channel].add_trace(go.Scatter(x=np.log10(corr_array[:,0]), 
                                                                      y=corr_array[:,1] - sigma,
                                                                      mode='lines',
                                                                      line = {'width': 0, 'smoothing': 0.7},
                                                                      legendgroup= fcs._filename[:-4] + ": " + str(position+1),
                                                                      showlegend=False))

                combined_traces_figures[channel].add_trace(go.Scatter(x=np.log10(corr_array[:,0]), 
                                                                      y=corr_array[:,1],
                                                                      mode='lines',
                                                                      showlegend=True,
                                                                      fillcolor = 'rgba(128,0,128, 0.3)',
                                                                      fill = 'tonexty',
                                                                      line = {'color': 'rgba(128,0,128, 1)', 'smoothing': 0.7},
                                                                      legendgroup=fcs._filename[:-4] + ": " + str(position+1),
                                                                      name=fcs._filename[:-4] + ": " + str(position+1)))
                                                                      #name=str(position+1)))

                combined_traces_figures[channel].add_trace(go.Scatter(x=np.log10(corr_array[:,0]), 
                                                                      y=corr_array[:,1] + sigma,
                                                                      mode='lines',
                                                                      fillcolor = 'rgba(128,0,128, 0.3)',
                                                                      fill = 'tonexty',
                                                                      line = {'width': 0, 'smoothing': 0.7},
                                                                      legendgroup = fcs._filename[:-4] + ": " + str(position+1),
                                                                      showlegend=False))
            else:
                combined_traces_figures[channel].add_trace(go.Scatter(x=np.log10(corr_array[:,0]), 
                                                                      y=corr_array[:,1],
                                                                      mode='lines',
                                                                      showlegend=True,
                                                                      line = {'color': 'rgba(128,0,128, 1)', 'smoothing': 0.7},
                                                                      legendgroup=fcs._filename[:-4] + ": " + str(position+1),
                                                                      name=fcs._filename[:-4] + ": " + str(position+1))) 
                                                                      #name=str(position+1))) 

            use_bounds = False
            if settings['use_bounds'] == 'True':
                use_bounds = True
            
            reject_fits_equal_to_bounds = False
            if settings['reject_fits_equal_to_bounds'] == "True":
                print('reject_fits_equal_to_bounds is true')
                reject_fits_equal_to_bounds = True
            
            fits, fit_params, fit_errors, sum_squared_errors, Rsquareds, valid_fits, reason_for_rejections, chi2s = [], [], [], [], [], [], [], []
            #print("valid_fits = %s" %valid_fits)

            for model in models:
                fit, parameters, errors, sse, Rsquared, valid_fit, reason_for_rejection, chi2 = fit_curve(corr_array[:, 0], corr_array[:, 1], model, sigma, structural_parameter, use_bounds=use_bounds, reject_fits_equal_to_bounds=reject_fits_equal_to_bounds)
                fits.append(fit)
                fit_params.append(parameters) 
                fit_errors.append(errors)
                sum_squared_errors.append(sse)
                Rsquareds.append(Rsquared)
                valid_fits.append(valid_fit)
                reason_for_rejections.append(reason_for_rejection)
                chi2s.append(chi2)
            # end of model for loop

            # Perform model selection, returns sorted list of indices to sort by along with the selected test score
            if multiple_models == True:
                # Remove models that produced invalid fits
                # Move None values to the end of the lists
                # Also if all fits were invalid then don't perform model
                good_indices = []
                bad_indices = []
                valid_models = []
                valid_sum_squared_errors = [] #len(models) > 1
                for m in range(len(models)):
                    if valid_fits[m] == True:
                        good_indices.append(m)
                        valid_models.append(models[m])
                        valid_sum_squared_errors.append(sum_squared_errors[m])
                    else:
                        bad_indices.append(m)
                indices, selection_scores = model_selection(good_indices, len(corr_array[:, 0]), valid_sum_squared_errors, valid_models, test=settings['model_selection_test'])
                indices.extend(bad_indices) # Add bad indices to the end
                #print("indices = %s" %indices)
                # Sort model and output of curve fitting by 
                models = reindex(models, indices)
                fits = reindex(fits, indices)
                fit_params = reindex(fit_params, indices)
                fit_errors = reindex(fit_errors, indices)
                sum_squared_errors = reindex(sum_squared_errors, indices)
                Rsquareds = reindex(Rsquareds, indices)
                #print("valid_fits = %s" %valid_fits)
                valid_fits = reindex(valid_fits, indices)
                reason_for_rejections = reindex(reason_for_rejections, indices)
                #print("valid_fits = %s" %valid_fits)
                chi2s = reindex(chi2s, indices)
                #print("Models = %s, scores = %s" %(models, selection_scores))

            # Save each of the model fits as a csv into a separate model folder alongside the data for replotting and other purposes
            if settings['save_curves'] == "True":
                fits_directory = os.path.join(analysis_directory, 'Fits')
                if not os.path.exists(fits_directory):
                    os.mkdir(fits_directory)

                for m in range(len(models)):
                    # Create a new directory for each of the fits
                    #correlation_curve_file_directory = fits_directory + models[m].name + '\\'
                    correlation_curve_file_directory = os.path.join(fits_directory, models[m].name)
                    if not os.path.exists(correlation_curve_file_directory):
                        os.mkdir(correlation_curve_file_directory)

                    correlation_curve_file_name = fcs._filename[:-4] + '_' + str(position+1) + '_' + str(channel) + '.csv'
                    correlation_curve_file_name = correlation_curve_file_directory + correlation_curve_file_name #correlation_curve_file_directory + correlation_curve_file_name.replace(" ", "_")
                    correlation_curve_file_name = os.path.expanduser(correlation_curve_file_name)

                    sigma_out = sigma
                    if sigma is None:
                        sigma_out = np.zeros(len(corr_array[:,0]))

                    try:
                        if fits[m] is None:
                            # If the fit failed then only save the data
                            np.savetxt(correlation_curve_file_name, np.array([corr_array[:,0], corr_array[:,1], sigma_out]).transpose(), fmt='%.10f', delimiter=',', newline='\n', header='Lag time, Correlation, Std Deviation')
                        else:
                            np.savetxt(correlation_curve_file_name, np.array([corr_array[:,0], corr_array[:,1], sigma_out, fits[m]]).transpose(), fmt='%.10f', delimiter=',', newline='\n', header='Lag time, Correlation, Std Deviation, Fit')

                    except Exception as e:
                        print('ERROR: Saving of fits to csv files could not be completed due to %s' %e)

            model_fits = []
            D, D_error = 0, 0
            #print("valid_fits = %s" %valid_fits)
            if valid_fits[0] == True:
                #HANNAH added multiple conditions here for the new models
                #print("models[0].name = %s" %models[0].name)

                '''if (models[0].name == "anomalous_diffusion" or models[0].name == "anomalous_diffusion_1D"
                      or models[0].name == "anomalous_diffusion_2D" or models[0].name == "anomalous_diffusion_1D_triplet"
                      or models[0].name == "anomalous_diffusion_2D_triplet"):'''
                if (models[0].name.find("anomalous_diffusion") != -1):
                    t_D_ind = models[0].args.index('tau_D')# Find the index relating to tau_D parameter,
                    alpha_ind = models[0].args.index('alpha') #- 1
                    D, D_error = calc_diffusion(fit_params[0][t_D_ind], fit_errors[0][t_D_ind], alpha=fit_params[0][alpha_ind], alpha_error=fit_errors[0][alpha_ind], r=wxy)
                elif models[0].name == "line":
                    D, D_error = 0, 0 
                else:
                    t_D_ind = models[0].args.index('tau_D')# Find the index relating to tau_D parameter,
                    D, D_error = calc_diffusion(fit_params[0][t_D_ind], fit_errors[0][t_D_ind], r=wxy)
                model_fits = []
                for j in range(len(fit_params[0])):
                    model_fits.append(fit_params[0][j])
                    model_fits.append(fit_errors[0][j])
                if settings["save_individual_plots"] == "True":
                    all_traces_figures[position].add_trace(go.Scatter(x=np.log10(corr_array[:,0]), 
                                                                      y=fits[0],
                                                                      mode='lines',
                                                                      legendgroup = fcs._filename[:-4] + ": " + str(position+1) + " " + str(channel),
                                                                      name=str(channel) + " fit"))
                                                                      #name=fcs._filename[:-4] + ": " + str(position+1) + " " + str(channel) + " fit"))
                combined_traces_figures[channel].add_trace(go.Scatter(x=np.log10(corr_array[:,0]), 
                                                                      y=fits[0],
                                                                      mode='lines',
                                                                      #legendgroup=str(position+1),
                                                                      name=fcs._filename[:-4] + ": " + str(position+1) + " " + str(channel) + " fit"))
                                                                      #name=str(channel) + " fit"))

            df_fit.loc[i] = [""] * best_fit_col_num # Create empty columns
            #["File", "Position", "Channel", "Model", "Model selection test", "Model selection value", "Bleaching %", "Mean intensity / Hz", "Valid fit", "R^2", "rChi^2"]
            df_fit.loc[i]['File'] = fcs._filename[:-4]
            df_fit.loc[i]['Position'] = position + 1
            df_fit.loc[i]['Model'] = models[0].name
            if multiple_models == True and valid_fits[0] == True:
                df_fit.loc[i]["Model selection test"]  = settings["model_selection_test"]
                df_fit.loc[i]["Model selection value"] = selection_scores[0]
            df_fit.loc[i]['Channel'] = channel
            if decay_rate != "":
                df_fit.loc[i]['Bleaching Half life / s'] = np.log(2) / decay_rate
            df_fit.loc[i]['Bleaching %'] = fraction_bleached*100
            df_fit.loc[i]['Mean intensity / Hz'] = mean_intensity
            df_fit.loc[i]['Volume / fL'] = V_eff * 1E15
            df_fit.loc[i]['ADF p-value'] = ADF_pvalue
            df_fit.loc[i]['KPSS p-value'] = KPSS_pvalue
            df_fit.loc[i]['Stationary'] = "Yes" if stationary else "No"
            df_fit.loc[i]['Valid fit'] = "Yes" if valid_fits[0] else "No" #valid_fit
            df_fit.loc[i]['Reason for rejection'] = reason_for_rejections[0]
            df_fit.loc[i]['R^2'] =  Rsquareds[0]
            df_fit.loc[i]['rChi^2'] =  chi2s[0]
            df_fit.loc[i]['D'] = D
            df_fit.loc[i]['D std'] = D_error
            try: 
                # Using a try here for when repeat measurements exist as the last entry never fits and therefore doesn't provide an N value
                df_fit.loc[i]['c / nM'] = 1E9 * fit_params[0][0] / (6.02214076E23 * V_eff) # Calculate the molar concentration in nM
                if mean_intensity != '': 
                    df_fit.loc[i]['CPM / kHz'] = fit_params[0][0] * 1000 / mean_intensity
            except:
                print("Error: Could not calculate concentration and CPM due to invalid fit or otherwise missing N value. This might be due to repeated measurements.")
            
            if valid_fit:
                for p in range(len(fit_params[0])):
                    df_fit.loc[i][models[0].args[p]] = fit_params[0][p]
                    df_fit.loc[i][models[0].args[p] + " std"] = fit_errors[0][p]

            # 15/4/21
            # Save data to individual dataframe for each model
            try:
                if multiple_models == True:
                    for m in range(len(models)):
                        row_num = model_dictionary[models[m].name]['dataframe'].shape[0] + 1
                        model_dictionary[models[m].name]['dataframe'].loc[row_num] = [""] * len(model_dictionary[models[m].name]['dataframe'].columns) # Create empty columns
                        model_dictionary[models[m].name]['dataframe'].loc[row_num]['File'] = fcs._filename[:-4]
                        model_dictionary[models[m].name]['dataframe'].loc[row_num]['Position'] = position + 1
                        model_dictionary[models[m].name]['dataframe'].loc[row_num]['Model'] = models[m].name
                        if valid_fits[m] == True:
                            model_dictionary[models[m].name]['dataframe'].loc[row_num]["Model selection test"]  = settings["model_selection_test"]
                            model_dictionary[models[m].name]['dataframe'].loc[row_num]["Model selection value"] = selection_scores[m]
                        model_dictionary[models[m].name]['dataframe'].loc[row_num]['Channel'] = channel
                        if decay_rate != "":
                            model_dictionary[models[m].name]['dataframe'].loc[row_num]['Bleaching Half life / s'] = np.log(2) / decay_rate
                        model_dictionary[models[m].name]['dataframe'].loc[row_num]['Bleaching %'] = fraction_bleached*100
                        model_dictionary[models[m].name]['dataframe'].loc[row_num]['Mean intensity / Hz'] = mean_intensity
                        model_dictionary[models[m].name]['dataframe'].loc[row_num]['Volume / fL'] = V_eff * 1E15
                        model_dictionary[models[m].name]['dataframe'].loc[row_num]['ADF p-value'] = ADF_pvalue
                        model_dictionary[models[m].name]['dataframe'].loc[row_num]['KPSS p-value'] = KPSS_pvalue
                        model_dictionary[models[m].name]['dataframe'].loc[row_num]['Stationary'] = "Yes" if stationary else "No"
                        model_dictionary[models[m].name]['dataframe'].loc[row_num]['Valid fit'] = "Yes" if valid_fits[m] else "No" #valid_fit
                        model_dictionary[models[m].name]['dataframe'].loc[row_num]['Reason for rejection'] = reason_for_rejections[m]
                        model_dictionary[models[m].name]['dataframe'].loc[row_num]['R^2'] =  Rsquareds[m]
                        model_dictionary[models[m].name]['dataframe'].loc[row_num]['rChi^2'] =  chi2s[m]
                        model_dictionary[models[m].name]['dataframe'].loc[row_num]['D'] = D
                        model_dictionary[models[m].name]['dataframe'].loc[row_num]['D std'] = D_error
                        try: 
                            # Using a try here for when repeat measurements exist as the last entry never fits and therefore doesn't provide an N value
                            model_dictionary[models[m].name]['dataframe'].loc[row_num]['c / nM'] = 1E9 * fit_params[m][0] / (6.02214076E23 * V_eff) # Calculate the molar concentration in nM
                            if mean_intensity != '': 
                                model_dictionary[models[m].name]['dataframe'].loc[row_num]['CPM / kHz'] = fit_params[m][0] * 1000 / mean_intensity
                        except:
                            print("Error: Could not calculate concentration and CPM due to invalid fit or otherwise missing N value. This might be due to repeated measurements.")

                        # PROBLEM HERE
                        if valid_fits[m] == True:

                            for p in range(len(fit_params[m])):
                                model_dictionary[models[m].name]['dataframe'].loc[row_num][models[m].args[p]] = fit_params[m][p]
                                model_dictionary[models[m].name]['dataframe'].loc[row_num][models[m].args[p] + " std"] = fit_errors[m][p]

                # End of loop for individual analysis of each entry in the file
            except Exception as e:
                print('Error: {:}'.format(e))
        if settings['average_auto_correlation'] == 'True':
            for key in average_auto_correlations.keys():
                average_auto_correlations[key]['auto_correlation'][:, 1] /= average_auto_correlations[key]['number_of_auto_correlations']
                all_traces_figures[-1].add_trace(go.Scatter(x=np.log10(average_auto_correlations[key]['auto_correlation'][:, 0]), 
                                                            y=average_auto_correlations[key]['auto_correlation'][:, 1],
                                                            mode='lines',
                                                            name=fcs._filename[:-4] + '_' + str(key) + '_average'))
                combined_traces_figures[channel].add_trace(go.Scatter(x=np.log10(average_auto_correlations[key]['auto_correlation'][:, 0]), 
                                                                      y=average_auto_correlations[key]['auto_correlation'][:, 1],
                                                                      mode='lines',
                                                                      name=fcs._filename[:-4] + '_' + str(key) + '_average'))

        for i in range(len(all_traces_figures)):
            if settings['save_individual_plots'] == 'True':
                if settings['average_auto_correlation'] == True and i == len(all_intensity_traces_figures) - 1:
                    image_file_path = os.path.join(image_directory, fcs._filename[:-4] + "_pos_" + str(i+1) + "_all_traces" + ".png") 
                else:
                    image_file_path = os.path.join(image_directory, fcs._filename[:-4] + "_average" + ".png")
                all_traces_figures[i].write_image(image_file_path)
            if settings['show_individual_plots'] == 'True':
                all_traces_figures[i].show()
        
        if settings['save_individual_intensity_plots'] == 'True':
            for i in range(len(all_intensity_traces_figures)):
                image_file_path = os.path.join(image_directory, fcs._filename[:-4] + "_pos_" + str(i+1) + "_intensity" + ".png") 
                all_intensity_traces_figures[i].write_image(image_file_path)

        if df_best_fit.empty == True:
            df_best_fit = df_fit
        else:
            df_best_fit = pd.concat([df_best_fit, df_fit])

        '''
        Analyse cross-correlation data if it exists

        Following the approach of Sadaie et al 2014 doi:10.1128/mcb.00087-14
        '''
        
        df_cross = pd.DataFrame(columns=['File', 'Position', 'AC1 Conc / nM', 'AC2 Conc / nM', 'CC21 Conc / nM', 'CC12 Conc / nM', 'Selected CC Conc / nM', 'Free AC1 Conc / nM', 'Free AC2 Conc /nM', 'Bound fraction AC1', 'Bound fraction AC2', 'Kd / nM'])
        
        def Kd_func(x, Kd):
            return x / (Kd + x)
        
        def line_func(x, m, c):
            return m * x + c

        def fit_func(x, y, f):
            fit_params, cov_mat = curve_fit(f, x, y)
            fit_errors = np.sqrt(np.diag(cov_mat))
            fit_residual = y - f(x, *fit_params)
            fit_Rsquared = 1 - np.var(fit_residual) / np.var(y)
            fit = f(x, *fit_params)
            #chi2 = (1/(len(y)-1)) * np.sum(((y - fit)/(sigma))**2)
            return fit, fit_Rsquared, fit_params, fit_errors
        
        if interactions == True:            
            x_As = []
            x_Bs = []
            y_As = []
            y_Bs = []
            #Complexes = []
            #FreexFrees = []
            for i in range(0, positions*channels, channels):
                try:
                    conc_conversion_factor = 1E9 / (6.02214076E23 * V_eff)

                    N_AC1 = df_fit.loc[i]["N"]
                    N_AC2 = df_fit.loc[i+1]["N"]
                    N_CC21 = df_fit.loc[i+2]["N"]
                    N_CC12 = df_fit.loc[i+3]["N"]

                    if (df_fit.loc[i]['Valid fit'] == False or
                        df_fit.loc[i+1]['Valid fit'] == False or
                        df_fit.loc[i+2]['Valid fit'] == False or
                        df_fit.loc[i+3]['Valid fit'] == False):
                        raise Exception('Cannot use an invalid fit')

                    # Calculate concentrations
                    C_AC1 = N_AC1 * conc_conversion_factor
                    C_AC2 = N_AC2 * conc_conversion_factor
                    C_CC21 = N_CC21 * N_AC1 * N_AC2 * conc_conversion_factor #N_CC21 * conc_conversion_factor
                    C_CC12 = N_CC12 * N_AC1 * N_AC2 * conc_conversion_factor #N_CC12 * conc_conversion_factor

                    if df_fit.loc[i]["Mean intensity / Hz"] > df_fit.loc[i+1]["Mean intensity / Hz"]:
                        # Intensity of Channel 1 is higher therefore select Channel 1 vs 2 cross-correlation
                        selected_cc = C_CC12
                    else:
                        selected_cc = C_CC21
                    
                    free_1 = C_AC1 - selected_cc
                    free_2 = C_AC2 - selected_cc
                    bound_fraction_1 = selected_cc / C_AC1
                    bound_fraction_2 = selected_cc / C_AC2

                    # From equation 2 of Sadaie et al 2014 (DOI: 10.1128/MCB.00087-14)
                    Kd = free_1 * free_2 / selected_cc

                    x_A = C_AC1 - selected_cc
                    y_A = selected_cc / C_AC2
                    if x_A > 0 and y_A > 0 and y_A < 1:
                        x_As.append(x_A)
                        y_As.append(y_A)

                    x_B = C_AC2 - selected_cc
                    y_B = selected_cc / C_AC1
                    if x_B > 0 and y_B > 0 and y_B < 1:
                        x_Bs.append(x_B)
                        y_Bs.append(y_B)

                    # From Ankers et al 2016 (doi: 10.7554/eLife.10473)
                    '''if free_1 and free_2 > 0 and selected_cc > 0:
                        Complexes.append(selected_cc)
                        FreexFrees.append(free_1 * free_2)'''

                    df_cross.loc[i] = [fcs._filename[:-4], i/4 + 1, C_AC1, C_AC2, C_CC21, C_CC12, selected_cc, free_1, free_2, bound_fraction_1, bound_fraction_2, Kd]

                except Exception as error:
                    print("ERROR Couldn't fit: %s" %error)
                    print('Could not analyse cross-correlation for position %s' %int(i/4 + 1))
                    df_cross.loc[i] = [fcs._filename[:-4], i/4 + 1, "", "", "", "", "", "", "", "", "", ""]

            interaction_data['x_A'] = interaction_data['x_A'] + x_As
            interaction_data['x_B'] = interaction_data['x_B'] + x_Bs
            interaction_data['y_A'] = interaction_data['y_A'] + y_As
            interaction_data['y_B'] = interaction_data['y_B'] + y_Bs
            #interaction_data['Complex'] = interaction_data['Complex'] + Complexes
            #interaction_data['Free x Free'] = interaction_data['Free x Free'] + FreexFrees

            '''
            Calculate the Kd on the file data set 
            '''
            print("x_As: {:}".format(x_As))
            if len(x_As) > 0 or len(x_Bs) > 0:
                fig = go.Figure()
                fig.update_layout(yaxis=dict(range=[0,1]))
                fig.update_layout(
                    xaxis_title="[Species A or B]-[Complex] / nM",
                    yaxis_title="[Complex]/[Species A or B]",
                    font=dict(
                        size=18,
                        color="black"
                    )
                )

                interaction_data['file'].append(fcs._filename[:-4])
                interaction_data['data_points'].append(len(x_As))                
                
                if len(x_As) > 0:
                    Kd_A_y_fit, Kd_A_fit_Rsquared, Kd_A_fit_params, Kd_A_fit_errors = fit_func(np.array(x_As), np.array(y_As), Kd_func)
                    interaction_data['Kd_A'].append(Kd_A_fit_params[0])
                    interaction_data['Kd_A_std'].append(Kd_A_fit_errors[0])
                    fig.add_trace(go.Scatter(x=x_As, y=y_As, mode='markers', name=fcs._filename[:-4] + ': [Comp]/[AC2]', line_color='red'))
                    x_ = np.arange(0, np.sort(x_As)[-1])
                    fig.add_trace(go.Scatter(x=x_, y=Kd_func(x_, *Kd_A_fit_params), mode='lines', name=fcs._filename[:-4] + ': Kd fit {:0.2f} +/- {:0.2f} nM'.format(Kd_A_fit_params[0], Kd_A_fit_errors[0]), line_color='red'))
                    

                if len(x_Bs) > 0:
                    Kd_B_y_fit, Kd_B_fit_Rsquared, Kd_B_fit_params, Kd_B_fit_errors = fit_func(np.array(x_Bs), np.array(y_Bs), Kd_func)
                    interaction_data['Kd_B'].append(Kd_B_fit_params[0])
                    interaction_data['Kd_B_std'].append(Kd_A_fit_errors[0])    
                    fig.add_trace(go.Scatter(x=x_Bs, y=y_Bs, mode='markers', name=fcs._filename[:-4] + ': [Comp]/[AC1]', line_color='green'))
                    x_ = np.arange(0, np.sort(x_Bs)[-1])
                    fig.add_trace(go.Scatter(x=x_, y=Kd_func(x_, *Kd_B_fit_params), mode='lines', name=fcs._filename[:-4] + ': Kd fit {:0.2f} +/- {:0.2f} nM'.format(Kd_B_fit_params[0], Kd_B_fit_errors[0]), line_color='green'))               

                '''fig_line = go.Figure()
                #fig_line.update_layout(yaxis=dict(range=[0,1]))
                fig_line.update_layout(
                    xaxis_title="[Complex] / nM",
                    yaxis_title="[Species A]x[Species B] / (nM)^2",
                    font=dict(
                        size=18,
                        color="black"
                    )
                )

                if len(Complexes) > 0:
                    x_ = np.arange(0, np.sort(Complexes)[-1])
                    Kd_line_fit, Kd_line_fit_Rsquared, Kd_line_fit_params, Kd_line_fit_errors = fit_func(np.array(Complexes), np.array(FreexFrees), line_func)
                    interaction_data['Kd_line'].append(Kd_line_fit_params[0])
                    interaction_data['Kd_line_std'].append(Kd_line_fit_errors[0])
                    fig_line.add_trace(go.Scatter(x=Complexes, y=FreexFrees, mode='markers', name=fcs._filename[:-4] + ': Data', line_color='purple'))
                    x_ = np.arange(0, np.sort(Complexes)[-1])
                    fig_line.add_trace(go.Scatter(x=x_, y=line_func(x_, *Kd_line_fit_params), mode='lines', name=fcs._filename[:-4] + ': Kd fit {:0.2f} +/- {:0.2f} nM'.format(Kd_line_fit_params[0], Kd_line_fit_errors[0]), line_color='purple'))'''   

                if settings["show_individual_Kd_plots"] == 'True':
                    fig.show()
                    #fig_line.show()

                if settings["save_individual_Kd_plots"] == 'True':
                    fig.write_image(os.path.join(image_directory, fcs._filename[:-4] + '_Kd.png'))
                    #fig_line.write_image(os.path.join(image_directory, fcs._filename[:-4] + '_line_Kd.png'))

            # Add the cross-correlation data to the overall dataframe
            if df_best_cross.empty == True:
                df_best_cross = df_cross
            else:
                df_best_cross = pd.concat([df_best_cross, df_cross])
        
        '''
        Save the analysis for each file if desired
        '''
        if settings['individual_sheets'] == 'True':
            mode = "a"
            if not os.path.exists(excel_file):
                mode = "w"
            with pd.ExcelWriter(excel_file, engine="openpyxl", mode=mode) as writer:
                df_fit.to_excel(writer, sheet_name=fcs._filename[:-4], index=False)
                if channels == 4:
                    df_cross.to_excel(writer, sheet_name=fcs._filename[:-4] + " cross-correlation", index=False)

        # End of loop of analysis for each file

    for key, value in combined_traces_figures.items():
        print("Saving %s" %(key))
        image_file_path = os.path.join(image_directory, key + ".png") 
        if settings["save_plots"] == "True":
            value.write_image(image_file_path)
        value.update(layout_showlegend=True)
        value.show()

    mode = "a"
    if not os.path.exists(excel_file):
        mode = "w"
    with pd.ExcelWriter(excel_file, engine="openpyxl", mode=mode) as writer:
        df_best_fit.to_excel(writer, sheet_name='Best fits', index=False)
        if settings["save_only_best_fits"] == "False":
            for model_name, value in model_dictionary.items():
                value['dataframe'].to_excel(writer, sheet_name=model_name, index=False)

        if interactions==True:
            df_best_cross.to_excel(writer, sheet_name = 'Interactions', index=False)
            
            summary_Kd_plot = go.Figure()
            summary_Kd_plot.update_layout(yaxis=dict(range=[0,1]))
            summary_Kd_plot.update_layout(xaxis_title="[Species A or B]-[Complex] / nM",
                                          yaxis_title="[Complex]/[Species A or B]",
                                          font=dict(
                                              size=18,
                                              color="black"
                                          )
                                        ) 

            #df_summary_interactions = pd.DataFrame(columns=['Description', 'Kd A', 'Kd A std', 'Kd B', 'Kd B std', 'Kd line', 'Kd line std', 'Data points'])
            df_summary_interactions = pd.DataFrame(columns=['Description', 'Kd A', 'Kd A std', 'Kd B', 'Kd B std', 'Data points'])
            df_summary_interactions['Description'] = pd.Series(['Summary'] + interaction_data['file'])
            max_value_A = 0
            max_value_B = 0
            if len(interaction_data['x_A']) > 0:
                Kd_A_y_fit, Kd_A_fit_Rsquared, Kd_A_fit_params, Kd_A_fit_errors = fit_func(np.array(interaction_data['x_A']), np.array(interaction_data['y_A']), Kd_func)
                df_summary_interactions['Kd A'] = pd.Series([Kd_A_fit_params[0]] + interaction_data['Kd_A'])
                df_summary_interactions['Kd A std'] = pd.Series([Kd_A_fit_errors[0]] + interaction_data['Kd_A_std'])
                df_summary_interactions['Data points'] = pd.Series([len(interaction_data['x_A'])] + interaction_data['data_points'])
                max_value_A = np.sort(interaction_data['x_A'])[-1]
            
            if len(interaction_data['x_B']) > 0:
                Kd_B_y_fit, Kd_B_fit_Rsquared, Kd_B_fit_params, Kd_B_fit_errors = fit_func(np.array(interaction_data['x_B']), np.array(interaction_data['y_B']), Kd_func)
                df_summary_interactions['Kd B'] = pd.Series([Kd_B_fit_params[0]] + interaction_data['Kd_B'] )
                df_summary_interactions['Kd B std'] = pd.Series([Kd_B_fit_errors[0]] + interaction_data['Kd_B_std'])
                df_summary_interactions['Data points'] = pd.Series([len(interaction_data['x_B'])] + interaction_data['data_points'])
                max_value_B = np.sort(interaction_data['x_B'])[-1]
            
            max_value = max_value_A if max_value_A > max_value_B else max_value_B
            x_ = np.arange(0, max_value)
            #x_ = np.arange(0, np.sort(interaction_data['Complex'])[-1])
            summary_Kd_plot.add_trace(go.Scatter(x=interaction_data['x_A'], y=interaction_data['y_A'], mode='markers', name='[Comp]/[AC2]', line_color='red'))
            summary_Kd_plot.add_trace(go.Scatter(x=x_, y=Kd_func(x_, *Kd_A_fit_params), mode='lines', name='Kd fit {:0.2f} +/- {:0.2f} nM'.format(Kd_A_fit_params[0], Kd_A_fit_errors[0]), line_color='red'))
            summary_Kd_plot.add_trace(go.Scatter(x=interaction_data['x_B'], y=interaction_data['y_B'], mode='markers', name='[Comp]/[AC1]', line_color='green'))
            summary_Kd_plot.add_trace(go.Scatter(x=x_, y=Kd_func(x_, *Kd_B_fit_params), mode='lines', name='Kd fit {:0.2f} +/- {:0.2f} nM'.format(Kd_B_fit_params[0], Kd_B_fit_errors[0]), line_color='green'))
            
            '''summary_Kd_line_plot = go.Figure()
            summary_Kd_line_plot.update_layout(xaxis_title="[Complex] / nM",
                                               yaxis_title="[Species A]x[Species B] / (nM)^2",
                                               font=dict(
                                                   size=18,
                                                   color="black"
                                               )
                                            )
            
            if len(interaction_data['Complex']) > 0:
                Kd_line_fit, Kd_line_fit_Rsquared, Kd_line_fit_params, Kd_line_fit_errors = fit_func(np.array(interaction_data['Complex']), np.array(interaction_data['Free x Free']), line_func)
                df_summary_interactions['Kd line'] = pd.Series([Kd_line_fit_params[0]] + interaction_data['Kd_line'] )
                df_summary_interactions['Kd line std'] = pd.Series([Kd_line_fit_errors[0]] + interaction_data['Kd_line_std'])

                x_ = np.arange(0, np.sort(interaction_data['Complex'])[-1])
                summary_Kd_line_plot.add_trace(go.Scatter(x=interaction_data['Complex'], y=interaction_data['Free x Free'], mode='markers', name='Data', line_color='red'))
                summary_Kd_line_plot.add_trace(go.Scatter(x=x_, y=line_func(x_, *Kd_line_fit_params), mode='lines', name='Kd fit {:0.2f} +/- {:0.2f} nM'.format(Kd_line_fit_params[0], Kd_line_fit_errors[0]), line_color='red'))'''

            df_summary_interactions.to_excel(writer, sheet_name='Interaction Summary', index=False)

            if settings["show_interactive_summary_Kd_plots"] == 'True':
                summary_Kd_plot.show()
                #summary_Kd_line_plot.show()
            
            if settings["save_summary_Kd_plots"] == 'True':
                summary_Kd_plot.write_image(os.path.join(image_directory, 'Summary_Kd.png'))
                #summary_Kd_line_plot.write_image(os.path.join(image_directory, 'Summary_line_Kd.png'))

    return

def main():
    settings_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "settings.json") 
    settings = dict()
    with open(settings_path, encoding='utf-8') as settings_file:
        settings = json.loads(settings_file.read())

    '''
    Load data folder using tkinter file dialog. Last opened folder is loaded if possible.
    '''
    print("Loaded settings")
    tkinter.Tk().withdraw() # close the root window
    init_dir = settings['recent_folder']
    directory = ""
    try:
        directory = filedialog.askdirectory(initialdir=init_dir)
    except:
        print("Error: Could not open recently accessed folder")
        directory = filedialog.askdirectory()
    print("selected folder: %s" %directory)
    # Store folder as recently accessed
    #with open(dir_path + "/" + file_name + ".json", "w+") as f:
    with open(settings_path, "w+") as settings_file:
        settings['recent_folder'] = directory
        json.dump(settings, settings_file, indent=4)

    #print(settings)
    fcs_files = []
    for filename in os.listdir(directory):
        if filename.endswith(".fcs"): 
            filename = os.path.join(directory, filename)
            print(filename)
            #fcs_files.append(ConfoCor3Fcs(filename))
            fcs_files.append(filename)

        elif filename.endswith(".sim"):
            #filename_path = os.path.join(directory, filename)
            filename = os.path.join(directory, filename)
            print(filename) 
            #fcs_files.append(SimulatedFCS(filename))
            fcs_files.append(filename)

    '''
        Sort FCS files by name
    '''
    print("%s FCS files found" %len(fcs_files))
    output_name = input("Please enter a name for the files: ")
    
    analysis_directory = os.path.join(directory, "Analysis") 
    if output_name == "":
        image_directory = os.path.join(directory, "Images")
        excel_file_path = os.path.join(analysis_directory, "results.xlsx")
        copy_of_settings_path = os.path.join(analysis_directory, "settings.txt")
    else: 
        image_directory = os.path.join(directory, "Images" + '_' + str(output_name))
        analysis_directory = os.path.join(directory, "Analysis" + '_' + str(output_name)) 
        excel_file_path = os.path.join(analysis_directory, "results" + '_' + str(output_name) + ".xlsx") 
        copy_of_settings_path = os.path.join(analysis_directory, "settings" + '_' + str(output_name) + ".txt")
    
    if not os.path.exists(image_directory):
        os.mkdir(image_directory)

    # Excel sheet for analysis results 
    if not os.path.exists(analysis_directory):
        os.mkdir(analysis_directory)

    # Make a copy of the settings and save to the analysis directory
    with open(copy_of_settings_path, "w+") as settings_file:
        json.dump(settings, settings_file, indent=4)

    
    analyse_fcs_files(fcs_files, image_directory, analysis_directory, excel_file_path, settings)

    print("Finished")

if __name__ == "__main__":
    main()
    exit_input = input("Press the enter key to exit...")

