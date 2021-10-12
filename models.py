# -*- coding: utf-8 -*-
'''
    models.py

    Alex Koch 2021

    Defines the auto and cross correlation models used in Fish and Cushion FCS analysis tool
'''

import numpy as np
from scipy.integrate import tplquad
import math 

'''

The structual parameter, a, is handled as a staticaly defined paramter until specifically overriden as per:
https://stackoverflow.com/questions/68645/are-static-class-variables-possible-in-python


'''

def lookup_model(model_name):
    if model_name == "normal_diffusion":
        return normal_diffusion
    elif model_name == "normal_diffusion_offset":
        return normal_diffusion_offset
    elif model_name == "anomalous_diffusion":
        return anomalous_diffusion
    elif model_name == "anomalous_diffusion_offset":
        return anomalous_diffusion_offset
    elif model_name == "anomalous_diffusion_triplet":
        return anomalous_diffusion_triplet
    elif model_name == "anomalous_diffusion_triplet_offset":
        return anomalous_diffusion_triplet_offset
    elif model_name == "anomalous_diffusion_1D":
        return anomalous_diffusion_1D
    elif model_name == "anomalous_diffusion_2D":
        return anomalous_diffusion_2D
    elif model_name == "anomalous_diffusion_1D_triplet":
        return anomalous_diffusion_1D_triplet
    elif model_name == "anomalous_diffusion_2D_triplet":
        return anomalous_diffusion_2D_triplet
    elif model_name == "normal_diffusion_triplet":
        return normal_diffusion_triplet
    elif model_name == "normal_diffusion_triplet_offset":
        return normal_diffusion_triplet_offset
    elif model_name == "multiple_components":
        return multiple_components
    elif model_name == "normal_diffusion_two_component":
        return normal_diffusion_two_component
    elif model_name == "normal_diffusion_two_component_offset":
        return normal_diffusion_two_component_offset
    elif model_name == "normal_diffusion_two_component_triplet":
        return normal_diffusion_two_component_triplet
    elif model_name == "normal_diffusion_two_component_triplet_offset":
        return normal_diffusion_two_component_triplet_offset
    #elif model_name == "normal_diffusion_two_component":
    #    return normal_diffusion_two_fast_processes
    elif model_name == "hybrid_model_binding":
        return hybrid_model_binding
    elif model_name == "reaction_dominant_binding":
        return reaction_dominant_binding
    elif model_name == "reaction_dominant_binding_triplet":
        return reaction_dominant_binding_triplet
    elif model_name == "normal_diffusion_bleaching":
        return normal_diffusion_bleaching
    elif model_name == "negative_cross_correlation":
        return negative_cross_correlation
    elif model_name == "cross_correlation_normal":
        return cross_correlation_normal
    elif model_name == "cross_correlation_normal_offset":
        return cross_correlation_normal_offset
    else:
        print("ERROR: Model specified incorrectly or is missing. Using Normal diffusion model")
        return normal_diffusion

N_=2000

class normal_diffusion:
    name = "normal_diffusion"
    args = ['N', 'tau_D']
    guess_values = [1, # N
                    0.01] # tau_D
    bounds = ([0, 1E-5], [N_, 10])
    a = 6.8
    @staticmethod
    def correlation(tau, N, tau_D):#, offset):
        return (1/N) * (1/( (1+(tau/tau_D)) * (1 + normal_diffusion.a**-2 * (tau/tau_D))**0.5 ) )

class normal_diffusion_offset:
    name = "normal_diffusion_offset"
    args = ['N', 'tau_D', 'offset']
    guess_values = [10, # N
                    0.01,
                    0] # tau_D
    bounds = ([0, 1E-5, 0], [N_, 10, 1])
    a=6.8
    @staticmethod
    def correlation(tau, N, tau_D, offset):
        return (1/N) * (1/( (1+(tau/tau_D)) * (1 + normal_diffusion_offset.a**-2 * (tau/tau_D))**0.5 ) ) + offset

class negative_cross_correlation:
    name = "negative_cross_correlation"
    args = ['gradient', 'intercept']
    guess_values = [0, # gradient
                    0] # intercept
    bounds = ([-1E3, -1], [1E3, 2])
    a = 6.8
    @staticmethod
    def correlation(tau, gradient, intercept):
        return gradient * tau + intercept

class cross_correlation_normal:
    name = "cross_correlation_normal"
    args = ['N', 'tau_D']
    guess_values = [1, # N
                    0.01] # tau_D
    bounds = ([0, 1E-5], [N_, 10])
    a = 6.8
    @staticmethod
    def correlation(tau, N, tau_D):#, offset):
        return N * (1/( (1+(tau/tau_D)) * (1 + normal_diffusion.a**-2 * (tau/tau_D))**0.5 ) )

class cross_correlation_normal_offset:
    name = "cross_correlation_normal_offset"
    args = ['N', 'tau_D', 'offset']
    guess_values = [10, # N
                    0.01, # tau_D
                    0] #offset
    bounds = ([0, 1E-5, 0], [1E6, 10, 1])
    a=6.8
    @staticmethod
    def correlation(tau, N, tau_D, offset):
        return N * (1/( (1+(tau/tau_D)) * (1 + normal_diffusion_offset.a**-2 * (tau/tau_D))**0.5 ) ) + offset

class anomalous_diffusion:
    name = "anomalous_diffusion"
    args = ["N", "tau_D", "alpha"]
    guess_values = [10,   # N
                    0.01, # tau_D
                    0.7]  # alpha
    bounds = ([0, 1E-5, 0.1], [N_, 10, 2])
    a=6.8
    @staticmethod
    def correlation(tau, N, tau_D, alpha):
        #HANNAH added in power of alpha in first term of denominator 
        return (1/N) * (1/( (1+(tau/tau_D)**alpha) * (1 + anomalous_diffusion.a**-2 * (tau/tau_D)**alpha)**0.5 ) )

class anomalous_diffusion_offset:
    name = "anomalous_diffusion_offset"
    args = ["N", "tau_D", "alpha", "offset"]
    guess_values = [10,   # N
                    0.01, # tau_D
                    0.7,  # alpha
                    0]
    bounds = ([0, 1E-5, 0.1, 0], [N_, 10, 2, 1])
    a=6.8
    @staticmethod
    def correlation(tau, N, tau_D, alpha, offset):
        #HANNAH added in power of alpha in first term of denominator 
        return (1/N) * (1/( (1+(tau/tau_D)**alpha) * (1 + anomalous_diffusion.a**-2 * (tau/tau_D)**alpha)**0.5 ) ) + offset

class anomalous_diffusion_triplet:
    name = "anomalous_diffusion_triplet"
    args = ["N", "tau_D", "alpha", "A_trip", "tau_F"]
    guess_values = [10,   # N
                    0.01, # tau_D
                    0.7,  # alpha
                    0.5,  # A_trip
                    1E-6] # tau_F
    bounds = ([0, 1E-5, 0.1, 0, 2E-7], [N_, 10, 2, 1, 2E-5])
    a=6.8
    @staticmethod
    def correlation(tau, N, tau_D, alpha, A_trip, tau_F):
        #HANNAH added in power of alpha in first term of denominator 
        return (1/N) * (1 + A_trip * np.exp(-tau/tau_F)) * (1/( (1+(tau/tau_D)**alpha) * (1 + anomalous_diffusion.a**-2 * (tau/tau_D)**alpha)**0.5 ) )

class anomalous_diffusion_triplet_offset:
    name = "anomalous_diffusion_triplet_offset"
    args = ["N", "tau_D", "alpha", "A_trip", "tau_F", "offset"]
    guess_values = [10,   # N
                    0.01, # tau_D
                    0.7,  # alpha
                    0.5,  # A_trip
                    1E-6, # tau_F
                    0]
    bounds = ([0, 1E-5, 0.1, 0, 2E-7, 0], [N_, 10, 2, 1, 2E-5, 1])
    a=6.8
    @staticmethod
    def correlation(tau, N, tau_D, alpha, A_trip, tau_F, offset):
        #HANNAH added in power of alpha in first term of denominator 
        return (1/N) * (1 + A_trip * np.exp(-tau/tau_F)) * (1/( (1+(tau/tau_D)**alpha) * (1 + anomalous_diffusion.a**-2 * (tau/tau_D)**alpha)**0.5 ) ) + offset
    
class anomalous_diffusion_1D:
    #HANNAH added this model
    name = "anomalous_diffusion_1D"
    args = ["N", "tau_D", "alpha"]
    guess_values = [10, # N
                0.01, # tau_D
                #5, #a
                0.5]#, # alpha
    bounds = ([0, 0, 0], [1E6, 1E-1, 2])
    a=6.8
    @staticmethod
    def correlation(tau, N, tau_D, alpha):#, offset):
        return (1/N) * (1/( (1+(tau/tau_D)**alpha)**0.5 ))# + offset
                        
class anomalous_diffusion_2D:
    #HANNAH added this model
    name = "anomalous_diffusion_2D"
    args = ["N", "tau_D", "alpha"]
    guess_values = [10, # N
                0.01, # tau_D
                #5, #a
                0.5]#, # alpha
    bounds = ([0, 0, 0], [1E6, 1E-1, 2])
    a=6.8
    @staticmethod
    def correlation(tau, N, tau_D, alpha):#, offset):
        return (1/N) * (1/( (1+(tau/tau_D)**alpha) ))# + offset
                        
class anomalous_diffusion_1D_triplet:
    #HANNAH added this model
    name = "anomalous_diffusion_1D_triplet"
    args = ["N", "tau_D", "alpha", "A_trip"]
    guess_values = [10, # N
                0.01, # tau_D
                #5, #a
                0.5, # alpha
                0.5]#, # A_trip
    bounds = ([0, 0, 0], [1E6, 1E-1, 2])
    a=6.8
    @staticmethod
    def correlation(tau, N, tau_D, alpha, A_trip):#, offset):
        tau_F = 1E-6
        return (1/N) * (1 + A_trip * np.exp(-tau/tau_F)) * (1/( (1+(tau/tau_D)**alpha)**0.5 ))# + offset
                        
class anomalous_diffusion_2D_triplet:
    #HANNAH added this model
    name = "anomalous_diffusion_2D_triplet"
    args = ["N", "tau_D", "alpha", "A_trip"]
    guess_values = [10, # N
                0.01, # tau_D
                #5, #a
                0.5, # alpha
                0.5]#, # A_trip
    bounds = ([0, 0, 0], [1E6, 1E-1, 2])
    a=6.8
    @staticmethod
    def correlation(tau, N, tau_D, alpha, A_trip):#, offset):
        tau_F = 1E-6
        return (1/N) * (1 + A_trip * np.exp(-tau/tau_F)) * (1/( (1+(tau/tau_D)**alpha)) )# + offset   
                        
class normal_diffusion_triplet:
    name = "normal_diffusion_triplet"
    #args = ["tau", "N", "tau_D", "a", "F", "tau_F", "offset"]
    #args = ["tau", "N", "tau_D", "F", "tau_F", "offset"]
    #args = ["tau", "N", "tau_D", "offset", "F", "tau_F"]
    #args = ["N", "tau_D", "offset", "F", "tau_F"]
    '''
    args = ["N", "tau_D", "F", "tau_F"]
    guess_values = [10, # N
                0.01, # tau_D
                #5, #a
                0.5, # F
                1E-6]#, # tau_F
                #0] # offset
    '''
    args = ["N", "tau_D", "A_trip", "tau_F"]
    guess_values = [10, # N
                0.01, # tau_D
                0.5, # A_trip
                1E-6]#, # tau_F
                #0] # offset
    #bounds = ([0, 0, 3, 0, 0, -0.5], [1E6, 1E-1, 7, 1, 1E-3, 1.5])
    #bounds = ([0, 0, 0, 0, -0.001], [1E6, 1E-1, 1, 1E-3, 0.001])
    bounds = ([0, 1E-5, 0, 2E-7], [1E6, 10, 1, 2E-5])
    a=6.8
    @staticmethod
    def correlation(tau, N, tau_D, A_trip, tau_F):#, offset):
        '''
        Normal diffusion correlation function with triplet state correction
        https://en.wikipedia.org/wiki/Fluorescence_correlation_spectroscopy
        Parameters
            tau: Float; the lag time in seconds 
            N: Float; Average number of particles in the confocal volume
            tau_D: Floats; Defines the diffusion dwell time
            a: Float; Structural parameter related to the shape of confocal volume and detector settings
            F: Float; Fraction of particles that have entered the triplet state
            tau_F: Float: Triplet state relaxation time
        '''
        
        #tau_F = 1E-6
        #return (1/N) * ((1 - F + F * np.exp(-tau/tau_F)) / (1-F)) * (1/( (1+(tau/tau_D)) * (1 + a**-2 * (tau/tau_D))**0.5 ) )# + offset
        return (1/N) * (1 + A_trip * np.exp(-tau/tau_F)) * (1/( (1+(tau/tau_D)) * (1 + normal_diffusion_triplet.a**-2 * (tau/tau_D))**0.5 ) )
    
class normal_diffusion_triplet_offset:
    name = "normal_diffusion_triplet_offset"

    args = ["N", "tau_D", "A_trip", "tau_F", "offset"]
    guess_values = [10, # N
                0.01, # tau_D
                0.5, # A_trip
                1E-6,#, # tau_F
                0] # offset
    #bounds = ([0, 0, 3, 0, 0, -0.5], [1E6, 1E-1, 7, 1, 1E-3, 1.5])
    #bounds = ([0, 0, 0, 0, -0.001], [1E6, 1E-1, 1, 1E-3, 0.001])
    bounds = ([0, 1E-5, 0, 2E-7, 0], [1E6, 1E-1, 1, 2E-5, 1])
    a=6.8
    @staticmethod
    def correlation(tau, N, tau_D, A_trip, tau_F, offset):
        '''
        Normal diffusion correlation function with triplet state correction
        https://en.wikipedia.org/wiki/Fluorescence_correlation_spectroscopy
        Parameters
            tau: Float; the lag time in seconds 
            N: Float; Average number of particles in the confocal volume
            tau_D: Floats; Defines the diffusion dwell time
            a: Float; Structural parameter related to the shape of confocal volume and detector settings
            F: Float; Fraction of particles that have entered the triplet state
            tau_F: Float: Triplet state relaxation time
        '''
        
        #tau_F = 1E-6
        #return (1/N) * ((1 - F + F * np.exp(-tau/tau_F)) / (1-F)) * (1/( (1+(tau/tau_D)) * (1 + a**-2 * (tau/tau_D))**0.5 ) )# + offset
        return (1/N) * (1 + A_trip * np.exp(-tau/tau_F)) * (1/( (1+(tau/tau_D)) * (1 + normal_diffusion_triplet.a**-2 * (tau/tau_D))**0.5 ) ) + offset
    
class normal_diffusion_two_fast_processes:
    '''
    From Hendrix et al https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2367191/pdf/4103.pdf
    Useful for red fluorescent proteins
    '''
    name = "normal_diffusion_two_fast_processes"
    args = ["N", "tau_D", "F_1", "F_2", "tau_F_1", "tau_F_2"]
    guess_values = [10, # N
                    0.01, # tau_D
                    0.1, # F_1
                    0.1, # F_2
                    1E-6, # tau_F_1
                    1E-6]#, # tau_F_2
                #0] # offset
    #bounds = ([0, 0, 3, 0, 0, -0.5], [1E6, 1E-1, 7, 1, 1E-3, 1.5])
    #bounds = ([0, 0, 0, 0, -0.001], [1E6, 1E-1, 1, 1E-3, 0.001])
    bounds = ([0, 0, 0, 0], [1E6, 1E-1, 1, 1E-3])
    a=6.8
    @staticmethod    
    def correlation(tau, N, tau_D, F_1, F_2, tau_F_1, tau_F_2):#, offset):
        '''
        Normal diffusion correlation function with 2 triplet states correction
        https://en.wikipedia.org/wiki/Fluorescence_correlation_spectroscopy
        Parameters
            tau: Float; the lag time in seconds 
            N: Float; Average number of particles in the confocal volume
            tau_D: Floats; Defines the diffusion dwell time
            a: Float; Structural parameter related to the shape of confocal volume and detector settings
            F: Float; Fraction of particles that have entered the triplet state
            tau_F: Float: Triplet state relaxation time
        '''
        return (1/N) * (1 + (F_1 / (1 - F_1)) * np.exp(-tau/tau_F_1)) * (1 + (F_2 / (1 - F_2)) * np.exp(-tau/tau_F_2)) * (1/( (1+(tau/tau_D)) * (1 + normal_diffusion_two_fast_processes.a**-2 * (tau/tau_D))**0.5 ) )

class multiple_components:
    name = "multiple_components"
    args = ["N", "tau_Ds", "a", "weights", "offset"]
    guess_values = [10, # N
                0.01, # tau_D
                5, #a
                0.5, # alpha
                0] # offset
    @staticmethod
    def correlation(tau, N, tau_Ds, a, weights, offset):
        '''
        Polydisperse diffusion correlation function used for fitting to multiple components
        https://en.wikipedia.org/wiki/Fluorescence_correlation_spectroscopy
        Parameters
            tau: Float; the lag time in seconds 
            N: Float; Average number of particles in the confocal volume
            tau_Ds: n sized array of floats; Defines the diffusion dwell times of the n different components 
            a: Float; Structural parameter related to the shape of confocal volume and detector settings
            weights: n sized array of floats; Weighting for each component
        '''
        components = [(weight/((1+(tau/tau_D))*(1+multiple_components.a**-2*(tau/tau_D))**0.5)) for weight, tau_D in zip(weights, tau_Ds)]
        return (1/N) * np.cumsum(components) + offset
    
class normal_diffusion_two_component:
    name = "normal_diffusion_two_component"
    args = ["N", "fraction", "tau_D", "tau_D_2"]
    guess_values = [10,   # N
                    0.5,   # fraction
                    0.001, # tau_D_1
                    0.1] # tau_D_2
    bounds = ([0, 0, 1E-5, 1E-5], [N_, 1, 10, 10])
    a=6.8
    @staticmethod
    def correlation(tau, N, fraction, tau_D_1, tau_D_2):
        '''
        Polydisperse diffusion correlation function used for fitting to two components
        https://en.wikipedia.org/wiki/Fluorescence_correlation_spectroscopy
        Parameters
            tau: Float; the lag time in seconds 
            N: Float; Average number of particles in the confocal volume
            fraction: float; Weighting for 1st component, the second component is given by 1 - fraction
            tau_D_1: float; Diffusion dwell times of the 1st component
            tau_D_2: float; Diffusion dwell times of the 1st component
            
        '''
        # Set N = 1 in each individual function to remove it from the fitting
        return (1/N) * (fraction * normal_diffusion.correlation(tau, 1, tau_D_1) + (1 - fraction) * normal_diffusion.correlation(tau, 1, tau_D_2))

class normal_diffusion_two_component_offset:
    name = "normal_diffusion_two_component_offset"
    args = ["N", "fraction", "tau_D", "tau_D_2", "offset"]
    guess_values = [10,    # N
                    0.5,   # fraction
                    0.001, # tau_D_1
                    0.1,   # tau_D_2
                    0]     # offset
    bounds = ([0, 0, 1E-5, 1E-5, 0], [N_, 1, 10, 10, 1])
    a=6.8
    @staticmethod
    def correlation(tau, N, fraction, tau_D_1, tau_D_2, offset):
        '''
        Polydisperse diffusion correlation function used for fitting to two components
        https://en.wikipedia.org/wiki/Fluorescence_correlation_spectroscopy
        Parameters
            tau: Float; the lag time in seconds 
            N: Float; Average number of particles in the confocal volume
            fraction: float; Weighting for 1st component, the second component is given by 1 - fraction
            tau_D_1: float; Diffusion dwell times of the 1st component
            tau_D_2: float; Diffusion dwell times of the 1st component
            
        '''
        # Set N = 1 in each individual function to remove it from the fitting
        return (1/N) * (fraction * normal_diffusion.correlation(tau, 1, tau_D_1) + (1 - fraction) * normal_diffusion.correlation(tau, 1, tau_D_2)) + offset

class normal_diffusion_two_component_triplet:
    name = "normal_diffusion_two_component_triplet"
    args = ["N", "fraction", "tau_D", "tau_D_2", "A_trip", "tau_F"]
    guess_values = [10,    # N
                    0.5,   # fraction
                    0.001, # tau_D_1
                    0.1,   # tau_D_2
                    0.5,     # A_trip
                    1E-6]      # tau_F
    bounds = ([0, 0, 1E-5, 1E-5, 2E-7, 0], [N_, 1, 10, 10, 2E-5, 1])
    a=6.8
    @staticmethod
    def correlation(tau, N, fraction, tau_D_1, tau_D_2, A_trip, tau_F):
        '''
        Polydisperse diffusion correlation function used for fitting to two components
        https://en.wikipedia.org/wiki/Fluorescence_correlation_spectroscopy
        Parameters
            tau: Float; the lag time in seconds 
            N: Float; Average number of particles in the confocal volume
            fraction: float; Weighting for 1st component, the second component is given by 1 - fraction
            tau_D_1: float; Diffusion dwell times of the 1st component
            tau_D_2: float; Diffusion dwell times of the 1st component
            
        '''
        # Set N = 1 in each individual function to remove it from the fitting
        return (1/N) * (1 + A_trip * np.exp(-tau/tau_F)) * (fraction * normal_diffusion.correlation(tau, 1, tau_D_1) + (1 - fraction) * normal_diffusion.correlation(tau, 1, tau_D_2))

class normal_diffusion_two_component_triplet_offset:
    name = "normal_diffusion_two_component_triplet_offset"
    args = ["N", "fraction", "tau_D", "tau_D_2", "A_trip", "tau_F", "offset"]
    guess_values = [10,    # N
                    0.5,   # fraction
                    0.001, # tau_D_1
                    0.1,   # tau_D_2
                    0.5,   # A_trip
                    1E-6,  # tau_F
                    0]     # offset
    bounds = ([0, 0, 1E-5, 1E-5, 2E-7, 0, 0], [N_, 1, 10, 10, 2E-5, 1, 1])
    a=6.8
    @staticmethod
    def correlation(tau, N, fraction, tau_D_1, tau_D_2, A_trip, tau_F, offset):
        '''
        Polydisperse diffusion correlation function used for fitting to two components
        https://en.wikipedia.org/wiki/Fluorescence_correlation_spectroscopy
        Parameters
            tau: Float; the lag time in seconds 
            N: Float; Average number of particles in the confocal volume
            fraction: float; Weighting for 1st component, the second component is given by 1 - fraction
            tau_D_1: float; Diffusion dwell times of the 1st component
            tau_D_2: float; Diffusion dwell times of the 1st component
            
        '''
        # Set N = 1 in each individual function to remove it from the fitting
        return (1/N) * (1 + A_trip * np.exp(-tau/tau_F)) * (fraction * normal_diffusion.correlation(tau, 1, tau_D_1) + (1 - fraction) * normal_diffusion.correlation(tau, 1, tau_D_2)) + offset

'''
    Models for binding/unbinding of a normally diffusing molecule onto an immobile substrate

    https://www.sciencedirect.com/science/article/pii/S0006349509008893
'''

class hybrid_model_binding:
    '''
    For k_on >> k_off
    '''
    name = "hybrid_model_binding"
    args = ['N', 'tau_D', 'k_on', 'k_off']
    guess_values = [10,   # N
                    0.01, # tau_D
                    100,    # k_on
                    0.01]    # k_off

    bounds = ([0.01, 1E-5, 0.01, 0.0001], [1E6, 1E-1, 1000, 1000])
    @staticmethod
    def correlation(tau, N, tau_D, k_on, k_off):#, offset):
        return hybrid_correlation_cython(tau,N, tau_D, k_on, k_off, iterations=40, samples=N_0)

class reaction_dominant_binding:
    '''
    For tau_D << 1/k_on

    Molecule moves around a lot faster than the binding rate
    '''
    name = "reaction_dominant_binding"
    #args = ['tau', 'N', 'tau_D', 'a', 'offset']
    #args = ['tau', 'N', 'tau_D', 'offset']
    #args = ['N', 'tau_D', 'offset']
    args = ['N', 'tau_D', 'k_on', 'k_off']
    guess_values = [10,   # N
                    0.01, # tau_D
                    1.0,    # k_on
                    1.0]    # k_off
    bounds = ([0, 1E-5, 0.0001, 0.0001], [N_, 10, 1000, 1000])
    a=6.8
    @staticmethod
    def correlation(tau, N, tau_D, k_on, k_off):#, offset):
        return ((k_off / (k_off + k_on)) * (1/N) * (1/( (1+(tau/tau_D)) * (1 + reaction_dominant_binding.a**-2 * (tau/tau_D))**0.5 ) )
                + (1/N) * (1/2**1.5) * (k_on / (k_off + k_on)) * np.exp(-k_off * tau))

class reaction_dominant_binding_triplet:
    '''
    For tau_D << 1/k_on

    Molecule moves around a lot faster than the binding rate

    + a triplet state component
    '''
    name = "reaction_dominant_binding"
    #args = ['tau', 'N', 'tau_D', 'a', 'offset']
    #args = ['tau', 'N', 'tau_D', 'offset']
    #args = ['N', 'tau_D', 'offset']
    args = ['N', 'tau_D', "A_trip", "tau_F", 'k_on', 'k_off']
    guess_values = [1,   # N
                    0.0035, # tau_D
                    0.5, # A_trip
                    1E-6, # tau_F
                    3E-6,    # k_on
                    0.1]    # k_off
    
    bounds = ([0, 0, 0, 0, 2.25E-6, 0.001], [1E6, 1E-1, 1, 1E-3, 5.5E-5, 10])
    a=6.8
    @staticmethod
    def correlation(tau, N, tau_D, A_trip, tau_F, k_on, k_off):
        return ((k_off / (k_off + k_on)) * (1/N) * (1 + A_trip * np.exp(-tau/tau_F)) * (1/( (1+(tau/tau_D)) * (1 + reaction_dominant_binding_triplet.a**-2 * (tau/tau_D))**0.5 ) )
                + (1/N) * (k_on / (k_off + k_on)) * np.exp(-k_off * tau))

class normal_diffusion_bleaching:
    name = "normal_diffusion_bleaching"
    #args = ['tau', 'N', 'tau_D', 'a', 'offset']
    #args = ['tau', 'N', 'tau_D', 'offset']
    #args = ['N', 'tau_D', 'offset']
    args = ['N', 'tau_D', 't_half']
    guess_values = [10, # N
                0.01, # tau_D
                5] # t_half
    bounds = ([0, 1E-5, 0], [1E6, 1E-1, 20])
    a = 6.8
    @staticmethod
    def correlation(tau, N, tau_D, t_half):
        return np.exp(-np.log(2) * tau / t_half) * (1/N) * (1/( (1+(tau/tau_D)) * (1 + normal_diffusion.a**-2 * (tau/tau_D))**0.5 ) )