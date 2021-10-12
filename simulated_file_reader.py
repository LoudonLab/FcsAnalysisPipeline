# -*- coding: utf-8 -*-
'''
    Alex Koch 2021

    Simulated file reader
    simulated_file_reader.py

    Reads .sim files (a renamed json file) that contains synthetically generated data and mimics the ConfoCor3Fcs class from 
    fcsfiles by Christoph Gohlke (https://pypi.org/project/fcsfiles/).

    Usage:

    fcs = SimulatedFCS('synthetic_data.sim')
    
    channels = fcs['FcsData']['FcsEntry'][0]['FcsDataSet']['Acquisition']['AcquisitionSettings']['Channels']
    > integer of the number of channels
    
    positions = int(fcs['FcsData']['FcsEntry'][0]['FcsDataSet']['Acquisition']['AcquisitionSettings']['SamplePositions']['SamplePositionSettings']['Positions'])
        > integer of the number of positions

    channel = fcs['FcsData']['FcsEntry'][channel_index]['FcsDataSet']['Channel']
        > One of the following strings:
            - Auto-correlation detector Meta1
            - Auto-correlation detector Meta2
            - Cross-correlation detector Meta1 versus detector Meta2
            - Cross-correlation detector Meta2 versus detector Meta1

    postion = fcs['FcsData']['FcsEntry'][i]['FcsDataSet']['Position']
        > integer of the position (zero indexed)

    wavelength = fcs['FcsData']['FcsEntry'][i]['FcsDataSet']['Acquisition']['AcquisitionSettings']['BeamPath']['BeamPathSettings']['Attenuator'][0]['Wavelength']
        > Excitation wavelength, String with example format '488 nm'

    count_rate_data = fcs['FcsData']['FcsEntry'][i]['FcsDataSet']['CountRateArray']
        > Array of pairs of time (in s) and detected frequency (in Hz) measurements, i.e. [[0e+00, 31000], [1e-03, 35000], ...]

    corr_array = fcs['FcsData']['FcsEntry'][i]['FcsDataSet']['CorrelationArray']
        >

    max_time = fcs['FcsData']['FcsEntry'][i]['FcsDataSet']['Acquisition']['AcquisitionSettings']['MeasurementTime']
        > The time over which the measurement was made. String with example format '10.000 s' 

    correlator_bin_time = fcs['FcsData']['FcsEntry'][i]['FcsDataSet']['Acquisition']['AcquisitionSettings']['CorrelatorBinning']
        > Correlator bin time which is the smallest increment by which the multi tau correlator algorithm analyses the data.
          String with exampled format '0.200 µs'

'''

import os
import numpy as np
import json

class SimulatedFCS(dict):
    """Carl Zeiss ConfoCor3 ASCII data file.

    No specification is available. The encoding is 'Windows-1252'.

    """
    #_header = 'Carl Zeiss ConfoCor3 - measurement data file - version 3.0 ANSI'

    def __init__(self, filename):
        """Read file content and parse into dictionary."""
        dict.__init__(self)

        self._filename = filename
        with open(filename, "r+", encoding='utf-8') as file:
            data = json.loads(file.read(), strict=False)
            for key, value in data.items():
                self[key] = value

            # Cast the lists in the file to numpy arrays
            for i in range(len(self['FcsData']['FcsEntry'])):
                self['FcsData']['FcsEntry'][i]['FcsDataSet']['CountRateArray'] = np.array(self['FcsData']['FcsEntry'][i]['FcsDataSet']['CountRateArray'])
                self['FcsData']['FcsEntry'][i]['FcsDataSet']['CorrelationArray'] = np.array(self['FcsData']['FcsEntry'][i]['FcsDataSet']['CorrelationArray'])

    def __str__(self):
        """Return string close to original file format."""
        result = [self._header]

        def append(key, value, indent='', index=''):
            """Recursively append formatted keys and values to result."""
            if index != '':
                index = str(index + 1)
            if isinstance(value, dict):
                result.append('%sBEGIN %s%s %i' % (
                    indent, key, index, value['_value']))
                for k, v in sorted(value.items(), key=sortkey):
                    append(k, v, indent+'\t')
                result.append('%sEND' % indent)
            elif isinstance(value, (list, tuple)):
                for i, val in enumerate(value):
                    append(key, val, indent, i)
            elif isinstance(value, np.ndarray):
                size = value.shape[0]
                if size != 1:
                    result.append('%s%sSize = %i' % (indent, key, size))
                result.append('%s%s = %s' % (
                    indent, key, ' '.join(str(i) for i in value.shape)))
                for i in range(size):
                    result.append('%s%s' % (
                        indent, '\t '.join('%.8f' % v for v in value[i])))
            elif key != '_value':
                result.append('%s%s%s = %s' % (indent, key, index, value))

        def sortkey(item):
            """Sort dictionary items by key string and value type."""
            key, value = item
            key = key.lower()
            if isinstance(value, (list, tuple)):
                return '~' + key
            if isinstance(value, np.ndarray):
                return '~~' + key
            if isinstance(value, dict):
                return '~~~' + key
            return key

        for key, val in sorted(self.items(), key=sortkey):
            append(key, val)
        return '\n'.join(result)

    def close(self):
        """Close open file."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass