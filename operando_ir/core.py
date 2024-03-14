# import numpy as np
import os
import glob
import re
import copy
import warnings
import numpy as np
import pandas as pd
import brukeropusreader as bkr
import matplotlib.pyplot as plt

from datetime import datetime
from scipy.signal import argrelextrema
from brukeropusreader.opus_data import OpusData

# import spectrochempy as scp
# import pandas as pd

# from .helpers import get_noise, get_uc, get_limits

def load_spectrum(path_to_file):
    # Load the spectrum
    spectrum = bkr.read_file(path_to_file)

    return spectrum


def load_spectra(path_to_directory):
    # create an empty dictionary to hold the spectra
    spectra_dict = {}

    # iterate over each file in the directory
    for filepath in glob.glob(os.path.join(path_to_directory, '*')):

        # match the extension, should be a period followed by a number
        match = re.match("\.\d+", os.path.splitext(filepath)[1])

        # set experiment name
        experiment_name = os.path.basename(filepath).split(".")[0]

        if match:
            # load the spectrum
            spectrum = bkr.read_file(filepath)

            # get the number from the extension to use as the key in the dictionary
            spectrum_number = int(match.group()[1:])  # remove the period and convert to integer

            # add the spectrum to the dictionary
            spectra_dict[spectrum_number] = spectrum

    return experiment_name, spectra_dict

def spectra_subtraction(spectra_dict, spectrum_to_subtract, background=False):
    # create a dictionary to hold the subtracted spectra
    subtracted_spectra_dict = copy.deepcopy(spectra_dict)

    # iterate over the spectra in the dictionary
    for spectrum_number, spectrum in spectra_dict.items():
        # Set spectrum type
        spectrum_type = "AB"
        if background:
            spectrum_type = "ScSm"

        # Ensure the spectra are the same size
        if len(spectrum[spectrum_type]) == len(spectrum_to_subtract[spectrum_type]):
            # subtract the spectrum
            subtracted_spectrum = spectrum[spectrum_type] - spectrum_to_subtract[spectrum_type]

            # Update the subtracted spectrum in the subtracted_spectra_dict
            subtracted_spectra_dict[spectrum_number]["AB"] = subtracted_spectrum
        else:
            print(f"Spectrum sizes do not match for spectrum number: {spectrum_number}. Skipping subtraction.")

    return subtracted_spectra_dict



def plot_spectra(spectra_dict, single_channel=False):
    # create a new figure
    plt.figure()

    # check what spectrum type is wanted
    spectrum_type = "AB"

    if single_channel:
        spectrum_type = "ScSm"

    # iterate over the spectra in the dictionary
    for filename, spectrum in spectra_dict.items():
        # get the x-values and y-values
        # here we're assuming that 'AB' block exists in the file and it is what we want to plot
        x_values = spectrum.get_range(spectrum_type)
        y_values = spectrum[spectrum_type][0:len(x_values)]

        # plot the spectrum
        # plt.plot(x_values, y_values, label=filename)
        plt.plot(x_values, y_values)

    # add a legend
    # plt.legend()

    # show the plot
    plt.show()

def intensity_profiles(spectra_dict, peaks, window=16, local_max=True):
    # create an empty dictionary to hold the local maxima
    intensities_dict = {}

    # create a dictionary to store max intensities
    max_intensities_dict = {peak: -np.inf for peak in peaks}

    # iterate over the spectra in the dictionary
    for spectrum_number, spectrum in spectra_dict.items():
        # get the x-values and y-values
        x_values = spectrum.get_range("AB")
        y_values = spectrum["AB"][0:len(x_values)]

        intensities_dict[spectrum_number] = {}
        temp_max_intensities = {}

        # iterate over the peaks
        for peak in peaks:
            # find the indices of the x-values within the window
            window_indices = (x_values >= (peak - window/2)) & (x_values <= (peak + window/2))

            if local_max:
                # find the indices of the local maxima within the window
                local_maxima_indices = argrelextrema(y_values[window_indices], np.greater)

                # find the maximum y-value within the window and its corresponding x-value
                if len(local_maxima_indices[0]) > 0:  # check if any local maxima were found
                    max_index = local_maxima_indices[0][np.argmax(y_values[window_indices][local_maxima_indices])]
                    max_x = x_values[window_indices][max_index]
                    max_y = y_values[window_indices][max_index]

                    # add the local maximum to the dictionary
                    intensities_dict[spectrum_number][peak] = (max_x, max_y)
                    
                    # update the max intensity for this peak if it's greater than the current max
                    max_intensities_dict[peak] = max(max_intensities_dict[peak], max_y)

            else:
                 # find the closest x-value within the window to the peak
                distances = abs(x_values[window_indices] - peak)
                closest_index = distances.argmin()

                # get the corresponding y-value
                closest_x = x_values[window_indices][closest_index]
                closest_y = y_values[window_indices][closest_index]

                # add the closest point to the dictionary
                intensities_dict[spectrum_number][peak] = (closest_x, closest_y)
                
                # update the max intensity for this peak if it's greater than the current max
                max_intensities_dict[peak] = max(max_intensities_dict[peak], closest_y)
        
    return intensities_dict, max_intensities_dict

def plot_intensity_profiles(intensity_profiles, times=None):
    # Create a figure
    plt.figure()

    # Iterate over all peaks in the intensity profiles
    for peak in set(peak for sublist in intensity_profiles.values() for peak in sublist.keys()):
        # Get the spectrum numbers and intensities for this peak
        spectrum_numbers = [spectrum_number for spectrum_number in intensity_profiles.keys() if peak in intensity_profiles[spectrum_number]]
        intensities = [intensity_profiles[spectrum_number][peak][1] for spectrum_number in spectrum_numbers]

        # Set x values either to spectrum number or time in min
        if times is None:
            x_values = spectrum_numbers
            title = "Intensity of Local Maxima over Spectrum Number"
            x_label = "Spectrum Number"
        
        else:
            x_values = [times[spectrum_number] for spectrum_number in spectrum_numbers]
            title = "Intensity of Local Maxima over Time (min)"
            x_label = "Time (min)"

        # Create a scatter plot of the intensities over spectrum number for this peak
        plt.scatter(x_values, intensities, label=f'Peak at {peak}')

    # Add a legend, title, and labels
    plt.legend()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Intensity (a.u.)")

    # Display the plot
    plt.show()

def peak_shifts(local_maxima_dict):
    # Create an empty dictionary to hold the peak shifts
    peak_shift_dict = {}

    # Iterate over all peaks in the local maxima dictionary
    for peak in set(peak for sublist in local_maxima_dict.values() for peak in sublist.keys()):
        # Create an empty list to hold the peak shifts for this peak
        peak_shifts = []

        reference_position = None  # Initialize the reference position as None

        # Iterate over all spectra
        for spectrum_number, spectrum_peaks in local_maxima_dict.items():
            # Check if the peak exists in the inner dictionary
            if peak in spectrum_peaks:
                # Use the first available peak position as the reference position
                if reference_position is None:
                    reference_position = spectrum_peaks[peak][0]

                # Calculate the peak shift for this spectrum
                peak_position = spectrum_peaks[peak][0]
                peak_shift = peak_position - reference_position

                # Add the peak shift to the list
                peak_shifts.append(peak_shift)

        # Add the list of peak shifts for this peak to the dictionary
        peak_shift_dict[peak] = peak_shifts

    return peak_shift_dict

def plot_peak_shifts(peak_shift_dict):
    # Iterate over all peaks in the peak shift dictionary
    for peak, peak_shifts in peak_shift_dict.items():
        # Create a scatter plot of the peak shifts over the spectrum number
        plt.scatter(range(len(peak_shifts)), peak_shifts, label=f'Peak at {peak}')

    # Add a legend, title, and labels
    plt.legend()
    plt.title("Peak Shifts over Spectrum Number")
    plt.xlabel("Spectrum Number")
    plt.ylabel("Peak Shift")

    # Display the plot
    plt.show()


def normalize_intensity_profiles(intensity_profiles, normalization='min_max'):
    normalized_intensity_profiles = {}
    intensities = {}

    for spectrum_number, peaks in intensity_profiles.items():
        # Create dictionary of intensities for each peak
        for peak, (position, intensity) in peaks.items():

            if peak not in intensities:
                intensities[peak] = []  # Initialize new list if necessary

            intensities[peak].append(intensity)

    if normalization == 'min_max':

        # Normalize the values for each peak to 0-1 range 
        for spectrum_number, peaks in intensity_profiles.items():

            normalized_peaks = {}

            for peak, (position, intensity) in peaks.items():
                min_value = min(intensities[peak])
                max_value = max(intensities[peak])
                normalized_intensity = (intensity - min_value) / (max_value - min_value) if max_value > min_value else 0.0
                normalized_peaks[peak] = (position, normalized_intensity)

            normalized_intensity_profiles[spectrum_number] = normalized_peaks
    
    elif normalization == 'max':

        # Normalize the values for each peak to maximum
        for spectrum_number, peaks in intensity_profiles.items():

            normalized_peaks = {}

            for peak, (position, intensity) in peaks.items():
                max_value = max(intensities[peak])
                normalized_intensity = intensity / max_value if max_value else 0.0
                normalized_peaks[peak] = (position, normalized_intensity)

            normalized_intensity_profiles[spectrum_number] = normalized_peaks

    elif isinstance(normalization, dict):
        
        # Normalize the values for each peak based on given value from normalization_dict
        normalization_dict = normalization
            
        for spectrum_number, peaks in intensity_profiles.items():
            
            normalized_peaks = {}
            
            for peak, (position, intensity) in peaks.items():
                if peak not in normalization_dict:
                    warnings.warn(f'Peak {peak} not found in normalization_dict. Defaulting to normalization value of 1.')
                    normalized_value = 1
                else:
                    normalized_value = normalization_dict[peak]

                normalized_intensity = intensity / normalized_value if normalized_value else 0.0
                normalized_peaks[peak] = (position, normalized_intensity)
                
            normalized_intensity_profiles[spectrum_number] = normalized_peaks

    return normalized_intensity_profiles

def times_relative(spectra_dict):
    times_relative = {}

    # Bruker format is 13:40:18.889 (GMT+0) // timezone need to be omitted
    start_time_str = spectra_dict[0]['ScSm Data Parameter']['TIM'].split(" ")[0]
    start_time = datetime.strptime(start_time_str, "%H:%M:%S.%f")

    for spectrum_number, spectrum in spectra_dict.items():
        # Same as above, timezone need to be omitted and str converted
        spectrum_time_str = spectrum['ScSm Data Parameter']['TIM'].split(" ")[0]
        spectrum_time = datetime.strptime(spectrum_time_str, "%H:%M:%S.%f")

        # calculation time relative
        spectrum_time_relative = (spectrum_time - start_time).total_seconds() / 60

        # Add the relative time to the dictionary
        times_relative[spectrum_number] = spectrum_time_relative

    return times_relative

def export_data(path_to_export_file, intensity_profiles, normalized_intensity_profiles=None, times_relative=None, max_intensities=None):
    # Initialize empty DataFrame
    df1 = pd.DataFrame()

    # Iterate over the spectra
    for spectrum, peaks in intensity_profiles.items():
        # For each peak, add the wavenumber and intensity as separate columns
        for peak, (wavenumber, intensity) in peaks.items():
            df1.at[spectrum, f'{peak}_wavenumber'] = wavenumber
            df1.at[spectrum, f'{peak}_intensity'] = intensity

    # Check if filename ends with '.xlsx', if not append it
    if not path_to_export_file.endswith('.xlsx'):
        path_to_export_file += '.xlsx'
        
    # Add the current date and time to the filename before '.xlsx'
    filename = path_to_export_file[:-5] + "_" + datetime.now().strftime("%Y%m%d %H%M") + '.xlsx'

    # Get the directory name from the filename
    directory = os.path.dirname(filename)

    # If the directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Write the DataFrame to an Excel file
    with pd.ExcelWriter(filename) as writer:
        df1.to_excel(writer, sheet_name='intensities')
        
        # If a normalized data is provided, create a DataFrame and write it to a second sheet
        if normalized_intensity_profiles is not None:
            df2 = pd.DataFrame()

            # Iterate over the spectra in the second dictionary
            for spectrum, peaks in normalized_intensity_profiles.items():
                for peak, (wavenumber, intensity) in peaks.items():
                    df2.at[spectrum, f'{peak}_wavenumber'] = wavenumber
                    df2.at[spectrum, f'{peak}_intensity'] = intensity

            df2.to_excel(writer, sheet_name='normalized intesities')

        # If times are provided, create a DataFrame and write it to a additional sheet
        if times_relative is not None:
            df3 = pd.DataFrame.from_dict(times_relative, orient='index')

            df3.to_excel(writer, sheet_name='times')

        # If max intensities are provided, create a DataFrame and write it to a additional sheet
        if max_intensities is not None:
            df4 = pd.DataFrame.from_dict(max_intensities, orient='index')

            df4.to_excel(writer, sheet_name='max. intensities')

    print(f"Data exported to {filename}")

def export_spectra(path_to_export_file, spectra_dict, spectra_numbers=None, single_channel=False):
    # Initialize empty DataFrame
    df1 = pd.DataFrame()

    # Iterate over the spectra and add data to the DataFrame
    for spectrum_number, spectrum in spectra_dict.items():
        if spectra_numbers is None or spectrum_number in spectra_numbers:
            x_values = spectrum.get_range("AB")
            y_values = spectrum["AB"][0:len(x_values)]

            # Create a new column for the spectrum
            df1[spectrum_number] = y_values

    # Add the x-values as a separate column
    df1.insert(0, "wavenumbers", x_values)

    # Check if filename ends with '.xlsx', if not append it
    if not path_to_export_file.endswith('.xlsx'):
        path_to_export_file += '.xlsx'
        
    # Add the current date and time to the filename before '.xlsx'
    filename = path_to_export_file[:-5] + "_" + datetime.now().strftime("%Y%m%d %H%M") + '_spectra.xlsx'

    # Get the directory name from the filename
    directory = os.path.dirname(filename)

    # If the directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Write the DataFrame to an Excel file
    with pd.ExcelWriter(filename) as writer:
        df1.to_excel(writer, sheet_name='spectra', index=False)
        
        # If a normalized data is provided, create a DataFrame and write it to a second sheet
        if single_channel:
            df2 = pd.DataFrame()

             # Iterate over the spectra and add data to the DataFrame
            for spectrum_number, spectrum in spectra_dict.items():
                if spectra_numbers is None or spectrum_number in spectra_numbers:
                    x_values = spectrum.get_range("ScSm")
                    y_values = spectrum["ScSm"][0:len(x_values)]

                    # Create a new column for the spectrum
                    df2[spectrum_number] = y_values

            # Add the x-values as a separate column
            df2.insert(0, "wavenumbers", x_values)

            df2.to_excel(writer, sheet_name='single channel', index=False)

    print(f"Spectra exported to {filename}")

def load_ms_data(path_to_file):
    # Open the file and read the 4th line
    with open(path_to_file, 'r') as file:
        lines = file.readlines()
        timestamp = lines[3].split("\t")[0]  # assuming timestamp is the first value in the 4th line
        headers = lines[8].split("\t")  # get the headers from the 9th line

    # Convert necessary headers to float
    for i in range(4, len(headers)):
        try:
            headers[i] = float(headers[i].strip())  # convert to float and remove newline characters
        except ValueError:
            pass  # keep as string if cannot convert to float

    # Convert timestamp string to datetime
    ms_timestamp = pd.to_datetime(timestamp)

    # Read the data from the 10th row, and onwards, without headers
    ms_data = pd.read_csv(path_to_file, delimiter='\t', skiprows=9, header=None)

    # Add headers manually
    ms_data.columns = headers

    # Convert the dataframe to dictionary
    ms_data_dict = ms_data.to_dict(orient='list')

    return ms_timestamp, ms_data_dict

def plot_ms_data(ms_data_dict, mz_ratios=None, time_correction=None):
    # If no specific m/z ratios are provided, plot all m/z ratios
    if mz_ratios is None:
        mz_ratios = [mz_ratio for mz_ratio in ms_data_dict.keys() if isinstance(mz_ratio, float) and mz_ratio >= 5.0]

    # Get time relative in min
    times = [time/60 for time in ms_data_dict['Time Relative (sec)']]  # divide each time value by 60
    if time_correction is not None:
        times = [time - time_correction for time in times]  # subtract time_correction from each time value


    # Plot data for each m/z ratio
    for mz_ratio in mz_ratios:
        y_values = ms_data_dict[mz_ratio]
        plt.semilogy(times, y_values, label=str(mz_ratio))

    plt.xlabel('Time relative (min)')
    plt.ylabel('Intensity (a.u.)')
    plt.legend(loc='best')
    plt.show()