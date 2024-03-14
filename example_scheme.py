import operando_ir.core as operando_ir

# suppress warnings
import warnings
warnings.filterwarnings("ignore")

path_to_files = "example-data/"
path_to_bkg = "example-data/background/"

# Load multiple spectra as opus files from directory
experiment_name, spectra_dict = operando_ir.load_spectra(path_to_files)
print(f"Number of spectra: {len(spectra_dict)}")

# Get time relative
times_relative = operando_ir.times_relative(spectra_dict)

# Plot all spectra in a single figure as single channel spectrum and absorbance
operando_ir.plot_spectra(spectra_dict, single_channel=True)
# operando_ir.plot_spectra(spectra_dict, single_channel=False)

# # Background substraction
bkg_data = operando_ir.load_spectrum(path_to_bkg + "bkg.0000")
spectra_dict = operando_ir.spectra_subtraction(spectra_dict, bkg_data, background=False)

# Plot all spectra after background substraction
operando_ir.plot_spectra(spectra_dict, single_channel=False)

# Get intensity profiles for given x-values (wavenumbers) in cm-1; window standard value is 4 cm-1
peaks = [2971, 2931, 2874, 2826, 2735, 1585, 1377, 1143, 1052, 1033]
intensity_profiles, max_intensities = operando_ir.intensity_profiles(spectra_dict, peaks, window=32, local_max=False)
operando_ir.plot_intensity_profiles(intensity_profiles, times=times_relative)

# Normalize intensitiy profiles
max_intensities = {2971: 0.29106664657592773, 2931: 0.3206338882446289, 2874: 0.48181843757629395, 2826: 0.35423707962036133, 2735: 0.11245644092559814, 1585: 0.80251145362854, 1377: 0.40361976623535156, 1143: 0.1350727081298828, 1052: 0.30144834518432617, 1033: 0.24913454055786133}
normalized_intensity_profiles = operando_ir.normalize_intensity_profiles(intensity_profiles, normalization=max_intensities)
operando_ir.plot_intensity_profiles(normalized_intensity_profiles)

# Import MS data and plot with time correction in min
ms_timestamp, ms_data_dict = operando_ir.load_ms_data(path_to_files + "ms_data/example.dat")
operando_ir.plot_ms_data(ms_data_dict, mz_ratios=[2, 4, 31, 32, 44], time_correction=-50)

# Export (normalized) intensity profiles as one single xlsx
export_file_path = path_to_files + "export/" + experiment_name + ".xlsx"
operando_ir.export_data(export_file_path, intensity_profiles, normalized_intensity_profiles, times_relative)

# Export spectra in one single xlsx file with common column of x-values
operando_ir.export_spectra(export_file_path, spectra_dict, spectra_numbers=None, single_channel=False)

# # to do
# # peak_shifts = operando_ir.peak_shifts(intensity_profiles)
# # operando_ir.plot_peak_shifts(peak_shifts)
# # 
