import numpy as np

from design_library.Lm_design import design_converter_Lm, design_converters_Lm
from design_library.Cr_design import design_converter_Cr, design_converters_Cr
from design_library.converter_limits import converter_limits_fmin

####################################################

def combined_design_Lm2Cr(Vin_min, Vin_max, Vout_, Iout_max, Vd_max, fmin_min, fmin_max, Lm_, max_Cr_num=3, printing=False, plot=False, savePrint=False, savePlot=False, npoints=100):

    Cr_s, Lr_s, Lm_s, fr_s = design_converters_Lm(Vin_min, Vin_max, Vout_, Iout_max, Vd_max, fmin_min, fmin_max, Lm_, printing=False, plot=False, savePrint=False, savePlot=False, npoints=npoints)

    f_s = sorted(Cr_s.keys())

    Cr_min = min(Cr_s.values())
    Cr_max = max(Cr_s.values())

    print(f"Cr_min = {Cr_min}")
    print(f"Cr_max = {Cr_max}")

    E6 = np.array([1, 1.5, 2.2, 3.3, 4.7, 6.8, 10, 15, 22, 33, 47, 68, 100])

    Cr_min_log = np.floor(np.log10(Cr_min)) - 1
    
    cap_s = 10**Cr_min_log * E6


    median_Cr = np.median(list(Cr_s.values()))
    median_id = np.argmin(np.abs(np.array(Cr_s.values()) - median_Cr))

    for Cr_num in range(1, max_Cr_num + 1):

        #generate Cr_num copies of cap_s
        caps_list = [cap_s] * Cr_num

        caps_list = np.meshgrid(*caps_list)

        cap_combinations = np.unique(sum(caps_list))

        min_error = np.inf
        best_Cr = None


        best_id = np.argmin(np.abs(cap_combinations - median_Cr))

        Cr_a = cap_combinations[best_id]
        Cr_error = abs(Cr - Cr_a)

        if Cr_error < min_error:
            min_error = Cr_error
            best_Cr = Cr_a

        print(f"{Cr_num} Cr = {best_Cr} error = {min_error}")

    else:
        print("No solution found")
####################################################