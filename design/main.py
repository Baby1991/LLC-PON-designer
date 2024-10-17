from design_library.Lm_design import design_converter_Lm, design_converters_Lm
from design_library.Cr_design import design_converter_Cr, design_converters_Cr
from design_library.converter_limits import converter_limits_fmin
from design_library.combined_design import combined_design_Lm2Cr

if __name__ == "__main__":
    #design_converter(385, 420, 12, 12.5, 0.1, 90e3, printing=True, plot=True) # Good solution
    #design_converter(385, 420, 12, 13,   0.1, 90e3, printing=True, plot=True) # Bad solution

    #design_converter(385, 420, 12, 17, 0.1, 64e3, printing=True, plot=True, savePrint=False, savePlot=False, askCr=True) # Good solution

    #converter_limits_Lm(385, 420, 12, 17, 0.1, 30e3, 120e3, 1500e-6, printing=True, npoints=50)
    converter_limits_fmin(385, 420, 12, 17, 0.1, 30e3, 70e3, plot=True, npoints=20, show=False)
    converter_limits_fmin(385, 420, 12, 35, 0.1, 30e3, 70e3, plot=True, npoints=20)

    #design_converter_Lm(385, 420, 12, 17, 0.1, 65e3, 1500e-6, printing=True, savePrint=True)
    #combined_design_Lm2Cr(385, 420, 12, 17, 0.1, 53e3, 71e3, 1500e-6, max_Cr_num=3, printing=True, plot=True, savePlot=False, savePrint=False, npoints=100)
    
