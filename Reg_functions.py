#Registration functions
import elastix
import os
import numpy as np
import SimpleITK as sitk


def Registration(fixed_image_path, atlas_path, parameter_path, ELASTIX_PATH, pnr ):
    "Function to registrate one atlas on one fixed image. pnr should be given as 'p102'"
    
    # Make a results directory if non exists
    if os.path.exists('Results/results_{}'.format(pnr)) is False:
        os.mkdir('Results/results_{}'.format(pnr))
    
    el = elastix.ElastixInterface(elastix_path=ELASTIX_PATH)
    output_dir = 'Results/results_{}'.format(pnr)
    el.register(
        fixed_image=fixed_image_path,
        moving_image=atlas_path,
        parameters=[parameter_path],
        output_dir=output_dir)
    return output_dir

#Functie om Mutual Information te bepalen
def mutual_information(fixed_img, atlas_img):
    """ Mutual information for joint histogram
    """
    hist_2d, x_edges, y_edges = np.histogram2d(fixed_img.ravel(), atlas_img.ravel(), bins=20)
    # Convert bins counts to probability values
    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))

def AdjustParameters(par_path, nrRes=3, penalty=0, nrSamples=4096):
    """ Function to adjust the parameters of Elastix. The function creates a new parameter file as filename_adj.txt and returns its path.
    Par_path is the path to the parameter file, including filename.txt
    The parameters need to be given as integers.
    If no value for a parameter is given, the default value is used.
    """

    # If parameter does not have the default value, set "to adjust" to True.
    adj_nrRes = False
    adj_penalty = False
    adj_nrSamples = False
    # Default values
    def_nrRes = 3
    def_penalty = 0
    def_nrSamples = 4096
    # Check if it needs to be adjusted
    if nrRes != def_nrRes:
        adj_nrRes = True
    if penalty != def_penalty:
        adj_penalty = True
    if nrSamples!= def_nrSamples:
        adj_nrSamples = True

    # Open the parameter file
    with open(par_path) as f:
        par_file = f.readlines()
    # Search per line if the parameter is specified there
    line_nr = 0
    for line in par_file:
        if adj_nrRes == True:
            if "NumberOfResolutions" in line:
                #Change from the default to the new value 
                line = line.replace(str(def_nrRes), str(nrRes)) 
                #Adjust it in the file
                par_file[line_nr] = line                                
        
        # adjust penalty term
        if adj_penalty == True:
            if "Metric1Weight" in line:
                line = line.replace(str(def_penalty), str(penalty))  
                par_file[line_nr] = line                                
        
        # The same as nrRes but now for nrSamples
        if adj_nrSamples == True:
            if "NumberOfSpatialSamples" in line:
                line = line.replace(str(def_nrSamples), str(nrSamples))  
                par_file[line_nr] = line                                

        # Keep track of the index of the lines
        line_nr = line_nr + 1
    
    # Close the file
    f.close()

    # Open a new file in writing mode and adjust the new parameters
    adj_par_path = par_path.replace(".txt", "_adj.txt")
    a_file = open(adj_par_path, "w")
    a_file.writelines(par_file)
    a_file.close()

    # The function has finished
    print("Parameter file adjusted")
    return adj_par_path


def bestPenalty(penalty_w, pnr, ELASTIX_PATH, fixed_path, moving_path, parameter_file_path):
    """ penalty_w is a list with the penalty weights you want to test as integers. It automatically first adjusts the 
    parameter file to the wanted penalty weight, then registrates with that parameter file and lastly calculates the mutual
    information value for that registration. The function returns the best resolution value for this patient.
    """
    MI_values = np.zeros(len(penalty_w))
    for idx in range(len(penalty_w)):
        PW = penalty_w[idx]
        # Adjust the penalty weight  
        adj_par_path = AdjustParameters(parameter_file_path, penalty=PW)
        # Registrate with the adjusted parameter file
        dir_res = Registration(fixed_path, moving_path,adj_par_path, ELASTIX_PATH, pnr)
    
        # Load the fixed and registrated atlas image
        itk_image = sitk.ReadImage(os.path.join(dir_res, 'result.0.mhd'))
        image_array_s = sitk.GetArrayFromImage(itk_image)
        f_image = sitk.ReadImage(fixed_path)
        fixed_image_s = sitk.GetArrayFromImage(f_image)
    
        # Calculate the mutual information value
        MI_value = mutual_information(fixed_image_s, image_array_s)
        MI_values[idx] = MI_value

    # Pick the penalty weight with the highest MI_value
    idx_max = np.argmax(MI_values)
    print("idx_max = ", idx_max)
    best_penalty = penalty_w[idx_max]
    print(MI_values)
    return best_penalty

def bestResolution(ResValues, pnr, ELASTIX_PATH, fixed_path, moving_path, parameter_file_path):
    """ ResValues is a list with the resolution values you want to test as integers. It automatically first adjusts the 
    parameter file to the wanted resolution, then registrates with that parameter file and lastly calculates the mutual
    information value for that registration. The function returns the best resolution value for this patient.
    """
    MI_values = np.zeros(len(ResValues))
    for idx in range(len(ResValues)):
        ResValue = ResValues[idx]
        # Adjust the Resolution value  
        adj_par_path = AdjustParameters(parameter_file_path, nrRes=ResValue)
        # Registrate with the adjusted parameter file
        dir_res = Registration(fixed_path, moving_path,adj_par_path, ELASTIX_PATH, pnr)
    
        # Load the fixed and registrated atlas image
        itk_image = sitk.ReadImage(os.path.join(dir_res, 'result.0.mhd'))
        image_array_s = sitk.GetArrayFromImage(itk_image)
        f_image = sitk.ReadImage(fixed_path)
        fixed_image_s = sitk.GetArrayFromImage(f_image)
    
        # Calculate the mutual information value
        MI_value = mutual_information(fixed_image_s, image_array_s)
        MI_values[idx] = MI_value

    # Pick the Resolution with the highest MI_value
    idx_max = np.argmax(MI_values)
    print("idx_max = ", idx_max)
    best_res = ResValues[idx_max]
    print(MI_values)
    return int(best_res)