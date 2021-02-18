#Registration functions
import elastix
import os
import numpy as np
import SimpleITK as sitk
import sys # only necessary for changing the directory to where the python script is (line )
sys.path.append('.')
sys.path.append('../NormData')


def Registration(fixed_image_path, atlas_path, parameter_path, ELASTIX_PATH, pnr ):
    "Function to registrate one atlas on one fixed image. pnr should be given as 'p102'"
    
    # Make a results folder if non exists (in Codes_def folder)
    if os.path.exists('Results') is False:
        os.mkdir('Results')

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


def Initialization_registration(fixed_image_path, atlas_path, parameter_path, ELASTIX_PATH, pnr, method ):
    "Function to registrate one atlas on one fixed image. pnr should be given as 'p102'"
    
    # Make a results directory if non exists
    if os.path.exists('Results/results_{}{}'.format(method, pnr)) is False:
        os.mkdir('Results/results_{}{}'.format(method, pnr))
    
    el = elastix.ElastixInterface(elastix_path=ELASTIX_PATH)
    output_dir = 'Results/results_{}{}'.format(method, pnr)
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

def AdjustParameters(par_path, nrRes=3, penalty=0.00001, finestRes=16):
    """ Function to adjust the parameters of Elastix. The function creates a new parameter file as filename_adj.txt and returns its path.
    Par_path is the path to the parameter file, including filename.txt
    The parameters need to be given as integers.
    If no value for a parameter is given, the default value is used.
    """

    # If parameter does not have the default value, set "to adjust" to True.
    adj_nrRes = False
    adj_penalty = False
    adj_finestRes = False
    # Default values
    def_nrRes = 3
    def_penalty = 0.000001
    def_finestRes = 16
    # Check if it needs to be adjusted
    if nrRes != def_nrRes:
        adj_nrRes = True
    if penalty != def_penalty:
        adj_penalty = True
    if finestRes!= def_finestRes:
        adj_finestRes = True

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
        if adj_finestRes == True:
            if "FinalGridSpacingInPhysicalUnits" in line:
                line = line.replace(str(def_finestRes), str(finestRes))  
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



def bestInitialization(methods, pnr, ELASTIX_PATH, fixed_path, moving_path_i, parameter_file_dir):
    """
    This function determines which initialization method does result in the best registration. 

    """
    
    MI_values = np.zeros(len(methods))
    for idx, method in enumerate(methods):
        print('Test initialization method {}'.format(method))
        if method == 'none':
            # pass initialization if method is none
            moving_path = moving_path_i
        elif method == 'rigid_affine':
            # first perform rigid registration
            parameters_init_r = '{}/parameters_rigid.txt'.format(parameter_file_dir)
            dir_res_init_reg = Initialization_registration(fixed_path, moving_path_i, parameters_init_r, ELASTIX_PATH, pnr, 'RA_rigid')
            
            # Load the fixed and initialized registrated atlas image
            initialized_image_r = sitk.ReadImage(os.path.join(dir_res_init_reg, 'result.0.mhd'))
            image_array_s = sitk.GetArrayFromImage(initialized_image_r)
            
            # determine new moving path for affine registration
            moving_path_r = '{}/result.0.mhd'.format(dir_res_init_reg)
            
            # secondly, perform affine registration
            parameters_init_RA = '{}/parameters_affine.txt'.format(parameter_file_dir)
            dir_res_init_RA = Initialization_registration(fixed_path, moving_path_r, parameters_init_RA, ELASTIX_PATH, pnr, 'RA_affine')
            
            # Load the fixed and initialized registrated atlas image
            initialized_image_RA = sitk.ReadImage(os.path.join(dir_res_init_RA, 'result.0.mhd'))
            image_array_s = sitk.GetArrayFromImage(initialized_image_RA)
            
            # determine new moving path for affine registration
            moving_path = '{}/result.0.mhd'.format(dir_res_init_RA)
        
        else:
            # select the right parameterfile for initialization
            parameters_initialization = '{}/parameters_{}.txt'.format(parameter_file_dir, method)
            
            # Registrate for initialization with the correct parameter file
            dir_res_init = Initialization_registration(fixed_path, moving_path_i, parameters_initialization, ELASTIX_PATH, pnr, method)
        
            # Load the fixed and initialized registrated atlas image
            initialized_image = sitk.ReadImage(os.path.join(dir_res_init, 'result.0.mhd'))
            image_array_s = sitk.GetArrayFromImage(initialized_image)
            
            # determine new moving path for nonrigid registration
            moving_path = '{}/result.0.mhd'.format(dir_res_init)
            
    
        # Registrate with nonrigid BSpline transform (actual registration)
        parameter_path = '{}/parameters.txt'.format(parameter_file_dir)
        
        # IVANA: changing the working directory where the python file is (comment out if it errors for you)
        #os.chdir(os.path.dirname(sys.argv[0]))
        #print("fixed path best initialization:", fixed_path)
        
        # Continue Bspline registration and load the images
        dir_res = Registration(fixed_path, moving_path, parameter_path, ELASTIX_PATH, pnr)
        f_image = sitk.ReadImage(fixed_path)
        fixed_image_s = sitk.GetArrayFromImage(f_image)
        registered_image_path = sitk.ReadImage(os.path.join(dir_res, 'result.0.mhd'))
        image_registered = sitk.GetArrayFromImage(registered_image_path)
        
        # Calculate the mutual information value
        MI_value = mutual_information(fixed_image_s, image_registered)
        MI_values[idx] = MI_value
    
    # Pick the initialization method with the highest MI_value
    print(MI_values)
    idx_max = np.argmax(MI_values)
    print("idx_max = ", idx_max)
    # best_method = methods[idx_max]
    return idx_max
        
def bestFinestResolution(FR_values, pnr, ELASTIX_PATH, fixed_path, moving_path, parameter_file_path):
    """
    This function determines which finest resolution gives the best registration, in order to optimize parameters. 

    Returns
    -------
    None.

    """
    MI_values = np.zeros(len(FR_values))
    for idx in range(len(FR_values)):
        FR_value = FR_values[idx]
        
        # Adjust the value for finest resolution  
        adj_par_path = AdjustParameters(parameter_file_path, finestRes=FR_value)
        
        # Registrate with the adjusted parameter file
        dir_res = Registration(fixed_path, moving_path, adj_par_path, ELASTIX_PATH, pnr)
    
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
    best_finestRes = FR_values[idx_max]
    print(MI_values)
    return best_finestRes

def bestPenalty(penalty_w, pnr, ELASTIX_PATH, fixed_path, moving_path, parameter_file_path):
    """ penalty_w is a list with the penalty weights you want to test as integers. It automatically first adjusts the 
    parameter file to the wanted penalty weight, then registrates with that parameter file and lastly calculates the mutual
    information value for that registration. The function returns the index of the best penalty value for this patient.
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

    # Pick the index of the penalty weight with the highest MI_value
    best_penalty_idx = np.argmax(MI_values)
    print("idx_max = ", best_penalty_idx)
    #best_penalty = penalty_w[idx_max]
    print(MI_values)
    return best_penalty_idx

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
        dir_res = Registration(fixed_path, moving_path, adj_par_path, ELASTIX_PATH, pnr)
    
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