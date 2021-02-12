# Optimization of parameters
# BELANGRIJK: we doen nu nog mutual information berekenen op slice 20, niet de hele image!
from Reg_functions import *
import SimpleITK as sitk

# Adjust to your directory!
ELASTIX_PATH = os.path.join("C:/Users/20175722/Documents/Master jaar 1/Kwartiel 3/8DM20 - CS/elastix-5.0.0-win64/elastix.exe") 
TRANSFORMIX_PATH = os.path.join("C:/Users/20175722/Documents/Master jaar 1/Kwartiel 3/8DM20 - CS/elastix-5.0.0-win64/transformix.exe")
fixed_path = 'NormData/norm_img_p102.mhd'     
moving_path = 'NormData/norm_img_p107.mhd'
parameter_file_path = 'New code Myrthe/parameters.txt'



# Test to adjust the parameter file with resolutions 1 to 6
ResValues = [1,2,3] #[1,2,3,4,5,6]

# Determine the best resolution with one atlas on all 7 train images
pnr_list = ['p102', 'p109'] #['p102', 'p109', 'p115', 'p117', 'p120', 'p128', 'p129']  # This is now manual, can be automated if needed/wanted
best_res_patient = np.zeros(len(pnr_list), dtype=int)
for idx in range(len(pnr_list)):
    pnr = pnr_list[idx]
    best_res_patient[idx] = bestResolution(ResValues, pnr, ELASTIX_PATH, fixed_path, moving_path, parameter_file_path)
print(best_res_patient)

# Majority voting
final_res = np.bincount(best_res_patient).argmax()
print(final_res)
print("done")

# To DO: Make txt file with best parameters

