from create_segmentations_util import *

model_name = 'alltrain'
name = "Nathalie"

if name == "GPU":
    CLASS_DATA_PATH = '/home/8dm20-4/ClassData'
    MODEL_SAVE_PATH = '/home/8dm20-4/Models'
    output_dir = '/home/8dm20-4/Output'

if name == "Nathalie":
    CLASS_DATA_PATH = r'C:\Nathalie\Tue\Master\Jaar 1\Q3\Capita Selecta\Project\Codes_def\ClassData'    #Data folder
    MODEL_SAVE_PATH = r'C:\Nathalie\Tue\Master\Jaar 1\Q3\Capita Selecta\Project\Models'                        #Define where the model was saved
    output_dir = r'C:\Nathalie\Tue\Master\Jaar 1\Q3\Capita Selecta\Project\Output'

#paths
image_dir_validatiom = os.path.join(CLASS_DATA_PATH, 'Validation/Slices')                       #path to validation images
model_savefile = model_name +'.sav'
output_model = os.path.join(MODEL_SAVE_PATH, model_savefile)                                #specify path where model was saved

main(image_dir_validatiom, output_model, output_dir)
