#Registration functions
import elastix
import os
import numpy as np


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