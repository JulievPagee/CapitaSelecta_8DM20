#option 1
#ref: https://stackoverflow.com/questions/31573872/how-to-read-multiple-images-and-create-a-3d-matrix-with-them
#arrays = []
#for number in range(0, 299):
#    numstr = str(number).zfill(3)
 #   fname = numstr + '.bmp'
 #   a = imread(fname, flatten=1)
 #   arrays.append(a)
#data = np.array(arrays)
#dit lijkt me vrij omslachtig, np.concatenate doet ook de job.

#option 2
# https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.dstack.html
#a = figure1   #np.array format
#b = figure2   #np.array format
#np.concatenate((a, b), axis=0)  #stackt ze in verticale richting (links =1, rechts =2)
#np.concatenate((a, b.), axis=1) #stackt ze in horizontale richting
#np.concatenate((a,b) axis=2)    #axis 2 obv bovenste redenatie
#option 3
#https://nilearn.github.io/modules/generated/nilearn.image.concat_imgs.html



#doel (333,271,86) array maken.
#option 2 lijkt me het handiste
#data_list = [ 'insert datanames here']
#save_path

def 2D_to_3D(data_list, save_path)
#data_list = [ 'insert datanames here']
#save_path meegeven
    final_image = np.array([])
    firstIteration = True
    for item in data_list:
        if firstIteration:
            final_image = item
            firstIteration=False
        else:
            final_image = np.concatenate(([item , final_image ]), axis=2)

    final_image= Image.fromarray(final_image)
    final_image.save(save_path)