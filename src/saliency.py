# -*- coding: utf-8 -*-
"""
THis script was not used and is not finished.
We tempted saliency map. 
"""
from tensorflow.keras.models import load_model
from tensorflow.keras.initializers import glorot_uniform
from vis.utils import utils
from tensorflow.keras import activations
import matplotlib.image as mpimg
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt

"""
in vis/visualization/saliency.py

replace: from scipy.misc import imresize

to: import cv2

then

replace: heatmap = imresize(heatmap, input_dims, interp='bicubic', mode='F')

to:

heatmap = cv2.resize(src=heatmap,
                       dsize=input_dims,
                       interpolation=cv2.INTER_CUBIC)
I just couldn't map the mode param.
"""
"The above fix, OR downgrading scipy to version 1.1.0 should make visualize_saliency import work"
from vis.visualization import visualize_saliency

def saliency_map(model_path):
    model_path = "../../uMultiClass/uMultiClass.h5"
    """
    https://opendatascience.com/visualizing-your-convolutional-neural-network-predictions-with-saliency-maps/
    """
    model = load_model(model_path, compile=False)
    img = mpimg.imread("F:\\CheXpert-v1.0-small\\valid\\patient64541\study1\\view1_frontal.jpg")
    layer_idx = utils.find_layer_idx(model, "fc_out")
    model.layers[layer_idx].activation = activations.linear
    #For below function, you must go into apply_modifictions and change
    #from keras.models import load_model to from tensorflow.keras.models import load_model
    model = utils.apply_modifications(model)

    #This function needs so much memory to run, with 16b I can't even run it
    #without crashing
    grads = visualize_saliency(model, layer_idx, filter_indices=None,
                               seed_input = img, backprop_modifier = None,
                               grad_modifier = "absolute")
    plt.imshow(grads, alpha=.6)
    gaus = ndimage.gaussian_filter(grads[:,:,2], sigma=5)
    plt.imshow(img)
    plt.imshow(gaus, alpha=.7)
