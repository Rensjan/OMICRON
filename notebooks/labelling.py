import sys
import os
import glob

sys.path.insert(1, os.path.join("..", "data")) # access to data folder
sys.path.insert(1, os.path.join("..", "utils"))# access to utils folder

from label_utils import image_analysis

# image_path = glob.glob(os.path.join(image_name))
image_path = "../data/png_unfiltered/51689389222_7e91bc6cf8_o'_half_2.png"


image = image_analysis(image_path)
# equalized_tensor = image.plotTensorEqualized()

svd_tensor = image.svd()
image.plotTensor(svd_tensor)




# tensor = image.plotTensor()
# image.svd(tensor)


#image.RGB_distribution(tensor)

# image.plotImage_url()



