from src.lwcc import LWCC
import matplotlib.pyplot as plt

# Image paths
img1 = "dataset/img01.jpg"
img2 = "dataset/img02.jpg"

# Initialize model and predict
#model = LWCC.load_model(model_weights= "SHA")
#count = LWCC.get_count(img2, model = model)
#print(f"Count for img1: {count}")

# Direct
#count = LWCC.get_count(img2)
#print(f"Count for img1: {count}")

# Test multiple
#counts = LWCC.get_count([img1, img2], model_name= "SFANet", model_weights="SHA")
#print(f"Counts for img1, img2: {counts}")

#counts = LWCC.get_count([img1, img2], model_name= "DM-Count", model_weights="SHA")
#print(f"Counts for img1, img2: {counts}")

#counts = LWCC.get_count([img1, img2], model_name= "CSRNet", model_weights="SHA")
#print(f"Counts for img1, img2: {counts}")

# Test density map
counts, density = LWCC.get_count(img2, model_name="DM-Count", model_weights="SHA", return_density=True)
print(f"Count: {counts}")
plt.imshow(density)

from src.lwcc.util import functions
from PIL import Image
import numpy as np
img, name = functions.load_image(img2, "none")
implot = plt.imshow(img[0,0,:,:], origin='upper')
size = list(img.size())
density = Image.fromarray(np.uint8(density * 255) , 'L')
density = density.resize((size[3], size[2]), Image.BILINEAR)

plt.imshow(density, alpha=.5,origin='upper', cmap= plt.get_cmap("plasma"))
plt.show()