from lwcc import LWCC
import matplotlib.pyplot as plt

# Image paths
img1 = "dataset/img01.jpg"
img2 = "dataset/img02.jpg"

# Initialize model and predict
"""
model = LWCC.load_model(model_weights= "SHA")
count = LWCC.get_count(img2, model = model)
print(f"Count for img1: {count}")
"""

# Direct
"""
count, density = LWCC.get_count(img2, model_name="Bay", return_density=True)
#print(f"Count for img1: {count}")
print(density.shape)
import matplotlib.pyplot as plt
fig = plt.figure()
plt.imshow(density)
plt.axis('off')
plt.show()
fig.savefig('out.png', bbox_inches='tight', pad_inches=0)
"""

# Test multiple
"""
counts = LWCC.get_count([img1, img2], model_name= "DM-Count", model_weights="SHB")
print(f"Counts for img1, img2: {counts}")

counts = LWCC.get_count([img1, img2], model_name= "DM-Count", model_weights="SHA")
print(f"Counts for img1, img2: {counts}")

counts = LWCC.get_count([img1, img2], model_name= "DM-Count", model_weights="QNRF")
print(f"Counts for img1, img2: {counts}")
"""

# Test density map
"""
counts, density = LWCC.get_count(img1, model_name="Bay", model_weights="SHB", return_density=True)
print(f"Count: {counts}")
plt.imshow(density)

from lwcc.util import functions
from PIL import Image
import numpy as np

fig = plt.figure()

img, name = functions.load_image(img1, "none")
implot = plt.imshow(img[0,0,:,:], origin='upper')
size = list(img.size())
density = Image.fromarray(np.uint8(density * 255) , 'L')
density = density.resize((size[3], size[2]), Image.BILINEAR)


plt.imshow(density, alpha=.5,origin='upper', cmap= plt.get_cmap("plasma"))
plt.legend('',frameon=False)
plt.axis('off')

fig.set_size_inches(15, 8.44)
plt.show()
fig.savefig('out.png', bbox_inches='tight', pad_inches=0)
"""

# Density multiple
"""
counts, densities = LWCC.get_count([img1, img2], model_name="DM-Count", model_weights="SHB",
                                return_density=True)
print(counts)
print(densities)
plt.imshow(densities["img01"])
plt.show()
plt.imshow(densities["img02"])
plt.show()

print(densities["img01"].shape)
"""
