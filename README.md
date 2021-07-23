# LWCC: A LightWeight Crowd Counting library for Python

![](https://raw.githubusercontent.com/tersekmatija/lwcc/master/imgs/lwcc_header_gif.gif)

![](https://img.shields.io/badge/state-of%20the%20art-orange) ![](https://img.shields.io/github/license/tersekmatija/lwcc?label=license)

LWCC is a lightweight crowd counting framework for Python. It wraps four state-of-the-art models all based on convolutional neural networks: [`CSRNet`](https://github.com/leeyeehoo/CSRNet-pytorch), [`Bayesian crowd counting`](https://github.com/ZhihengCV/Bayesian-Crowd-Counting), [`DM-Count`](https://github.com/cvlab-stonybrook/DM-Count), and [`SFANet`](https://github.com/pxq0312/SFANet-crowd-counting). The library is based on PyTorch.

## Installation

The easiest way to install library LWCC and its prerequisites is to use the package manager [pip](https://pip.pypa.io/en/stable/). 

```python
pip install lwcc
```

## Usage
You can import the library and use its functionalities by:

```python
from lwcc import LWCC
```
#### Count estimation
Most straightforward way to use the library:
```python
img = "path/to/image"
count = LWCC.get_count(img)
```
This uses CSRNet pretrained on SHA (default). You can choose a different model pretrained on different data set using:
```python
count = LWCC.get_count(img, model_name = "DM-Count", model_weights = "SHB")
```
The result is a float with predicted count.

#### Multiple images
Library allows prediction of count for multiple images with a single call of *get_count*.
You can simply pass a list of image paths:

```python
img1 = "path/to/image1"
img2 = "path/to/image2"
count = LWCC.get_count([img1, img2])
```

Result is then a dictionary of pairs *image_name : image_count*:
![result](https://raw.githubusercontent.com/tersekmatija/lwcc/master/imgs/result.png)

#### Density map
You can also request a density map by setting flag *return_density = True*. The result is then a tuple *(count, density_map)*, where *density_map* is a 2d array with predicted densities. The array is smaller than the input image and its size depends on the model. 

```python
import matplotlib.pyplot as plt

count, density = LWCC.get_count(img, return_density = True)

plt.imshow(density)
plt.show()
```
![result_density](https://raw.githubusercontent.com/tersekmatija/lwcc/master/imgs/result_density.png)

This also works for multiple images (list of image paths as input). Result is then a tuple of two dictionaries, where the first dictionary is the same as above (pairs of *image_name : image_count*) and the second dictionary contains pairs of *image_name : density_map*.

#### Loading the model
You can also directly access the PyTorch models by loading them first with the *load_model* method. 
```python
model = LWCC.load_model(model_name = "DM-Count", model_weights = "SHA")
```
The loaded *model* is a PyTorch model and you can access its weights as with any other PyTorch model.

You can use it for inference as: 
```python
 count = LWCC.get_count(img, model = model)
```

## Models

LWCC currently offers 4 models (CSRNet, Bayesian crowd counting, DM-Count, SFANet) pretrained on [Shanghai A](https://ieeexplore.ieee.org/document/7780439), [Shanghai B](https://ieeexplore.ieee.org/document/7780439), and [UCF-QNRF](https://www.crcv.ucf.edu/data/ucf-qnrf/) datasets. The following table shows the model name and MAE / MSE result of the available pretrained models on the test sets. 

|   Model name |      SHA       |      SHB      |      QNRF       |
| -----------: | :------------: | :-----------: | :-------------: |
|   **CSRNet** | 75.44 / 113.55 | 11.27 / 19.32 | *Not available* |
|      **Bay** | 66.92 / 112.07 | 8.27 / 13.56  | 90.43 / 161.41  |
| **DM-Count** | 61.39 / 98.56  | 7.68 / 12.66  | 88.97 / 154.11  |
|   **SFANet** |*Not available* | 7.05 / 12.18  | *Not available* |

Valid options for *model_name* are written in the first column and thus include: `CSRNet`, `Bay`, `DM-Count`, and `SFANet`.
Valid options for *model_weights* are written in the first row and thus include: `SHA`, `SHB`,  and `QNRF`.

**Note**: Not all *model_weights* are supported with all *model_names*. See the above table for possible combinations.

## FAQ - Frequently asked questions

#### Is GPU support available?
No, GPU support is currently not supported yet, but is planned for the future version.

#### Can I load custom weights?
Full support of loading custom pretrained weights is not supported, but is planned in the future version.

#### Can I train the models myself?
The library does not support training, only inference.

## Support
If you like the library please show us your support by ⭐️ starring the project!

## Citation
Although the paper has not been published yet, please provide the link to this GitHub repository if you use LWCC in your research.

## License
This library is licensed under MIT license (see [LICENSE](https://github.com/tersekmatija/lwcc/blob/master/LICENSE)). Licenses of the models wrapped in the library will be inherited, depending on the model you use ( [`CSRNet`](https://github.com/leeyeehoo/CSRNet-pytorch), [`Bayesian crowd counting`](https://github.com/ZhihengCV/Bayesian-Crowd-Counting), [`DM-Count`](https://github.com/cvlab-stonybrook/DM-Count), and [`SFANet`](https://github.com/pxq0312/SFANet-crowd-counting)).

