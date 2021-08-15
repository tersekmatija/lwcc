from .models import CSRNet, SFANet, Bay, DMCount
from .util.functions import load_image

import torch


def load_model(model_name="CSRNet", model_weights="SHA"):
    """
    Builds a model for Crowd Counting and initializes it as a singleton.
    :param model_name: One of the available models: CSRNet.
    :param model_weights: Name of the dataset the model was pretrained on. Possible values vary on the model.
    :return: Built Crowd Counting model initialized with pretrained weights.
    """

    available_models = {
        'CSRNet': CSRNet,
        'SFANet': SFANet,
        'Bay': Bay,
        'DM-Count': DMCount
    }

    global loaded_models

    if "loaded_models" not in globals():
        loaded_models = {}

    model_full_name = "{}_{}".format(model_name, model_weights)
    if model_full_name not in loaded_models.keys():
        model = available_models.get(model_name)
        if model:
            model = model.make_model(model_weights)
            loaded_models[model_full_name] = model
            print("Built model {} with weights {}".format(model_name, model_weights))
        else:
            raise ValueError("Invalid model_name. Model {} is not available.".format(model_name))

    return loaded_models[model_full_name]


def get_count(img_paths, model_name="CSRNet", model_weights="SHA", model=None, is_gray=False, return_density=False,
              resize_img = True):
    """
    Return the count on image/s. You can use already loaded model or choose the name and pre-trained weights.
    :param img_paths: Either String (path to the image) or a list of strings (paths).
    :param model_name: If not using preloaded model, choose the model name. Default: "CSRNet".
    :param model_weights: If not using preloaded model, choose the model weights.  Default: "SHA".
    :param model: Possible preloaded model. Default: None.
    :param is_gray: Are the input images grayscale? Default: False.
    :param return_density: Return the predicted density maps for input? Default: False.
    :param resize_img: Should images with high resolution be down-scaled? This is especially good for high resolution
            images with relatively few people. For very dense crowds, False is recommended. Default: True
    :return: Depends on whether the input is a String or list and on the return_density flag.
        If input is a String, the output is a float with the predicted count.
        If input is a list, the output is a dictionary with image names as keys, and predicted counts (float) as values.
        If return_density is True, function returns a tuple (predicted_count, density_map).
        If return_density is True and input is a list, function returns a tuple (count_dictionary, density_dictionary).
    """

    # if one path to array
    if type(img_paths) != list:
        img_paths = [img_paths]

    # load model
    if model is None:
        model = load_model(model_name, model_weights)

    # load images
    imgs, names = [], []

    for img_path in img_paths:
        img, name = load_image(img_path, model.get_name(), is_gray, resize_img)
        imgs.append(img)
        names.append(name)

    imgs = torch.cat(imgs)

    with torch.set_grad_enabled(False):
        outputs = model(imgs)

    counts = torch.sum(outputs, (1, 2, 3)).numpy()
    counts = dict(zip(names, counts))

    densities = dict(zip(names, outputs[:, 0, :, :].numpy()))


    if len(counts) == 1:
        if return_density:
            return counts[name], densities[name]
        else:
            return counts[name]

    if return_density:
        return counts, densities

    return counts
