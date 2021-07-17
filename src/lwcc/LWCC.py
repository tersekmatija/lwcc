from .models import CSRNet, SFANet

import os

import torch
from torchvision import transforms
from PIL import Image


def load_model(model_name = "CSRNet", model_weights = "SHA"):
    """
    Builds a model for Crowd Counting and initializes it as a singleton.
    :param model_name: One of the available models: CSRNet.
    :param model_weights: Name of the dataset the model was pretrained on. Possible values vary on the model.
    :return: Built Crowd Counting model initialized with pretrained weights.
    """

    available_models = {
        'CSRNet': CSRNet,
        'SFANet': SFANet
    }

    global loaded_models

    if not "loaded_models" in globals():
        loaded_models = {}

    if not model_name in loaded_models.keys():
        model = available_models.get(model_name)
        if model:
            model = model.make_model(model_weights)
            loaded_models[model_name] = model
            print("Built model {} with weights {}".format(model_name, model_weights))
        else:
            raise ValueError("Invalid model_name. Model {} is not available.".format(model_name))

    return loaded_models[model_name]


def get_count(img_paths, model_name="CSRNet", model_weights="SHA", model=None, is_gray=False, return_density = False):
    # if one path to array
    if type(img_paths) != list:
        img_paths = [img_paths]

    # load model
    if model is None:
        model = load_model(model_name, model_weights)

    # get counts
    counts = {}
    densities = {}
    for img_path in img_paths:
        img, name = load_image(img_path, is_gray)

        with torch.set_grad_enabled(False):
            output = model(img)
            count = torch.sum(output).item()
        counts[name] = count

        if return_density:
            densities[name] = output

    if len(counts) == 1:
        if return_density:
            return counts[name], densities[name]
        else:
            return counts[name]

    if return_density:
        return counts, densities

    return counts


def load_image(img_path, is_gray=False):
    if not os.path.isfile(img_path):
        raise ValueError("Confirm that {} exists".format(img_path))

    # set transform
    if is_gray:
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    else:
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    # preprocess image
    img = Image.open(img_path).convert('RGB')

    # FOR SFANET
    height, width = img.size[1], img.size[0]
    height = round(height / 16) * 16
    width = round(width / 16) * 16
    img = img.resize((width, height), Image.BILINEAR)
    ###

    img = trans(img)
    img = img.unsqueeze(0)

    name = os.path.basename(img_path).split('.')[0]

    return img, name
