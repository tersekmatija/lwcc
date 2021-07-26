from pathlib import Path
import gdown
import os

from torchvision import transforms
from PIL import Image

def build_url(path):
    url = "https://github.com/tersekmatija/lwcc_weights/releases/download/v0.1/{}".format(
        path
    )

    return url

def weights_check(model_name, model_weights):
    # create dir if does not exists
    Path("/.lwcc/weights").mkdir(parents=True, exist_ok=True)

    # download weights if not available
    home = str(Path.home())

    file_name = "{}_{}.pth".format(model_name, model_weights)
    url = build_url(file_name)
    output = os.path.join(home, "/.lwcc/weights/", file_name)
    print(output)

    if not os.path.isfile(output):
        print(file_name, " will be downloaded to ", output)
        gdown.download(url, output, quiet=False)

    return output

def load_image(img_path, model_name, is_gray=False, resize_img = True):
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

    # resize image
    if resize_img:
        long = max(img.size[0], img.size[1])
        factor = 1000 / long
        img = img.resize((int(img.size[0] * factor), int(img.size[1] * factor)),
                         Image.BILINEAR)

    # different preprocessing for SFANet
    if model_name == "SFANet":
        height, width = img.size[1], img.size[0]
        height = round(height / 16) * 16
        width = round(width / 16) * 16
        img = img.resize((width, height), Image.BILINEAR)


    img = trans(img)
    img = img.unsqueeze(0)

    name = os.path.basename(img_path).split('.')[0]

    return img, name
