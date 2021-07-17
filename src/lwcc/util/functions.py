from pathlib import Path
import gdown
import os

def build_url(path):
    url = "https://github.com/tersekmatija/lwcc/releases/download/v0.1/{}".format(
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