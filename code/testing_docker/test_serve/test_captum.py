import numpy as np
import torchvision.transforms as T
import requests
from matplotlib.colors import LinearSegmentedColormap
import os

from PIL import Image

from captum.attr import visualization as viz

def test_cifar10_http():
    serve = os.getenv("serve_host", "localhost")
    res = requests.post(f"http://{serve}:8080/explanations/cifar10/1.0", files={'data': open(f'./test_serve/cat.jpg', 'rb')})
    nt = res.json()
    img_path = "./test_serve/cat.jpg"

    inp_image = Image.open(img_path)
    to_tensor = T.Compose([
        T.Resize((32, 32)),
        T.ToTensor()
    ])
    inp_image = to_tensor(inp_image)

    inp_image = inp_image.numpy()
    attributions = np.array(nt)

    inp_image, attributions = inp_image.transpose(1, 2, 0), attributions.transpose(1, 2, 0)
    default_cmap = LinearSegmentedColormap.from_list('custom blue', 
                                                    [(0, '#ffffff'),
                                                    (0.25, '#000000'),
                                                    (1, '#000000')], N=256)
    b = viz.visualize_image_attr_multiple(attributions,
                                inp_image,
                                ["original_image", "heat_map"],
                                ["all", "positive"],
                                cmap=default_cmap,
                                show_colorbar=True,
                                use_pyplot=False)
    b[0].savefig('b_integ_grad_noise.png') 
    assert inp_image.shape == attributions.shape