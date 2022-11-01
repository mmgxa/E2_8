import pytest
import requests
import os

@pytest.mark.parametrize("img", ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"])
def test_cifar10_http(img):
    serve = os.getenv("serve_host", "localhost")

    res = requests.post(f"http://{serve}:8080/predictions/cifar10/1.0", files={'data': open(f'./test_serve/{img}.jpg', 'rb')})
    data = res.json()
    assert max(data, key=data.get) == img