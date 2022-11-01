import pytest
import subprocess
import sys
import os.path
import json

@pytest.mark.parametrize("img", ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"])
def test_cifar10_grpc(img):

    x = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "serve/ts_scripts/torchserve_grpc_client.py"), "infer",  "cifar10", os.path.join(os.path.dirname(__file__), f"{img}.jpg")], shell=False, capture_output=True)
    xo = x.stdout
    t = xo.decode("utf-8")
    res = json.loads(t)
    assert max(res, key=res.get) == img