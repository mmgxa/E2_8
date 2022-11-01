import torch
import torch.nn.functional as F
import torchvision.transforms as T

from ts.torch_handler.vision_handler import VisionHandler
from ts.utils.util import map_class_to_label

class ImageClassifier(VisionHandler):
    """
    ImageClassifier handler class. This handler takes an image
    and returns the name of object in that image.
    """
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2471, 0.2435, 0.2616)
    
    topk = 5
    # These are the standard Imagenet dimensions
    # and statistics
    image_processing = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )

    def set_max_result_classes(self, topk):
        self.topk = topk

    def get_max_result_classes(self):
        return self.topk

    def postprocess(self, data):
        ps = F.softmax(data, dim=-1)
        probs, classes = torch.topk(ps, self.topk, dim=1)
        probs = probs.tolist()
        classes = classes.tolist()
        return map_class_to_label(probs, self.mapping, classes)