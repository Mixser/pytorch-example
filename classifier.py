from io import BytesIO

import torch
from torchvision.transforms import transforms
from PIL import Image


class ImageClassifier(object):
    def __init__(self, model, classes_map):
        self._model = model
        self._classes_map = classes_map

    @classmethod
    def _process_image(cls, img):

        normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

        transformation = transforms.Compose([
            transforms.Resize(400),  # this need for really big images
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalization
        ])

        img_tensor = transformation(img)

        return img_tensor.unsqueeze(0).float()

    def _get_class(self, index):
        index = str(index)

        return self._classes_map.get(index, None)

    def _classify(self, img):

        img_tensor = self._process_image(img)

        output = self._model.forward(img_tensor)

        output = torch.exp(output)

        probs, classes = output.topk(1, dim=1)

        return probs.item(), self._get_class(classes.item())

    def classify(self, image_path):
        if isinstance(image_path, str):
            img = Image.open(image_path)
        elif isinstance(image_path, (bytes, bytearray)):
            img = Image.open(BytesIO(image_path))
        else:
            raise ValueError("Invalid image path")

        return self._classify(img)

