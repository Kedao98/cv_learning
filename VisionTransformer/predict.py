import json
import os

import cv2
import torch
from torchvision import transforms

# device info
DEVICE = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
print(f"using {DEVICE} device.")

DATA_TRANSFORM = transforms.Compose([transforms.ToPILImage(),
                                     transforms.Resize((224, 224)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


def read_image(image_path):
    assert os.path.exists(image_path), f"file: {image_path} does not exist"
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = DATA_TRANSFORM(image)
    image = torch.unsqueeze(image, dim=0)
    return image


def predict(image_path):
    image = read_image(image_path)
    with torch.no_grad():
        # predict class
        output = torch.squeeze(MODEL(image.to(DEVICE))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_class = torch.argmax(predict).numpy()

    print(f"class: {CLASS_DICT[str(predict_class)]:10}   prob: {predict[predict_class]:.3}")


if __name__ == '__main__':
    from model import vit_base_patch16_224_in21k as create_model

    MODEL, WEIGHTS_PATH, CLASS_DICT_PATH = create_model(num_classes=5, has_logits=False)

    # load model
    assert WEIGHTS_PATH, f"file: {WEIGHTS_PATH} does not exist"
    MODEL = MODEL.to(DEVICE)
    MODEL.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    MODEL.eval()

    # data info
    with open(CLASS_DICT_PATH, "r") as f:
        CLASS_DICT = json.load(f)
    predict("dataset/test.png")
