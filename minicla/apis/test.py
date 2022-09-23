import torch
import logging
import os
import json

from PIL import Image
from torchvision import transforms

class Tester:
    def __init__(self, model, config):

        self.config = config

        # 解析配置
        self.device = self.config['device']

        cla_dict = open(config["save_path"] + '/' + config["model_name"] + '/class_indices.json')
        self.class_dict = json.load(cla_dict)
        cla_dict.close()

        self.model = model.to(self.device)

        self.data_transform = transforms.Compose(
            [transforms.Resize(256),
             transforms.CenterCrop(224),
             transforms.ToTensor(),
             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def preprocessing(self, img_path):

        img = Image.open(img_path)
        img = self.data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        return img

    @torch.no_grad()
    def test(self, img):

        output = torch.squeeze(self.model(img.to(self.device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

        print_res = "class: {}   prob: {:.3}".format(self.class_dict[str(predict_cla)],
                                                 predict[predict_cla].numpy())

        logging.info(print_res)


    def run(self, img_path):

        if os.path.isdir(img_path):
            for img in os.listdir(img_path):
                img = os.path.join(img_path, img)
                img = self.preprocessing(img)
                self.test(img)
        else:
            img = self.preprocessing(img_path)
            self.test(img)

        return





