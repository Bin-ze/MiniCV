import torch
import logging

import torch.nn as nn


class Validator:
    def __init__(self, model, dataloder, val_num, config):

        self.config = config

        # 解析配置
        self.device = self.config['device']

        self.val_num = val_num

        self.val_loader = dataloder

        self.model = model.to(self.device)


    @torch.no_grad()
    def validate(self):

            acc = 0.0  # accumulate accurate number / epoch
            with torch.no_grad():
                val_bar = self.val_loader
                for step, val_data in enumerate(val_bar):
                    val_images, val_labels = val_data
                    outputs = self.model(val_images.to(self.device))
                    predict_y = torch.max(outputs, dim=1)[1]
                    acc += torch.eq(predict_y, val_labels.to(self.device)).sum().item()

            val_accurate = acc / self.val_num

            logging.info('validation')
            logging.info('val_accuracy: %.3f' %(val_accurate))


    def run(self):

        self.validate()
        return





