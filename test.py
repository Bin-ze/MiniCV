import os
import json
import argparse
import logging
import sys
import torch

from minicla.builder import build_model
from minicla.apis.test import Tester



def Parse_config(config):
    with open(config, 'r') as f:
        config = json.load(f)

    config['model_name'] = config["model"]["object"]

    return config


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        stream=sys.stdout,
                        format="%(asctime)s | %(filename)s:%(lineno)d | %(levelname)s | %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='cla_config/alexnet.json')
    parser.add_argument("--model_path", type=str, default='runs/AlexNet/best.pth')
    parser.add_argument("--img_path", type=str, default='/mnt/c/Users/Bin-ze/Desktop/food_fake_real_classification_清洗/val/3/0001.jpg')
    args = parser.parse_args()

    config = Parse_config(args.config)
    logging.info(config)


    # instance model
    model = build_model(config['model'])
    model.load_state_dict(torch.load(args.model_path))

    # 初始化Trainer
    Tester = Tester(model=model, config=config)

    Tester.run(args.img_path)














