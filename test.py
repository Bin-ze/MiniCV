import sys
import json
import torch
import logging
import argparse

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
    parser.add_argument("--config", type=str, default='cla_config/mobilenet.json')
    parser.add_argument("--model_path", type=str, default='runs/MobileNetV2/best.pth')
    parser.add_argument("--img_path", type=str, default='/mnt/data/beijing_daima/0724_liutao_singledish/0302_yi/val_after_5/1')
    args = parser.parse_args()

    config = Parse_config(args.config)
    logging.info(config)


    # instance model
    model = build_model(config['model'])
    model.load_state_dict(torch.load(args.model_path))

    # 初始化Trainer
    Tester = Tester(model=model, config=config)

    Tester.run(args.img_path)