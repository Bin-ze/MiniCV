import os
import json
import argparse
import logging
import sys
import torch

from minicla.builder import build_model
from torchvision import transforms, datasets
from minicla.apis.val import Validator



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
    args = parser.parse_args()

    config = Parse_config(args.config)
    logging.info(config)

    # instance dataset
    data_transform = {
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}


    validate_dataset = datasets.ImageFolder(root=os.path.join(config["dataset_path"], "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    val_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=4, shuffle=False,
                                                  num_workers=config["num_worker"])

    # instance model
    model = build_model(config['model'])
    model.load_state_dict(torch.load(args.model_path))

    # 初始化Trainer
    Validator = Validator(model=model, dataloder=val_loader, val_num=val_num, config=config)

    Validator.run()