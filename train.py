import os
import json
import argparse
import logging
import sys
import torch

import torch.optim as optim

from torchvision import transforms, datasets
from minicla.builder import build_model
from minicla.apis.train import Trainer



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
    args = parser.parse_args()

    config = Parse_config(args.config)
    logging.info(config)



    # instance dataset
    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),  # cannot 224, must (224, 224)
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    train_dataset = datasets.ImageFolder(root=os.path.join(config["dataset_path"], "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)



    cla_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in cla_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)

    save_path = f'{config["save_path"]}//{config["model_name"]}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(f'{save_path}/class_indices.json', 'w') as json_file:
        json_file.write(json_str)


    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config["batch_size"], shuffle=True,
                                               num_workers=config["num_worker"])

    validate_dataset = datasets.ImageFolder(root=os.path.join(config["dataset_path"], "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    val_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=4, shuffle=False,
                                                  num_workers=config["num_worker"])


    # instance model
    model = build_model(config['model'])

    # build optimizer
    optimizer = optim.Adam(model.parameters(),config["lr"])

    # 初始化Trainer
    Trainer = Trainer(model=model, optimizer=optimizer, dataloder=[train_loader, val_loader],
                      train_num=train_num, val_num=val_num, save_path=save_path, config=config)

    Trainer.run()














