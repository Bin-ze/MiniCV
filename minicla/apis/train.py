import torch
import logging

import torch.nn as nn

class Trainer:
    def __init__(self, model, optimizer, dataloder, train_num, val_num,save_path, config):

        self.config = config
        # build loss
        self.loss = nn.CrossEntropyLoss()
        # 解析配置
        self.device = self.config['device']
        self.epochs = self.config["epoch"]
        self.val_frequency = self.config["val_frequency"]
        self.save_path = save_path

        self.train_num = train_num
        self.val_num = val_num
        self.best_acc = 0

        self.train_loader, self.val_loader = dataloder
        self.train_steps = len(self.train_loader)

        self.model = model.to(self.device)

        self.optimizer = optimizer

    def run_epoch(self, epoch):

        self.model.train()

        self.running_loss = 0
        train_bar = self.train_loader
        for step, data in enumerate(train_bar):
            images, labels = data
            self.optimizer.zero_grad()
            outputs = self.model(images.to(self.device))
            loss = self.loss(outputs, labels.to(self.device))
            loss.backward()
            self.optimizer.step()

            # print statistics
            self.running_loss += loss.item()

            if step % 100 == 0:
                train_bar = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     self.epochs,
                                                               loss)
                logging.info(train_bar)

    @torch.no_grad()
    def validate(self, epoch):

            acc = 0.0  # accumulate accurate number / epoch
            with torch.no_grad():
                val_bar = self.val_loader
                for val_data in val_bar:
                    val_images, val_labels = val_data
                    outputs = self.model(val_images.to(self.device))
                    predict_y = torch.max(outputs, dim=1)[1]
                    acc += torch.eq(predict_y, val_labels.to(self.device)).sum().item()

            val_accurate = acc / self.val_num

            logging.info('validation')
            logging.info('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
                  (epoch + 1, self.running_loss / self.train_steps, val_accurate))

            if not self.config["save_best_pth_only"]:
                torch.save(self.model.state_dict(), self.save_path + f'epoch_{epoch}.pth')

            if val_accurate > self.best_acc:
                self.best_acc = val_accurate
                torch.save(self.model.state_dict(), self.save_path + 'best.pth')

    def run(self):

        for epoch in range(self.epochs):

            self.run_epoch(epoch)

            if (epoch + 1) % self.val_frequency == 0:

                self.validate(epoch)


        return