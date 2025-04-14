import argparse
import os
import glob
import torch.nn as nn
import torch
import torchvision
import albumentations as A
import pandas as pd
import matplotlib.pyplot as plt
from albumentations.pytorch import ToTensorV2
from torchvision.models.efficientnet import efficientnet_b0
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from ex_01_0718_customdataset import MyUSLicensePlatesDataset

class Classifier_US_LicensePlate :
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.train_losses = []
        self.valid_losses = []
        self.train_accs = []
        self.valid_accs = []

    def train(self, train_loader, val_loader, epochs, optimizer, criterion, start_epoch=0) :
        best_val_acc = 0.0
        print("Training....")

        for epoch in range(start_epoch, epochs) :
            train_loss = 0.0
            val_loss = 0.0
            train_acc = 0.0
            val_acc = 0.0

            self.model.train()
            train_loader_iter = tqdm(train_loader, desc=(f"Epoch : {epoch + 1}/{epochs}"), leave=False)

            for index, (data, target) in enumerate(train_loader_iter) :
                data, target = data.float().to(self.device), target.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                _,pred = torch.max(outputs, 1)
                train_acc += (pred == target).sum().item()

                train_loader_iter.set_postfix({"Loss" : loss.item()})

            train_loss /= len(train_loader)
            train_acc = train_acc / len(train_loader.dataset)

            # eval()
            self.model.eval()
            with torch.no_grad() :
                for data, target in val_loader :
                    data, target = data.float().to(self.device), target.to(self.device)
                    output = self.model(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    val_acc += pred.eq(target.view_as(pred)).sum().item()
                    val_loss += criterion(output, target).item()

            val_loss /= len(val_loader)
            val_acc = val_acc / len(val_loader.dataset)

            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.valid_losses.append(val_loss)
            self.valid_accs.append(val_acc)

            print(f"Epoch [{epoch + 1}/{epochs}], Train loss: {train_loss:.4f}, "
                  f"Val loss: {val_loss:.4f}, Train ACC: {train_acc:.4f}, Val ACC: {val_acc:.4f}")

            if val_acc > best_val_acc :
                torch.save(self.model.state_dict(), "./ex01_0718_efficientnet_b0_best.pt")
                best_val_acc = val_acc

            # save the model state and optimizer state after each epoch
            torch.save({
                'epoch':epoch + 1,
                'model_state_dict' : self.model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),
                'train_losses' : self.train_losses,
                'train_accs' : self.train_accs,
                'val_losses' : self.valid_losses,
                'val_accs' : self.valid_accs,
            },"./weight/0718/ex01_0718_efficientnet_b0_checkpoint.pt")

        torch.save(self.model.state_dict(), "./ex01_0718_efficientnet_b0_last.pt")

        self.save_result_to_csv()
        self.plot_loss()
        self.plot_accuracy()

    def save_result_to_csv(self):
        df = pd.DataFrame({
            'Train Loss' : self.train_losses,
            'Train ACC' : self.train_accs,
            'Validation Loss' : self.valid_losses,
            'Validation ACC' : self.valid_accs
        })
        df.to_csv('./train_val_result_ex01.csv', index=False)

    def plot_loss(self):
        plt.figure()
        plt.plot(self.train_losses, label="Train loss")
        plt.plot(self.valid_losses, label="val loss")
        plt.xlabel("Epoch")
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig("./ex01_loss_plot.jpg")

    def plot_accuracy(self):
        plt.figure()
        plt.plot(self.train_accs, label="Train Accuracy")
        plt.plot(self.valid_accs, label="Valid Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig("./ex01_accuracy_plot.jpg")

    def run(self, args) :
        self.model = efficientnet_b0(pretrained=True)
        self.model.classifier[0] = nn.Dropout(p=0.5, inplace=True)
        self.model.classifier[1] = nn.Linear(1280, out_features=50)
        self.model.to(self.device)

        train_transforms = A.Compose([
            A.Resize(width=224,height=224),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
            A.RandomShadow(),
            A.RandomRain(),
            A.RandomFog(),
            A.RGBShift(r_shift_limit=5, g_shift_limit=5, b_shift_limit=5, p=0.3),
            A.VerticalFlip(),
            A.HorizontalFlip(),
            A.RandomBrightness(),
            A.RandomRotate90(),
            A.RandomGamma(),
            ToTensorV2()
        ])

        val_transforms = A.Compose([
            A.Resize(width=224, height=224),
            ToTensorV2()
        ])

        # dataset and dataloader
        train_dataset = MyUSLicensePlatesDataset(args.train_dir, transform=train_transforms)
        val_dataset = MyUSLicensePlatesDataset(args.val_dir, transform=val_transforms)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

        epochs = args.epochs
        criterion = CrossEntropyLoss().to(self.device)
        optimizer = AdamW(self.model.parameters(), lr=args.learning_rate,
                          weight_decay=args.weight_decay)
        start_epoch = 0

        if args.resume_training :
            checkpoint = torch.load(args.checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.train_losses = checkpoint['train_losses']
            self.train_accs = checkpoint['train_accs']
            self.valid_losses = checkpoint['val_losses']
            self.valid_accs = checkpoint['val_accs']
            start_epoch = checkpoint['epoch']

        self. train(train_loader, val_loader, epochs, optimizer, criterion, start_epoch=start_epoch)

if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, default="./US_license_plates_dataset/train/",
                        help='directory path to the training dataset')
    parser.add_argument("--val_dir", type=str, default="./US_license_plates_dataset/valid/",
                        help="directory path to the valid dataset ")
    parser.add_argument("--epochs", type=int, default=50,
                        help="number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=124,
                        help='batch size for training and validation')
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="learning rate for optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-2,
                        help='weight decay for optimizer')
    parser.add_argument("--resume_training", action='store_true',
                        help='resume training from the last checkpoint')
    parser.add_argument("--checkpoint_path", type=str,
                        default="./weight/0718/ex01_0718_efficientnet_b0_checkpoint.pt",
                        help="path to the checkpoint file")
    parser.add_argument("--checkpoint_folder_path", type=str,
                        default="./weight/0718")
    args = parser.parse_args()

    weight_folder_path = args.checkpoint_folder_path
    os.makedirs(weight_folder_path, exist_ok=True)

    classifier = Classifier_US_LicensePlate()
    classifier.run(args)

