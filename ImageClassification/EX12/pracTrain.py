import argparse
import os
import torch
from torchvision.models.resnet import resnet50
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from pracCustomDataset import SportsDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm


class SportsClassifier:
    # train에 앞서 기존 저장된 Checkpoint를 불러오기 용이하도록 하는 클래스
    def __init__(self, model, optimizer, criterion):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # device는 어떤 학습 시행에도 상관없이 항상 같은 값이므로, 내부 변수로 고정
        self.model = model.to(self.device)
        self.optimizer = optimizer
        # 체크포인트 로딩을 할 때, 불러온 값들을 올려서 사용해야 하기 때문에
        # 내부 변수로 지정
        self.criterion = criterion.to(self.device)
        # 손실 함수는 .to(device)로 GPU로 보내야 하기 때문에,
        # device가 지정된 class 내부로 편입

        self.start_epochs = 0
        # 중단된 체크포인트의 epoch로부터 이어 학습할 수 있도록
        # 기록된 epoch가 있으면 불러와서 사용할 필요가 있음

        self.train_loss_list = []
        self.train_acc_list = []
        self.val_loss_list = []
        self.val_acc_list = []
        # 이상 통계를 위한 loss, acc 기록을 위한 list들

    def train(self, train_loader, val_loader, args):
        best_val_acc = 0.0
        # best.pt 저장을 판독하기 위해 이번 학습 주기 전체의 최고점을 저장

        for epoch in range(self.start_epochs, args.epochs):
            # 시작 epoch 횟수 (start_epochs) 는 체크포인트에서 로딩할 가능성 있음
            # 그러나 총 epoch 횟수는 커맨드라인에서 지정하므로
            # args.epochs로 지정을 하게 됨
            self.model.train()
            train_correct = 0.0
            train_loss = 0.0
            val_correct = 0.0
            val_loss = 0.0

            train_loader_iter = tqdm(train_loader,
                                     desc=(f"Epoch: {epoch + 1} / {args.epochs}"),
                                     leave=False)
            for index, (data, label) in enumerate(train_loader_iter):
                data = data.float().to(self.device)
                label = label.to(self.device)
                # DataLoader에서 불러와서 GPU에 넘김

                outputs = self.model(data)
                # 추측

                self.optimizer.zero_grad()
                loss = self.criterion(outputs, label)
                loss.backward()
                self.optimizer.step()
                # 역전파 과정

                train_loss += loss.item()

                _, pred = torch.max(outputs, 1)
                train_correct += (pred == label).sum().item()

            train_loss /= len(train_loader)    
            train_acc = train_correct / len(train_loader.dataset)

            self.train_loss_list.append(train_loss)
            self.train_acc_list.append(train_acc)

            print(f"train accuracy : {train_acc}")

            if epoch % 1 == 0:
                # validation 횟수 제한 (epoch n번당 1번씩 돌도록)
                self.model.eval()
                with torch.no_grad():
                    for data, label in val_loader:
                        data = data.float().to(self.device)
                        label = label.to(self.device)

                        output = self.model(data)

                        pred = output.argmax(dim=1, keepdim=True)

                        val_correct += pred.eq(label.view_as(pred)).sum().item()
                        val_loss += self.criterion(output, label).item()
                    
                val_loss /= len(val_loader)
                val_acc = val_correct / len(val_loader.dataset)

                self.val_loss_list(val_loss)
                self.val_acc_list(val_acc)

                if val_acc > best_val_acc:
                    torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                # 다음에 checkpoint를 이어서 학습하기 위해 필요한 값들
                "train_losses": self.train_loss_list,
                "train_accs": self.train_acc_list,
                "val_losses": self.val_loss_list,
                "val_accs": self.val_acc_list
            }, args.checkpoint_path.replace(".pt",
                                            "_best.pt"))
                    # 파일 저장 경로도 통상적으로 커맨드라인 인자로 받게 됨
                    # args.checkpoint_path == "~/~~.pt"
                    # -> ~~_best.pt
                    best_val_acc = val_acc
                    # 최고 정확도 갱신

            # epoch가 끝나면 checkpoint 저장
            torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                # 다음에 checkpoint를 이어서 학습하기 위해 필요한 값들
                "train_losses": self.train_loss_list,
                "train_accs": self.train_acc_list,
                "val_losses": self.val_loss_list,
                "val_accs": self.val_acc_list
            }, args.checkpoint_path.replace(".pt", f"_{epoch}.pt"))
            # 매번 epoch마다 epoch number까지 포함한 파일 이름으로 저장
            # 만약 매 epoch 모두를 저장하지 않을 경우, replace를 제거

        torch.save({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_losses": self.train_loss_list,
                "train_accs": self.train_acc_list,
                "val_losses": self.val_loss_list,
                "val_accs": self.val_acc_list
            }, args.checkpoint_path.replace(".pt", "_last.pt"))           



    def load_ckpt(self, ckpt_file):
        '''
        model.state_dict()
        optimizer.state_dict()
        '''
        ckpt = torch.load(ckpt_file)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.start_epochs = ckpt["epoch"]

        self.train_loss_list = ckpt["train_losses"]
        self.train_acc_list = ckpt["train_accs"]
        self.val_loss_list = ckpt["val_losses"]
        self.val_acc_list = ckpt["val_accs"]
        # 저장한 통계값을 알맞는 key 값으로 불러와서 self 변수에 저장


if __name__ == "__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    # 커맨드라인에서 총 epoch 수를 지정할 수 있도록 args에 추가
    parser.add_argument("--checkpoint_path", type=str,
                        default="./weight/sports_classifier_checkpoint.pt")
    parser.add_argument("--checkpoint_folder_path", type=str,
                        default="./weight/")
    # 체크포인트 .pt 파일 저장경로
    parser.add_argument("--resume_training", action="store_true")
    # store_true가 지정되면 커맨드라인에 이 인자가 있을 때 True가 들어감
    parser.add_argument("--learning_rate", type=float,
                        default=0.001)
    parser.add_argument("--weight_decay", type=float,
                        default=0.01)
    parser.add_argument("--batch_size", type=int,
                        default=64)
    parser.add_argument("--num_workers", type=int,
                        default=0)
    # lr 및 weight decay 등등의 hyper parameter들도 args로 받는게 좋음
    parser.add_argument("--resume_epoch", type=int,
                        default=0)
    # default 값은 임시로 0으로 지정 
    # (다른 숫자는 있을지 여부가 확실하지 않음)
    parser.add_argument("--train_dir", type=str,
                        default="./US_license_plates_dataset/train")
    # 폴더를 여는 데이터셋 기준 (pracCustomDataset.py)
    # csv를 여는 데이터셋일 경우 csv 경로를 지정 (pracCustomDataset_csv.py)
    parser.add_argument("--val_dir", type=str,
                        default="./US_license_plates_dataset/valid")
    # 통상적으로 데이터셋으로 사용할 폴더 역시 인자값으로 받도록 지정
    
    args = parser.parse_args()
    
    weight_folder_path = args.checkpoint_folder_path
    os.makedirs(weight_folder_path, exist_ok=True)

    model = resnet50(pretrained=True)
    # model.fc.in_features = 224
    # in_features 지정
    # 현재 데이터셋은 224x224로 지정되어있으므로 별도 지정 필요하지 않음
    model.fc.out_features = 100
    # 현재 데이터셋의 클래스 수는 100개 고정이므로 args로 뺄 필요 없음

    optimizer = Adam(model.parameters(), lr=args.learning_rate,
                     weight_decay=args.weight_decay)
    criterion = CrossEntropyLoss()

    classifier = SportsClassifier(model, optimizer, criterion)

    if args.resume_training:
        # classifier.load_ckpt(args.checkpoint_path)
        # 통상적으로는 checkpoint를 마지막 지점만 남기므로
        # 저장할 이름과 똑같이 사용해도 무방 (변경시 바꿔줘야 함)
        classifier.load_ckpt(
            args.checkpoint_path.replace(".pt",
                                         f"{args.resume_epoch}.pt"))
        # 파일 이름에 epoch 수를 지정해서 저장했을 경우,
        # 해당 숫자도 터미널에서 받을 필요가 있음

        # checkpoint_path를 폴더 이름을 제외하고 정의했을 경우,
        # os.path.join(args.checkpoint_folder_path, args.checkpoint_path) 처럼 붙여서 사용해야 함

    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        ToTensorV2()
    ])
    # train transform에 필요한 augmentation은 학습자가 적절히 판단하여 추가
    val_transform = A.Compose([
        ToTensorV2()
    ])

    train_dataset = SportsDataset(args.train_dir,
                                  train_transform)
    val_dataset = SportsDataset(args.val_dir,
                                val_transform)
    
    train_dataloader = DataLoader(train_dataset,
                                  batch_size = args.batch_size,
                                  num_workers = args.num_workers)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers)

    classifier.train(train_dataloader, val_dataloader, args)