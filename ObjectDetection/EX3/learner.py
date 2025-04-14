import torch
import numpy as np
import time
import os
from tqdm import tqdm
import cv2

class SegLearner:
    def __init__(self, model, optimizer, criterion, train_dataloader, valid_dataloader, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.criterion = criterion.to(self.device)
        self.optimizer = optimizer
        # 모델, 손실함수, 옵티마이저

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        # 데이터로더

        self.args = args
        # 터미널 인자로 받은 필요값들

        self.start_epoch = 0
        self.metrics = {
            "train_loss": [],
            "train_acc": [],
            "train_miou": [],
            "val_loss": [],
            "val_acc": [],
            "val_miou": []
        }
        # resume이 걸릴 경우 / 학습이 저장될 경우 필요한 값들

    def train(self):
        for epoch in range(self.start_epoch, self.args.epochs):
            self.model.train()
            train_loss = 0.0
            train_corr = 0.0
            val_loss = 0.0
            val_corr = 0.0

            train_iou = 0.0
            val_iou = 0.0
            # mIoU 계산을 위한 IoU 총합치 저장 변수

            for i, (inputs, labels) in enumerate(tqdm(self.train_dataloader)):
                inputs = inputs.float().to(self.device)
                labels = labels.long().to(self.device)

                self.optimizer.zero_grad() # 가중치 업데이트를 위한 optimizer 초기화

                outputs = self.model(inputs) # 순전파
                outputs = outputs["out"] # deeplab은 output이 dict 형태로 쓸 수 있도록 나오므로, 출력치 key로 받아옴

                loss = self.criterion(outputs, labels)
                loss.backward() # 역전파
                self.optimizer.step() # 가중치 업데이트

                preds = torch.argmax(outputs, dim=1) # output으로부터 class에 대한 예측값을 얻음
                
                train_loss += loss.item()
                corrects = torch.sum(preds == labels.data) # labels와 preds는 이미지 형태
                # 즉, 위의 line은 label 이미지와 preds 이미지를 겹쳤을 때, 일치하는 픽셀의 갯수가 나옴

                # 이미지 형태이므로, 여기에서 나오는 총 길이는 최대 520x520 크기일것 (pretrained 기준)
                batch_size = inputs.size(0)
                train_corr += corrects.double() / (batch_size * 520 * 520)
                # Pixel accuracy를 이용한 정확도 계산

                train_iou += self.calc_iou(preds, labels.data)
            
            _t_loss = train_loss / len(self.train_dataloader)
            # 이번 epoch의 평균 train loss
            _t_acc = train_corr / len(self.train_dataloader.dataset)
            # 이번 epoch의 평균 pixel accuracy
            _t_iou = train_iou / len(self.train_dataloader.dataset)
            # 이번 epoch의 miou

            self.metrics["train_loss"].append(_t_loss)
            self.metrics["train_acc"].append(_t_acc)
            self.metrics["train_miou"].append(_t_iou)

            print(f"[{epoch + 1} / {self.args.epochs}] train loss : {_t_loss}", 
                  f"train acc : {_t_acc}, train mIoU : {_t_iou}")

            # validation 시작
            self.model.eval()
            with torch.no_grad():
                for val_i, (inputs, labels) in enumerate(tqdm(self.valid_dataloader)):
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)

                    outputs = self.model(inputs)
                    outputs = outputs["out"]
                    loss = self.criterion(outputs, labels)
                    preds = torch.argmax(outputs, dim=1)
                
                    val_loss += loss.item()
                    corrects = torch.sum(preds == labels.data)

                    batch_size = inputs.size(0)
                    val_corr += corrects.double() / (batch_size * 520 * 520)
                    # Pixel accuracy를 이용한 정확도 계산
                    val_iou += self.calc_iou(preds, labels.data)

            _v_loss = val_loss / len(self.valid_dataloader)
            _v_acc = val_corr / len(self.valid_dataloader.dataset)
            _v_miou = val_iou / len(self.valid_dataloader.dataset)

            self.metrics["val_loss"].append(_v_loss)
            self.metrics["val_acc"].append(_v_acc)
            self.metrics["val_miou"].append(_v_miou)

            print(f"[{epoch + 1} / {self.args.epochs}] valid loss : {_v_loss}", 
                  f"valid acc : {_v_acc}, valid mIoU : {_v_miou}")
            
            self.save_ckpts(epoch)
            

    def load_ckpts(self):
        '''
        path: .pt 파일이 저장된 위치
        '''
        ckpt_path = os.path.join(self.args.weight_folder_path, self.args.weight_file_name)
        # 터미널 인자 args로부터 지정된 weight 로딩 경로를 받아옴

        ckpt = torch.load(ckpt_path) # .pt 파일을 불러와서 dictionary 형태로 선언
        self.model.load_state_dict(ckpt["model"]) # dict 안에 있는 "model"키로 저장할 모델 가중치 로드
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.start_epoch = ckpt["epoch"]
        self.metrics = ckpt["metrics"]

    def save_ckpts(self, epoch, file_name=None):
        # 체크포인트 저장을 처리하기 위한 편의함수
        if not os.path.exists(self.args.weight_folder_path):
            os.makedirs(self.args.weight_folder_path, exist_ok=True)
        # 모델 가중치가 저장될 폴더가 없을 경우, 오류가 날 수 있으므로
        # 터미널 인자 args에서 받은 model_folder_path가 있는지 확인 후, 없으면 생성

        if file_name is None:
            to_save_path = os.path.join(self.args.weight_folder_path, self.args.weight_file_name)
        else:
            to_save_path = os.path.join(self.args.weight_folder_path, file_name)
        # file name 커스텀을 위한 조건식 부분

        torch.save(
            {
                "model": self.model.state_dict(), # 현재 가중치 값
                "optimizer": self.optimizer.state_dict(), # optimizer의 현재 수치
                "epoch": epoch,
                "metrics": self.metrics
            }, to_save_path
        )


    @staticmethod
    def calc_iou(preds, labels):
        total_iou = 0.0
        # 들어온 batch의 IoU 총합
        for inp, ans in zip(preds, labels):
            # inp == preds에서 들어온, 예측치가 담겨있는 단일 이미지(텐서)
            # ans == labels에서 들어온, 정답치가 담겨있는 단일 이미지(텐서)
            inp = inp.cpu().numpy() 
            # inp는 device에 넘어가있는 텐서이므로, cpu로 넘겨준 뒤 텐서에서 numpy 행렬로 변환
            ans = ans.cpu().numpy()

            union_section = np.logical_or(inp, ans)
            inter_section = np.logical_and(inp, ans)
            # 위에서 or, and 연산을 통해 얻은 numpy 행렬은 Boolean 행렬이며, 계산에 사용할 수 있는 수가 아님

            # cv2.imshow("union", union_section.astype(np.uint8) * 255)
            # 마스크 확인용 imshow 코드

            uni_sum = np.sum(union_section)
            inter_sum = np.sum(inter_section)
            # 해당하는 행렬의 총 픽셀 수 (T/F Boolean mask 형태로 나올 것이므로 sum을 하면 픽셀 수를 얻음)
            # == 해당 영역의 넓이
            if uni_sum != 0:
                total_iou += inter_sum / uni_sum
            else:
                total_iou += 0
            # 성공적인 데이터 입력의 경우 uni_sum이 0이 될 일은 없음 (정답지 mask가 완전히 검은 이미지는 없으므로)
            # 그러나 만약의 경우를 대비하여 0으로 나눈 오류가 나지 않도록 처리

            # 교집합 넓이 / 합집합 넓이 = IoU

        return total_iou