import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision.models.mobilenetv2 import mobilenet_v2
from ex02_0714_customdataset import My_ex02_customdata

label_dict = {0: "Chickenpox", 1: "Cowpox", 2: "Healthy", 3: "HFMD", 4: "Measles",
                            5: "Monkeypox"}

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## model setting
    model = mobilenet_v2()
    in_features = 1280
    model.classifier[1] = nn.Linear(in_features_, 6)

    ## model load
    model.load_state_dict(torch.load(f='./model_pt/ex02_0714_best_mobilenet_v2.pt'))
    # print(list(model.parameters()))

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))
    ])

    val_dataset = My_ex02_eustomdata("./dataset/val/", transforms=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    model.to(device)
    model.eval()

    correct = 0
    from tqdm import tqdm
    import cv2

    with torch.no_grad():
        for data, target, path in tqdm(val_loader) :
            target_ = target.item()

            data, target = data.to(device), target.to(device)
            output = model(data)
            
            pred = output.argmax(dim=1, keepdim=True)

            target_label = label_dict[target_]
            true_label_text = f"true : {target_label}"
            pred_label_text = f"pred : {pred_label}"

            img = cv2.imread(path[0])
            img = cv2.resize(img, (500,500))
            img = cv2.rectangle(img, (0,0), (500,100), (255,255,255), -1)
            img = cv2.putText(img, pred_label_text, (0,30), cv2.FONT_ITALIC, 1, (255,0,0), 2)
            img = cv2.putText(img, true_label_text, (0,75), cv2.FONT_ITALIC, 1, (255,0,0), 2)

            # cv2.imshow("test", img)
            # if cv2.waitKey() == ord('q') :
            #     exit()
            #     correct += pred.eq(target.view_as(pred)).sum().item()

    print("test ACC : {}/{} [{:.0f}]%\n".format(correct, len(val_loader.dataset),
                                                100*correct / len(val_loader.dataset)))
            
if __name__ == "__main__":
    main()