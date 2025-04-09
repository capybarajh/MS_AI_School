import torch
from torch.utils.data import Dataset, DataLoader

class HeightWeightDataset(Dataset):
    def __init__(self, csv_path):
        self.data = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            next(f)
            for line in f:
                _, height, weight = line.strip().split(",")
                height = float(height)
                weight = float(weight)
                convert_to_kg_data = round(self.convert_to_kg(weight), 2)
                convert_to_cm_data = round(self.inch_to_cm(height), 1)

                self.data.append([convert_to_cm_data, convert_to_kg_data])

    def __getitem__(self, index):
        data = torch.tensor(self.data[index], dtype=torch.float)
        return data

    def __len__(self):
        return len(self.data)

    def convert_to_kg(self, weight_lb):
        return weight_lb * 0.453592
    
    def inch_to_cm(self, inch):
        return inch * 2.54
    



if __name__ == "__main__":
    dataset = HeightWeightDataset("./hw_200.csv")
    dataloader = DataLoader(dataset, batch_size = 1, shuffle=True)

    for batch in dataloader:
        x = batch[:, 0].unsqueeze(1)
        y = batch[:, 1].unsqueeze(1)
        print(x, y)