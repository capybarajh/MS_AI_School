class CustomDataset:
    def __init__(self,path,mode='train'):
# if mode == 'train':
#       file_list = os.listdir(path)
        self.a = [1,2,3,4,5]
        pass
    def __getitem__(self,index):

        
        return self.a[index]
    def __len__(self):
        
        return len(self.a)

dataset_inst = CustomDataset('./images/', 'valid')

for item in dataset_inst:
    print(item)

print(len(dataset_inst))
