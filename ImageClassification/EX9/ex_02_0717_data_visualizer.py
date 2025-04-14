import os
import matplotlib.pyplot as plt

class DataVisulizer :
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.all_data = {}
        self.train_data = {}
        self.val_data = {}
        self.test_data ={}

    def load_data(self):
        train_dir = os.path.join(self.data_dir, 'train')
        val_dir = os.path.join(self.data_dir, 'validation')
        test_dir = os.path.join(self.data_dir, 'test')

        for label in os.listdir(train_dir) :
            label_dir = os.path.join(train_dir, label)
            # print(label_dir) # ./food_dataset\train\burger
            count = len(os.listdir(label_dir))
            self.all_data[label] = count
            self.train_data[label] = count

        for label in os.listdir(val_dir) :
            label_dir = os.path.join(val_dir, label)
            count = len(os.listdir(label_dir))
            self.val_data[label] = count
            if label in self.all_data :
                self.all_data[label] += count
            else :
                self.all_data[label] = count

        for label in os.listdir(test_dir) :
            label_dir = os.path.join(test_dir, label)
            count = len(os.listdir(label_dir))
            self.test_data[label] = count
            if label in self.all_data :
                self.all_data[label] += count
            else :
                self.all_data[label] = count

    def visualize_data(self):
        labels = list(self.all_data.keys())
        counts = list(self.all_data.values())

        plt.figure(figsize=(10,6))
        plt.bar(labels, counts)
        plt.title("label data number")
        plt.xlabel("labels")
        plt.ylabel('data number')
        plt.xticks(rotation=45, ha='right', fontsize=8)

        plt.show()

if __name__ == "__main__" :
    test = DataVisulizer("./food_dataset")
    test.load_data()
    test.visualize_data()