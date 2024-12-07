import os

from torch.utils.data import Dataset
from torchvision.io import read_image

from FishRecord import FishRecord


class FishDataset(Dataset):
    def __init__(self, root_path, prefix, transform, device="cpu"):
        self.root_path = root_path
        self.prefix = prefix
        self.transform = transform
        self.device = device
        self.data_list = self.label_processing()
        self.X, self.T = self.data_preprocessing()


    def __len__(self):
        return len(self.X)


    def __getitem__(self, idx):
        return self.X[idx], self.T[idx]


    def label_processing(self):
        label_file_path = os.path.join(self.root_path, "class_id.csv")

        with open(label_file_path, 'r') as file:
            next(file)
            data_list = [FishRecord(self.root_path, self.prefix, line.strip().split()[0]) for line in file]

        return data_list


    def data_preprocessing(self):
        X_list = []
        T_list = []

        for record in self.data_list:
            image_X = read_image(record.file_path)
            copies = self.species_to_copies_map(record.species_idx)
            for _ in range(copies):
                tensor_X = self.transform(image_X)
                X_list.append(tensor_X.to(self.device))
                T_list.append(record.species.to(self.device))

        return X_list, T_list


    def species_to_copies_map(self, species_idx: int):
        if species_idx == 1:
            return 1
        elif species_idx == 3 or species_idx == 4:
            return 3
        elif species_idx == 2 or species_idx == 5:
            return 4
        else:
            return 10