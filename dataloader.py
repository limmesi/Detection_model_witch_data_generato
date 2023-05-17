import torch
from torchvision.transforms import transforms

from utils import generate_data, load_files


class MyDataloader(torch.utils.data.Dataset):
    def __init__(self):
        self.device = torch.device('cpu')
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        pills, imgs = load_files()
        self.pills = pills
        self.images = imgs

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, bbox = generate_data(self.pills, self.images, idx)
        image = self.transform(image)
        image = torch.tensor(image, dtype=torch.float32, device=self.device)
        bbox = torch.tensor(bbox, dtype=torch.float32, device=self.device)
        yield image, bbox
    
