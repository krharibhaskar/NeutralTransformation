import os
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import numpy as np
from numpy.random import choice
import torch

class dataset(Dataset):
    def __init__(self,  root, neutral_template_path, num_domains, height, width, hot_vector, keep_resolution=False, input_dim = 3, no_flip = True, crop_size = 300):
        self.root = root
        self.num_domains = num_domains
        self.input_dim = input_dim
        self.height = height
        self.width = width
        self.crop_size = crop_size
        self.no_flip = no_flip
        self.dataset, self.domain_num_list, self.domain_names = self.make_dataset()
        self.dataset_size = max([len(self.dataset[i]) for i in range(len(self.dataset))]) #size of the biggest domain
        self.neutral_template_path = neutral_template_path
        self.hot_vector = hot_vector
        self.keep_resolution = keep_resolution
        return

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        y1_= int(choice(self.domain_num_list, 1, replace=False)) #create an int in the range 0 to len(domain_names)
        y1_c_org = torch.FloatTensor(np.zeros((self.num_domains,))) #initialize one hot vector for image x1
        #y2_c_org = torch.FloatTensor(np.ones((self.num_domains,))) #initialize one hot vector for image x2
        y2_c_org = torch.FloatTensor(np.array(self.hot_vector)) #initialize one hot vector for image x2
        #y2_c_org = self.hot_vector
        #x1 = self.load_img(self.directory)
        x1 = self.load_img(self.dataset[y1_][index % len(self.dataset[y1_])])
        x2 = self.load_img(self.neutral_template_path)
        y1_c_org[y1_] = 1
        #return (x1, y1_c_org), (x2, y2_c_org)
        return {'x1':x1, 'x2':x2, 'y1':y1_c_org, 'y2':y2_c_org, 'name': self.dataset[y1_][index % len(self.dataset[y1_])]}

    def load_img(self, img_name):
        img = Image.open(img_name).convert('RGB')
        img = self.transform_image(img, self.height, self.width, self.crop_size, self.no_flip)
        if self.input_dim == 1:
          img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
          img = img.unsqueeze(0)
        return img

    def transform_image(self, image, height, width, crop_size, no_flip):
        transform_list = []
        if not self.keep_resolution:
            transform_list.append(transforms.Resize((height, width), Image.BICUBIC))
        if not no_flip:
            transform_list.append(transforms.RandomHorizontalFlip())
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        return torch.Tensor(transforms.Compose(transform_list)(image))

    def make_dataset(self):
        """
        Supports two input layouts:

        1) Domain-folder layout:
           root/
           ├── domain_1/
           │   ├── img_001.png
           │   └── img_002.png
           └── domain_2/
               └── img_003.png

        2) Flat-folder layout:
           root/
           ├── img_001.png
           ├── img_002.png
           └── img_003.png
        """
        image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

        root_items = sorted(os.listdir(self.root))
        root_files = [
            os.path.join(self.root, item)
            for item in root_items
            if os.path.isfile(os.path.join(self.root, item))
            and item.lower().endswith(image_extensions)
        ]

        root_dirs = [
            item
            for item in root_items
            if os.path.isdir(os.path.join(self.root, item))
        ]

        dataset = {}

        # Case 1: flat folder containing images directly
        if root_files:
            dataset[0] = root_files
            domain_names = [os.path.basename(os.path.normpath(self.root))]
            return dataset, sorted(dataset.keys()), domain_names

        # Case 2: original layout with one subfolder per domain
        for i, domain in enumerate(root_dirs):
            domain_dir = os.path.join(self.root, domain)
            fnames = [
                os.path.join(domain_dir, f)
                for f in sorted(os.listdir(domain_dir))
                if os.path.isfile(os.path.join(domain_dir, f))
                and f.lower().endswith(image_extensions)
            ]

            if fnames:
                dataset[i] = fnames

        if not dataset:
            raise ValueError(
                f"No image files found in '{self.root}'. "
                "Use either a flat folder of images or folders containing images."
            )

        return dataset, sorted(dataset.keys()), root_dirs
