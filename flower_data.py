import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

class FlowerData:

    def __init__(self):
        self.data_dir =  os.getcwd()

        self.train = "train"
        self.val = "val"
        self.test = "test"

        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.resize_size = 255
        self.crop_size = 224
        self.scale_size = 256
        self.random_rotation = 30

    def get_split_data_dirs(self):
        """
        Summary:
            Returns a path dict to directories for training, validation and testing.
        
        Returns:
            split_data_dirs(dict) - A path dict.
        """   
        train_dir = os.path.join(self.data_dir, "train")
        valid_dir = os.path.join(self.data_dir, "valid")
        test_dir = os.path.join(self.data_dir, "test")

        split_data_dirs = {
            self.train: train_dir, 
            self.val: valid_dir, 
            self.test: test_dir}

        return split_data_dirs

    def get_transforms(self):
        """
        Summary:
            Returns transforms for training, validation and testing.

        Returns:
            data_transforms(dict) - A transform dict.
        """                      
        train_transform = transforms.Compose([
            transforms.RandomRotation(self.random_rotation),
            transforms.RandomResizedCrop(self.crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)])

        validation_transform = transforms.Compose([
            transforms.Resize(self.resize_size),
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)])

        test_transform = transforms.Compose([
            transforms.Resize(self.resize_size),
            transforms.CenterCrop(self.crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)])

        data_transforms = {
            self.train: train_transform, 
            self.val: validation_transform, 
            self.test: test_transform}      

        return data_transforms

    def get_datasets(self):
        """
        Summary:
            Returns a dict with datasets for training, validation and testing.

        Returns:
            image_datasets(dict) - A dataset dict.
        """  
        split_data_dirs = self.get_split_data_dirs()
        data_transforms = self.get_transforms()

        train_dataset = datasets.ImageFolder(
            root=split_data_dirs[self.train],
            transform=data_transforms[self.train])

        validation_dataset = datasets.ImageFolder(
            root=split_data_dirs[self.val],
            transform=data_transforms[self.val])

        test_dataset = datasets.ImageFolder(
            root=split_data_dirs[self.test],
            transform=data_transforms[self.test])

        image_datasets = {
            self.train: train_dataset, 
            self.val: validation_dataset, 
            self.test: test_dataset}

        return image_datasets       

    def get_dataloader(self, batch_size=64):
        """
        Summary:
            Returns a dataloader dict for training, validation and testing.

        Parameters:
            batch_size(int) - (Optional) A value for batch_size.

        Returns:
            dataloaders(dict) - A dataloader dict.
        """  
        image_datasets = self.get_datasets()
        train_dataloader = DataLoader(image_datasets[self.train], batch_size=batch_size, shuffle=True)
        validation_dataloader = DataLoader(image_datasets[self.val], batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(image_datasets[self.test], batch_size=batch_size, shuffle=True)
        dataloaders = {
            self.train: train_dataloader, 
            self.val: validation_dataloader, 
            self.test: test_dataloader}

        return dataloaders

    def get_class_to_idx(self):
        """
        Summary:
            Returns the train class to index dict.

        Returns:
            cti_dict(dict) - The train class to index dict.
        """  
        image_datasets = self.get_datasets()
        cti_dict = image_datasets[self.train].class_to_idx

        return cti_dict

    def process_image(self, image):
        """
        Summary:
            Processes an image.

        Parameters:
            image(image) - A pil image.

        Returns:
            np_image(np.array) - The processed image as np.array.
        """ 

        proc_image = image.copy()
        
        w_h_ratio = proc_image.width / proc_image.height
        sw, sh = (self.scale_size, int(self.scale_size/w_h_ratio)) if w_h_ratio < 1 else (int(self.scale_size*w_h_ratio), self.scale_size)
        proc_image = proc_image.resize(size=(sw, sh))

        l_r_border = (proc_image.width-self.crop_size)/2
        t_b_borer = (proc_image.height-self.crop_size)/2
        left = l_r_border
        top = t_b_borer
        right = proc_image.width-l_r_border
        bottom = proc_image.height-t_b_borer
        proc_image = proc_image.crop((left, top, right, bottom))

        np_image = np.array(proc_image)
        np_image = np_image/self.resize_size

        np_image = (np_image-self.mean)/self.std
        np_image = np_image.transpose((2, 0, 1))

        return np_image
