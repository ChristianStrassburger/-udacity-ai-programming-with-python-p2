import argparse
import json
import os
import torch
from PIL import Image
from flower_net import FlowerNet
from flower_data import FlowerData

class FlowerPrediction:

    def __init__(self):
        self.top_k = 5
        self.category_names = None

    def get_category_names(self, classes, category_names_path):
        """
        Summary:
            Returns the category names.

        Parameters:
            classes (array): An array with mapped class ids. 
            category_names_path (str): Path to 'cat_to_name'-json file.

        Returns:
            cls_names (array) - A array with class names.
        """ 
        if not os.path.exists(category_names_path):
            return classes

        with open(category_names_path, 'r') as f:
            cat_to_name = json.load(f)
        
        cls_names = [""] * len(classes)

        for idx, cls in enumerate(classes):
            cat_name = cls + " - " + cat_to_name[cls] if cls in cat_to_name.keys() else str(cls)
            cls_names[idx] = cat_name

        return cls_names

    def predict(self, flower_data, flower_net, image_path, category_names_path=None, top_k = 5, gpu=False):
        """
        Summary:
            Predicts the 'top_k' classes.

        Parameters:
            flower_data (FlowerData): A class for data handling.
            flower_net (FlowerNet): A class for model handling.
            image_path (str): A path to an image.
            category_names_path (str): (Optional) A path to a 'cat_to_name'-json file.
            top_k (int): The number of top classes.
            gpu (bool): If 'True' the gpu is used.

        Returns:
            (top_ps, top_classes) (array, array) - Arrays with top percentages and class names.
        """ 
        print(f"Predict - image: {image_path}.. top_k: {top_k}.. gpu: {gpu}")
        model = flower_net.model

        if not model:
            return

        if not os.path.exists(image_path):
            print(f"Invalid image path: {image_path}")
            return

        with Image.open(image_path) as pil_image:
            image = flower_data.process_image(pil_image)

        device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")

        image = torch.tensor(image, dtype=torch.float32)
        batch = torch.unsqueeze(image, 0)
        image = image.to(device)

        model.eval()
        logps = model(batch)
        ps = torch.exp(logps)

        top_ps, top_classes = ps.topk(k=top_k, dim=1)
        top_ps = top_ps.detach()

        top_ps = top_ps.numpy()[0]
        top_classes = top_classes.numpy()[0]
        model.idx_to_class 
        cls_mapping =  [model.idx_to_class[cls_idx] for cls_idx in top_classes]

        if category_names_path:
            cls_mapping = self.get_category_names(cls_mapping, category_names_path)

        print()
        print(f"Top {top_k} classes:")
        for t_cls, t_ps in list(zip(cls_mapping, top_ps)):
            ps = round(t_ps * 100, 2)
            print(f"class {t_cls}: {ps} %")
        print()
        
        return (top_ps, cls_mapping)


def get_input_args(flower_pred):
    parser = argparse.ArgumentParser()
    parser.add_argument("image_cp_path", nargs=2)  

    parser.add_argument("-tk", "--top_k", default=flower_pred.top_k)
    parser.add_argument("-cn", "--category_names", default=flower_pred.category_names)
    parser.add_argument("-g", "--gpu", action="store_true")
    
    return parser.parse_args()


def main():
    flower_data = FlowerData()
    flower_net = FlowerNet()

    flower_pred = FlowerPrediction()
    input_args = get_input_args(flower_pred)
    image_path = input_args.image_cp_path[0]
    checkpoint_path = input_args.image_cp_path[1]
    category_names_path = input_args.category_names
    gpu = input_args.gpu
    top_k = int(input_args.top_k)
    
    
    if os.path.exists(checkpoint_path):
        
        checkpoint = flower_net.load_checkpoint(checkpoint_path)

        if not checkpoint:
            return

        top_ps, top_classes = flower_pred.predict(
            flower_data = flower_data,
            flower_net = flower_net,
            image_path = image_path, 
            category_names_path = category_names_path, 
            gpu = gpu, 
            top_k = top_k)
    else:
        print(f"Invalid checkpoint path: {checkpoint_path}")

if __name__ == "__main__":
    main()