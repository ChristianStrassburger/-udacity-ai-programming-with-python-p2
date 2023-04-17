import io
import os
from torch import nn
import torch
from torchvision import models
from torch import optim

class FlowerNet:

    def __init__(self):
        self.__densenet121 = "densenet121"
        self.__vgg13 = "vgg13"
        self.__resnet152 = "resnet152"
        self.supported_arch = {
            self.__densenet121:self.__densenet121, 
            self.__vgg13:self.__vgg13, 
            self.__resnet152: self.__resnet152}

        self.__class_to_idx_cp = "class_to_idx"
        self.__idx_to_class_cp = "idx_to_class"
        self.__arch_cp = "arch"
        self.__num_features_cp = "num_features"
        self.__hidden_units_cp = "hidden_units"
        self.__output_size_cp = "output_size"
        self.__dropout_cp = "dropout_p"
        self.__state_dict_cp = "state_dict"
        self.__epochs_cp = "epochs"
        self.__optim_state_cp = "optim_state"
        self.__pretrained_cp = "pretrained"

        self.save_dir = os.getcwd()
        self.arch = self.__vgg13
        self.learning_rate = 0.01
        self.hidden_units = 512
        self.epochs = 1
        self.gpu = False

        self.output_size = 102
        self.dropout_p = 0.2

        self.pretrained = True
        self.model = None
        self.num_features = 0
        self.use_classifier_model = True

        self.criterion = nn.NLLLoss()
        self.optimizer = None
        self.print_every = 25

    def set_model_parameter_gradients(self, model, requires_grad):
        """
        Summary:
            Sets the parameter value requires_grad.
        Parameters:
            model: A torchvision model.
            requires_grad(bool) - The parameter value for requires_grad.
        """ 
        for param in model.parameters():
            param.requires_grad = requires_grad

    def initialize_model(self):
        """
        Summary:
            Returns an initialized model.

        Returns:
            model(obj) - The initialized model.
        """ 
        requires_grad = not self.pretrained

        if self.arch == self.supported_arch[self.__densenet121]:
            self.model = models.densenet121(pretrained=self.pretrained)
            self.set_model_parameter_gradients(self.model, requires_grad)
            self.num_features = self.model.classifier.in_features
            self.use_classifier_model = True
        elif self.arch == self.supported_arch[self.__vgg13]:
            self.model = models.vgg13(pretrained=self.pretrained)
            self.set_model_parameter_gradients(self.model, requires_grad)
            self.num_features = self.model.classifier[0].in_features
            self.use_classifier_model = True
        elif self.arch == self.supported_arch[self.__resnet152]:
            self.model = models.resnet152(pretrained=self.pretrained)
            self.set_model_parameter_gradients(self.model, requires_grad)
            self.num_features = self.model.fc.in_features
            self.use_classifier_model = False
        else:
            print(f"The model architecture: '{self.arch}' is not supported. Supported architectures are: {list(self.supported_arch.keys())}")

        if self.model:
            classifier = nn.Sequential(
                nn.Linear(self.num_features, self.hidden_units),
                nn.ReLU(),
                nn.Dropout(p=self.dropout_p),
                nn.Linear(self.hidden_units, self.output_size),
                nn.LogSoftmax(dim=1))
            if self.use_classifier_model:
                self.model.classifier = classifier
                self.optimizer = optim.Adam(self.model.classifier.parameters(), lr=self.learning_rate)
            else:
                self.model.fc = classifier
                self.optimizer = optim.Adam(self.model.fc.parameters(), lr=self.learning_rate)

        return self.model

    def get_idx_to_class(self, class_to_idx):
        """
        Summary:
            Returns an idx to class mapping.

        Parameters:
            class_to_idx (dict): A dict with a class to index mapping.

        Returns:
            idx_to_class(dict): An idx to class mapping.
        """ 
        idx_to_class = {}

        if class_to_idx:
            for key, value in class_to_idx.items():
                idx_to_class[value] = key
        return idx_to_class

    def save_checkpoint(self, checkpoint_name="checkpoint.pth"):
        """
        Summary:
            Saves a checkpoint.

        Parameters:.
            checkpoint_name (str): Path to checkpoint.pth file.
        """ 
        try:
            checkpoint = {self.__class_to_idx_cp: self.model.class_to_idx,
                        self.__idx_to_class_cp: self.model.idx_to_class,
                        self.__arch_cp: self.arch,
                        self.__num_features_cp: self.num_features,
                        self.__hidden_units_cp: self.hidden_units,
                        self.__output_size_cp: self.output_size,
                        self.__dropout_cp: self.dropout_p,
                        self.__state_dict_cp: self.model.state_dict(),
                        self.__epochs_cp: self.epochs,
                        self.__optim_state_cp: self.optimizer.state_dict(),
                        self.__pretrained_cp: self.pretrained}

            save_path = os.path.join(self.save_dir, checkpoint_name)
            print(f"Save checkpoint: {save_path}")
            torch.save(checkpoint, save_path)
        except Exception as ex:
            print(f"Error saving checkpoint: {ex}")
        print()

    def load_checkpoint(self, filepath):
        """
        Summary:
            Loads a checkpoint

        Parameters:
            filepath (str): A path to a checkpoint.pth file.

        Returns:
            checkpoint(obj) - A model checkpoint.
        """ 
        print(f"Load_checkpoint: {filepath}")
        checkpoint = None

        if not os.path.exists(filepath):
            return checkpoint

        try:
            with open(filepath, 'rb') as f:
                buffer = io.BytesIO(f.read())
                checkpoint = torch.load(buffer)

                self.arch = checkpoint[self.__arch_cp]
                self.num_features = checkpoint[self.__num_features_cp] 
                self.hidden_units = checkpoint[self.__hidden_units_cp]
                self.output_size = checkpoint[self.__output_size_cp]
                self.dropout_p = checkpoint[self.__dropout_cp]
                self.state_dict = checkpoint[self.__state_dict_cp]
                self.epochs = checkpoint[self.__epochs_cp]
                self.optim_state = checkpoint[self.__optim_state_cp]
                self.pretrained = checkpoint[self.__pretrained_cp]

                self.initialize_model()
                self.model.load_state_dict(self.state_dict)
                self.optimizer.load_state_dict(self.optim_state)
                self.model.class_to_idx = checkpoint[self.__class_to_idx_cp]
                self.model.idx_to_class = checkpoint[self.__idx_to_class_cp]
                self.model.eval()
        except Exception as ex:
            print(f"Error loading checkpoint: {ex}")                
        print()

        return checkpoint