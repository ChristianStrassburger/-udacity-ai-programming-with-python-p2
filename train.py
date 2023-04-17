import argparse
import torch
from flower_net import FlowerNet
from flower_data import FlowerData


class FlowerTrain:

    def __init__(self):
        pass

    def train(self, flower_data, flower_net, save_checkpoint = True):
        """
        Summary:
            Trains the model and saves a model checkpoint.

        Parameters:
            flower_data (FlowerData): A class for data handling.
            flower_net (FlowerNet): A class for model handling.
            save_checkpoint (bool): If 'True', a checkpoint is saved.

        Returns:
            model(obj) - The trained model.
        """ 
        dataloaders = flower_data.get_dataloader()
        train_dataloader = dataloaders[flower_data.train]
        validation_dataloader = dataloaders[flower_data.val]

        flower_net.initialize_model()
        model = flower_net.model

        if not model:
            return

        device = torch.device("cuda" if flower_net.gpu and torch.cuda.is_available() else "cpu")
        model.to(device)

        criterion = flower_net.criterion
        optimizer = flower_net.optimizer
        print_every = flower_net.print_every
        epochs = flower_net.epochs
        steps = 0
        running_loss = 0

        print()
        print(f"Start training - arch: {flower_net.arch}"
                f".. hidden_units: {flower_net.hidden_units}"
                f".. learning_rate: {flower_net.learning_rate}"
                f".. epochs: {epochs}"
                f".. device: {device}")

        for epoch in range(epochs):
            for images, labels in train_dataloader:
                steps += 1
                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()
                logps = model(images)
                batch_loss = criterion(logps, labels)
                batch_loss.backward()
                optimizer.step()

                running_loss += batch_loss.item()

                if steps % print_every == 0:
                    test_loss = 0
                    accuracy = 0
                    model.eval()

                    with torch.no_grad():
                        for images, labels in validation_dataloader:

                            images, labels = images.to(device), labels.to(device)
                            logps = model(images)
                            batch_loss = criterion(logps, labels)
                            test_loss += batch_loss.item()

                            ps = torch.exp(logps)
                            top_ps, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f"Epoch {epoch+1}/{epochs}.. "
                        f"Train loss:  {running_loss/print_every:.3f}.. "
                        f"Validation loss: {test_loss/len(validation_dataloader):.3f}.. "
                        f"Validation accuracy: {accuracy/len(validation_dataloader):.3f}")
                    running_loss = 0
                    model.train()

        print()

        if save_checkpoint:
            image_datasets = flower_data.get_datasets()
            class_to_idx = image_datasets[flower_data.train].class_to_idx
            flower_net.model.class_to_idx = class_to_idx
            flower_net.model.idx_to_class = flower_net.get_idx_to_class(class_to_idx)

            checkpoint_name = "checkpoint_" + flower_net.arch + ".pth"
            flower_net.save_checkpoint(checkpoint_name)

        return model


def get_input_args(flower_net):
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", nargs=1)
    parser.add_argument("-sd", "--save_dir", default=flower_net.save_dir)
    parser.add_argument("-ar", "--arch", default=flower_net.arch)
    parser.add_argument("-lr", "--learning_rate", default=flower_net.learning_rate)
    parser.add_argument("-hd", "--hidden_units", default=flower_net.hidden_units)
    parser.add_argument("-e", "--epochs", default=flower_net.epochs)
    parser.add_argument("-g", "--gpu", action="store_true")
    return parser.parse_args()


def set_flower_net(flower_net, input_args):
    flower_net.save_dir = input_args.save_dir
    flower_net.arch = input_args.arch
    flower_net.learning_rate = float(input_args.learning_rate)
    flower_net.hidden_units = int(input_args.hidden_units)
    flower_net.epochs = int(input_args.epochs)
    flower_net.gpu = input_args.gpu


def set_flower_data(flower_data, input_args):
    flower_data.data_dir =  input_args.data_dir[0]  


def main():
    flower_data = FlowerData()
    flower_net = FlowerNet()

    input_args = get_input_args(flower_net)
    set_flower_net(flower_net, input_args)
    set_flower_data(flower_data, input_args)

    flower_train = FlowerTrain()
    flower_train.train(flower_data, flower_net)


if __name__ == "__main__":
    main()
