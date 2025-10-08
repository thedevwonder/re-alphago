from torch.onnx.symbolic_opset9 import dim
from torch.utils.data import DataLoader
from sl_policy_network import SLPolicyNetwork
from dlgo.util.godataloader import GoDataLoader
import torch
import torch.optim as optim
import torch.nn as nn


class SLPolicyTrainer:
    def __init__(self, model):
        # initialize hyperparams
        self.model = model
        self.optimizer = None
        self.criterion = None

    def initialize(self):
        self.optimizer = optim.SGD(
            params = self.model.parameters(),
            lr = 0.003
        )
        self.criterion = nn.NLLLoss()
        return
    def train(self, loader):
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_sample = 0
        for (x, y) in loader:
            self.optimizer.zero_grad()
            log_probs, policy_logit = self.model(x)
            
            loss = self.criterion(log_probs, y)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_sample += y.size(0)
            prediction = log_probs.argmax(dim=1)
            correct_predictions += (prediction == y).sum().item()

        avg_loss = total_loss / len(loader)
        acc = correct_predictions / total_sample

        return avg_loss, acc
    def evaluate(self, loader):
        self.model.eval()
        total_loss = 0
        total_sample = 0
        correct_predictions = 0
        with torch.no_grad():
            for (x, y) in loader:
                log_probs, _ = self.model(x)
                loss = self.criterion(log_probs, y)
                prediction = log_probs.argmax(dim=1)
                total_loss += loss.item()
                total_sample += y.size(0)
                correct_predictions += (prediction == y).sum().item()


        avg_loss = total_loss / len(loader)
        acc = correct_predictions / total_sample
        return avg_loss, acc

def main():
    # load the model
    # prepare the dataset
    # load the dataset to a loader
    # load the trainer -> initialize the hyperparameters
    # for n epochs train, evaluate, save

    # load the model
    # TODO: customize features and filters and read more about how they operate behind the scenes
    # TODO: what are inbuilt filters in pytorch?
    print("Load model")
    model = SLPolicyNetwork()
    num_of_epochs = 5

    # prepare the dataset
    print("fetch and load data")
    training_feature_path = 'data/KGS-2019_04-19-1255-_train_features.npy'
    training_label_path = 'data/KGS-2019_04-19-1255-_train_labels.npy'
    test_feature_path = 'data/KGS-2019_04-19-1255-_test_features.npy'
    test_label_path = 'data/KGS-2019_04-19-1255-_test_labels.npy'
    training_dataset = GoDataLoader(training_feature_path, training_label_path).load_data()
    test_dataset = GoDataLoader(test_feature_path, test_label_path).load_data()

    # load the dataset to a loader
    # this is what we will pass to our Trainer
    training_loader = DataLoader(training_dataset, batch_size=16, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    # load the trainer
    trainer = SLPolicyTrainer(model)
    trainer.initialize()

    for epoch in range(num_of_epochs):
        training_loss, training_acc = trainer.train(training_loader)
        test_loss, test_acc = trainer.evaluate(test_loader)
        print(f"epoch: {epoch}, training loss: {training_loss}, training acc: {training_acc}, test loss: {test_loss}, test_acc: {test_acc}")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'train_acc': training_acc,
            'test_acc': test_acc,
        }, f'sl_policy_epoch_{epoch}.pth')
    return

if __name__ == '__main__':
    main()