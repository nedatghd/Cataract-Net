import torch
from torch import nn, optim
from utils import *
from utils import save_experiment, EarlyStopping
from data import *
from data import prepare_data, OCTDataset
from HybridModel import HybridModel
from attvgg import AttnVGG
from attvgg_cbam_acmix import AttnVGG_CBAM_ACMix
import argparse
import torch
from torch import nn, optim
import yaml
import random
from inference import inference
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR
from focal_loss.focal_loss import FocalLoss
def set_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#for reproducibility 
set_seeds(seed=1912)

class Trainer:
    """
    The simple trainer.
    """

    def __init__(self, model, config, optimizer, loss_fn_clf, loss_fn_reg, exp_name, device, early_stopping, scheduler):
        self.model = model.to(device)
        self.config = config
        self.optimizer = optimizer
        self.loss_fn_clf = loss_fn_clf
        self.loss_fn_reg = loss_fn_reg
        self.exp_name = exp_name
        self.device = device
        self.early_stopping = early_stopping
        self.scheduler = scheduler

    def train(self, trainloader, testloader, epochs):
        """
        Train the model for the specified number of epochs.
        """
        # Keep track of the losses and accuracies
        train_losses, test_losses, accuracies, avgmseies = [], [], [], []
        # Train the model
        for i in range(epochs):
            train_loss = self.train_epoch(trainloader)
            accuracy, test_loss, avg_mse = self.evaluate(testloader)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            accuracies.append(accuracy)
            avgmseies.append(avg_mse)
            # Print current learning rate

            current_lr = self.scheduler.get_last_lr()
            print(f"Epoch==> {i+1}, Train loss: {train_loss:.4f}, Test loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}, MSE: {avg_mse:.4f},Learning Rate: {current_lr}")
            # Early stopping check
            self.early_stopping(test_loss, accuracy, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break
            self.scheduler.step(metrics = test_loss)
        # Save the experiment
        save_experiment(self.exp_name, self.config, self.model, train_losses, test_losses, accuracies, avgmseies)

    def train_epoch(self, trainloader):
        """
        Train the model for one epoch.
        """
        self.model.train()
        total_loss = 0
        for batch in trainloader:
            # Move the batch to the device
            
            # batch = [t.to(self.device) for t in batch[:2]]
            images, labels, visual_acuity, patient = batch
            # print("image size", images.size())
            # print(images.dtype)
            images = images.to(self.device).float()
            labels = labels.to(self.device)
            labels = torch.nn.functional.one_hot(labels, num_classes=3)
            visual_acuity = visual_acuity.to(self.device).float()

            # Zero the gradients
            self.optimizer.zero_grad()
            # Get predictions

            clf_output, reg_output,_,_ = self.model(images)

            reg_output = torch.squeeze(reg_output, dim=1) 
            # print(clf_output)
            # print(labels)
            # Calculate the loss
            clf_loss = self.loss_fn_clf(clf_output, labels) 
 
            # print(clf_loss.dtype)
            reg_loss = self.loss_fn_reg(reg_output, visual_acuity)


            # print(reg_loss.dtype)

            loss = clf_loss + reg_loss

            # Backpropagate the loss
            loss.backward()
            # Update the model's parameters
            self.optimizer.step()


            total_loss += loss.item() * len(images)
        return total_loss / len(trainloader.dataset)

    @torch.no_grad()
    def evaluate(self, testloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        avg_mse = 0
        with torch.no_grad():
            for batch in testloader:
                # Move the batch to the device
                images, labels, visual_acuity, patient = batch
                # print(images.size())
                # print(images)
                # print(labels)
                images = images.to(self.device).float()
                labels = labels.to(self.device)
                labels_onehot = torch.nn.functional.one_hot(labels, num_classes=3)
                visual_acuity = visual_acuity.to(self.device).float()

                # Get predictions
                clf_output, reg_output,_,_ = self.model(images)
                # clf_output, reg_output = self.model(images)

                reg_output = torch.squeeze(reg_output, dim=1) 

                # Calculate the loss
                loss = self.loss_fn_clf(clf_output, labels_onehot) + self.loss_fn_reg(reg_output, visual_acuity)
                total_loss += loss.item() * len(images)

                # Calculate the accuracy
                predictions = torch.argmax(clf_output, dim=1)
                # print("prediction",predictions)
                # print("true label", labels)
                correct += torch.sum(predictions == labels).item()
                mse = F.mse_loss(reg_output, visual_acuity)
                mse += mse.item()

        accuracy = correct / len(testloader.dataset)
        avg_mse = mse / len(testloader.dataset)
        avg_loss = total_loss / len(testloader.dataset)
        return accuracy, avg_loss, avg_mse.item()


def main(args):
    CONFIG_PATH = args.config_path
    LR = args.lr
    WEIGHT_DECAY = args.weight_decay
    BASE_DIR = args.base_dir
    EXP_NAME = args.exp_name
    TRAIN_DIR=args.train_dir
    TEST_DIR = args.test_dir
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    PATIENCE = args.patience 
    DEVICE = args.device


    if not os.path.exists(os.path.join(BASE_DIR, EXP_NAME)):
        os.makedirs(os.path.join(BASE_DIR, EXP_NAME), exist_ok=True)

    # parse config file
    with open(os.path.join(CONFIG_PATH), "r") as f:
            config = yaml.safe_load(f)

    # Load  dataset
    trainloader, testloader, _ = prepare_data(batch_size=BATCH_SIZE,train_dir=TRAIN_DIR, test_dir=TEST_DIR)
    # Create the model, optimizer, loss function and trainer

    model = AttnVGG(num_classes=3, normalize_attn=True).to(DEVICE)
    # model = AttnVGG_CBAM_ACMix(num_classes=3, normalize_attn=True).to(DEVICE)
    # model = GFNetMultiOutput().to(DEVICE)
    # model = ResMLPMultiOutput().to(DEVICE)
    # optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=WEIGHT_DECAY) 

    # Withoout class weights
    loss_fn_clf = FocalLoss(gamma=0.7)
    loss_fn_reg = nn.MSELoss()  


    # Define the early stopping mechanism
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True, path=f'{BASE_DIR}\{EXP_NAME}\checkpoint.pth')    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.75, verbose=True)  # Reduce learning rate on plateau
    trainer = Trainer(model, config, optimizer, loss_fn_clf,loss_fn_reg, EXP_NAME, device=DEVICE, early_stopping=early_stopping, scheduler = scheduler)
    trainer.train(trainloader, testloader, EPOCHS)

    # Plot training Results
    config, model, train_losses, test_losses, accuracies, mse = load_experiment(f"{EXP_NAME}")

    
    import matplotlib.pyplot as plt
    # Create two subplots of train/test losses and accuracies
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.plot(train_losses, label="Train loss")
    ax1.plot(test_losses, label="Test loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax2.plot(accuracies)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax3.plot(mse)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("MSE")
    plt.savefig(f"{BASE_DIR}\\{EXP_NAME}\metrics.png")
    # plt.show()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Training Script')
    parser.add_argument('--config_path', default="cateract.yml", help="Directory for configs")  
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning Rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight Decay')
    parser.add_argument('--base_dir', default="./data/raw_wsi_tcga_images/", help="Directory for slides")
    parser.add_argument('--exp_name', default="./data/raw_wsi_tcga_images/", help="Directory for slides")
    parser.add_argument('--train_dir', default="./data/raw_wsi_tcga_images/", help="Directory for slides")
    parser.add_argument('--test_dir', default="./data/raw_wsi_tcga_images/", help="Directory for slides")
    parser.add_argument('--epochs', type=int, default=100, help='Number of Epochs')
    parser.add_argument('--batch_size', type=int, default=6, help='Batch Size')
    parser.add_argument('--patience', type=int, default=10, help='Number of patience')
    parser.add_argument('--device', type=str, default="cuda", help='Base Model Name')

    
    args = parser.parse_args()
    
    main(args)