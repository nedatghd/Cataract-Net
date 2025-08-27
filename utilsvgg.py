import json, os, math
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
from vit_pytorch import *
from sklearn.metrics import classification_report
from data import *
# Prepare Data 📊
# Import libraries
from attvgg import AttnVGG
import torch
import torchvision
import torchvision.transforms as transforms
from data import OCTDataset
#Utils 🛠️
import json, os, math
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
import torchvision
import torchvision.transforms as transforms
from vit_pytorch.recorder import Recorder
import numpy as np
import sklearn.metrics as metrics
from imblearn.metrics import sensitivity_score, specificity_score
import pdb
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, cohen_kappa_score
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from skimage.transform import resize
from torchvision.utils import make_grid


def visualize_images():
    trainset = data = OCTDataset(root_dir='ALL', transform=None, patient_csv="Cataractfull.xlsx")
    classes = ('normal_cataract', 'mild_cataract', 'severe_cataract')
    # Pick 30 samples randomly
    indices = torch.randperm(len(trainset))[:30]
    images = [np.asarray(trainset[i][0]) for i in indices]
    labels = [trainset[i][1] for i in indices]
    # Visualize the images using matplotlib
    # Assume images, labels, and classes are already defined
    fig = plt.figure(figsize=(15, 12))  # Adjust the figure size as needed

    for i in range(30):
        ax = fig.add_subplot(6, 5, i+1, xticks=[], yticks=[])
        ax.imshow(images[i])
        ax.set_title(classes[labels[i]], fontsize=12)  # Change the fontsize as needed

    plt.tight_layout()  # Adjusts subplot params for a nicer layout
    plt.show()


class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pth'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_accuracy = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, acc, model):
        score = -val_loss
        accuracy = acc
        if self.best_score is None:
            self.best_score = score
            self.best_accuracy = accuracy
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        elif score > self.best_score + self.delta and accuracy > self.best_accuracy + self.delta:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
    
def save_experiment(experiment_name, config, model, train_losses, test_losses, accuracies, avgmseies, base_dir="experiments"):
    outdir = os.path.join(base_dir, experiment_name)
    os.makedirs(outdir, exist_ok=True)

    # Save the config
    configfile = os.path.join(outdir, 'config.json')
    with open(configfile, 'w') as f:
        json.dump(config, f, sort_keys=True, indent=4)

    # Save the metrics
    jsonfile = os.path.join(outdir, 'metrics.json')
    with open(jsonfile, 'w') as f:
        data = {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'accuracies': accuracies,
            'averagemse' : avgmseies, 
        }
        json.dump(data, f, sort_keys=True, indent=4)


def load_experiment(experiment_name, checkpoint_name="checkpoint.pth", base_dir="experiments", device="cuda"):
    outdir = os.path.join(base_dir, experiment_name)
    # Load the config
    configfile = os.path.join(outdir, 'config.json')
    with open(configfile, 'r') as f:
        config = json.load(f)
    # Load the metrics
    jsonfile = os.path.join(outdir, 'metrics.json')
    with open(jsonfile, 'r') as f:
        data = json.load(f)
    train_losses = data['train_losses']
    test_losses = data['test_losses']
    accuracies = data['accuracies']
    averagemse = data['averagemse']

    # Load the model
    model = AttnVGG(num_classes=3, normalize_attn=True).to(device)
    cpfile = os.path.join(outdir, checkpoint_name)
    model.load_state_dict(torch.load(cpfile))
    return config, model, train_losses, test_losses, accuracies, averagemse

def generate_confusion_matrix(gt, pred, classes, ckp_directory):
    # Set up the matplotlib figure
    plt.figure(figsize=(8, 6))
    import seaborn as sns  # Optional for better visualization
    cm = confusion_matrix(gt, pred)
    # Use seaborn to create a heatmap for better visualization
    # Create a heatmap using seaborn
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Save the figure
    plt.savefig(f'{ckp_directory}/confusion_matrix.png')  # Save as PNG file
    plt.close()  # Close the figure to free up memory
    return cm

def compute_isic_metrics(gt, pred):
    gt_np = gt.cpu().detach().numpy()
    pred_np = pred.cpu().detach().numpy()

    gt_class = np.argmax(gt, axis=1)
    pred_class = np.argmax(pred, axis=1)

    ACC = accuracy_score(gt_class, pred_class)
    BACC = balanced_accuracy_score(gt_class, pred_class) # balanced accuracy
    Prec = precision_score(gt_class, pred_class, average='macro')
    Rec = recall_score(gt_class, pred_class, average='macro')
    F1 = f1_score(gt_class, pred_class, average='macro')
    AUC_ovo = metrics.roc_auc_score(gt_np, pred_np, average='macro', multi_class='ovo')
    AUC_macro = metrics.roc_auc_score(gt_class, pred_np, average='macro', multi_class='ovo')

    SPEC = specificity_score(gt_class, pred_class, average='macro')

    kappa = cohen_kappa_score(gt_class, pred_class, weights='quadratic')

    print(confusion_matrix(gt_class, pred_class))
    return ACC, BACC, Prec, Rec, F1, AUC_ovo, kappa
    #return ACC, BACC, Prec, Rec, F1, AUC_ovo, AUC_macro, SPEC, kappa

def compute_f1_score(gt, pred):
    gt_class = gt.cpu().detach().numpy()
    pred_np = pred.cpu().detach().numpy()

    #gt_class = np.argmax(gt_np, axis=1)
    pred_class = np.argmax(pred_np, axis=1)

    F1 = f1_score(gt_class, pred_class, average='macro')
    #AUC_ovo = metrics.roc_auc_score(gt_np, pred_np, average='macro', multi_class='ovo')
    #AUC_macro = metrics.roc_auc_score(gt_class, pred_np, average='macro', multi_class='ovo')

    #SPEC = specificity_score(gt_class, pred_class, average='macro')

    # print(confusion_matrix(gt_class, pred_class))
    return F1

def Extract_attention(a,up_factor):
    # compute the heatmap
    if up_factor > 1:
        a = F.interpolate(a, scale_factor=up_factor, mode='bilinear', align_corners=False)
    attn = make_grid(a, nrow=1, normalize=True, scale_each=True)
    attn = attn.permute((1,2,0)).mul(255).byte().cpu().numpy()
    attn = cv2.applyColorMap(attn, cv2.COLORMAP_JET)
    attn = cv2.cvtColor(attn, cv2.COLOR_BGR2RGB)
    attn = np.float32(attn) / 255
    # vis = 0.6 * img + 0.4 * attn
    return attn
    
    
@torch.no_grad()
def visualize_attention(ckp_directory, checkpoint_name, output_dir=None, device="cuda"):
    """
    Visualize the attention maps of the first 4 images.
    """
    os.makedirs(output_dir,exist_ok=True)
    # Load the model
    model = AttnVGG(num_classes=3, normalize_attn=True).to(device)
    cpfile = os.path.join(ckp_directory, checkpoint_name)
    model.load_state_dict(torch.load(cpfile))

    model.eval()

    testset = OCTDataset(root_dir="test_zone3", transform=None, patient_csv="Cataractfull.xlsx")
    # Convert the images to tensors
    test_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))])
    
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1,shuffle=False)
    
    classes = ('normal_cataract', 'mild_cataract', 'severe_cataract')
    
    for batch in test_loader:
        image, labels, visual_acuity, patient = batch
        # print(patient)
        # print(image.size())
        image_transformed = test_transform(image.squeeze(0).numpy()).to(device).unsqueeze(0)
        # Get the attention maps from the last block
        logit, Predicted_visual_acuity, attention_maps1, attention_maps2 = model(image_transformed) 
        # Get the predictions
        prediction = torch.argmax(logit, dim=1)
        # print("predicted Label",prediction)

        first_pool = Extract_attention(attention_maps1,8)
        # print(first_pool.shape)
        second_pool = Extract_attention(attention_maps2,16)
        # print(second_pool.shape)
        # print(image.shape)
        raw_image = image[0].cpu().numpy()
        # print(raw_image.shape)
        raw_image=cv2.resize(raw_image,(224,224))
        # raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)


        # Plot the images and the attention maps
        fig = plt.figure(figsize=(20, 20))
        mask = np.concatenate([np.zeros((224, 224,3)), np.ones((224, 224,3))], axis=1)
        
        ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
        raw_image = resize(raw_image, (224, 224), anti_aliasing=True)
        img = np.concatenate((raw_image, raw_image), axis=1)
        ax.imshow(img)
        # Mask out the attention map of the left image
        pool = (first_pool + second_pool)/2
        extended_attention_map = np.concatenate((np.zeros((224, 224,3)), pool), axis=1)
        ax.imshow(extended_attention_map, alpha=0.5, cmap='jet')
        extended_attention_map = np.ma.masked_where(mask==1, extended_attention_map)
        # ax.imshow(extended_attention_map, alpha=0.1, cmap='jet')

        # Show the ground truth and the prediction
        gt = classes[labels]
        pred = classes[prediction]
        ax.set_title(f"patient_num_{patient[0][1:]}:groundtruth_Label and Visual Acuity {gt, visual_acuity.detach().cpu().numpy()[0]} // predicted_Label and Visual Acuity {pred, Predicted_visual_acuity.detach().cpu().numpy()[0][0]}", color=("green" if gt==pred else "red"))
        if output_dir is not None:
            plt.tight_layout()  # Adjusts subplot params for a nicer layout
            plt.savefig(f"{output_dir}/attention_map_patient_{patient[0][1:]}.png")


        # plt.tight_layout()  # Adjusts subplot params for a nicer layout
        # plt.show()
