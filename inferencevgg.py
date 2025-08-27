import torch
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from utils import *
from utils import compute_isic_metrics
from data import OCTDataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from attvgg import AttnVGG
from torchvision.utils import make_grid
import cv2
import os
from sklearn.metrics import (balanced_accuracy_score, precision_score, 
                             recall_score, f1_score, classification_report)
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, cohen_kappa_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

@torch.no_grad()
def inference(config, ckp_directory, checkpoint_name, output_dir=None, device="cuda", train_dir="train_zone3", test_dir="test_zone3"):

    # Open a text file to save the output
    output_file_path = os.path.join(output_dir, 'inference_output.txt')
    with open(output_file_path, 'w') as f:
        # Redirect print statements to the file
        def print_to_file(*args, **kwargs):
            print(*args, **kwargs)  # Print to console
            print(*args, file=f, **kwargs)  # Print to file

        label2idx = {
            'R1': 0,
            'R2': 1,
            'R3': 2
        }

        idx2label = {y: x for x, y in label2idx.items()}

        trainloader, testloader, classes = prepare_data(batch_size=1, train_dir=train_dir, test_dir=test_dir)

        # Load the model
        model = AttnVGG(num_classes=3, normalize_attn=True).to(device)
        cpfile = os.path.join(ckp_directory, checkpoint_name)
        model.load_state_dict(torch.load(cpfile))

        model.eval()
        correct = 0
        avg_mse = 0
        avg_mae = 0
        all_class_pred = []
        all_label_gt = []
        all_reg_output = []
        all_true_visual_acuity = []
        patients = []

        with torch.no_grad():
            for batch in testloader:
                # Move the batch to the device
                images, labels, visual_acuity, patient = batch
                images = images.to(device).float()
                labels = labels.to(device)
                visual_acuity = visual_acuity.to(device).float()

                # Get predictions
                clf_output, reg_output, a1, a2 = model(images)
                reg_output = torch.squeeze(reg_output, dim=1) 
                
                # Calculate the accuracy
                predictions = torch.argmax(clf_output, dim=1)
                correct += torch.sum(predictions == labels).item()
                mse = F.mse_loss(reg_output, visual_acuity)
                mae = F.l1_loss(reg_output, visual_acuity)
                avg_mse += mse.item()
                avg_mae += mae.item()

                all_reg_output.append(reg_output.cpu().item())
                all_true_visual_acuity.append(visual_acuity.cpu().item())
                all_class_pred.append(predictions.cpu().item())
                all_label_gt.append(labels.cpu().item())
                patients.append(patient)

        # Create a DataFrame from the lists
        data = {
            'Predicted Class': all_class_pred,
            'Ground Truth Label': all_label_gt,
            "predicted_visual_acuity": all_reg_output,
            "True visual acuity": all_true_visual_acuity,
            'Patient ID': patients
        }

        df = pd.DataFrame(data)
        df.to_excel(f'{ckp_directory}/predictions_output.xlsx', index=False)

        accuracy = correct / len(testloader.dataset)
        avg_mse = avg_mse / len(testloader.dataset)
        avg_mae = avg_mae / len(testloader.dataset)

        print_to_file(f'Accuracy of the network on the {len(testloader.dataset)} test images: {100 * accuracy:.2f} %')
        print_to_file(f'Average MSE of the network on the {len(testloader.dataset)} test images for visual acuity prediction: {avg_mse:.4f}')
        print_to_file(f'Average MAE of the network on the {len(testloader.dataset)} test images for visual acuity prediction: {avg_mae:.4f}')

        BACC = balanced_accuracy_score(all_label_gt, all_class_pred)  # balanced accuracy
        Prec = precision_score(all_label_gt, all_class_pred, average='macro')
        Rec = recall_score(all_label_gt, all_class_pred, average='macro')
        F1 = f1_score(all_label_gt, all_class_pred, average='macro')
        SPEC = specificity_score(all_label_gt, all_class_pred, average='macro')
        kappa = cohen_kappa_score(all_label_gt, all_class_pred, weights='quadratic')
        cm = generate_confusion_matrix(all_label_gt, all_class_pred, classes, ckp_directory)

        # Calculate accuracy for each class
        class_accuracy = cm.diagonal() / cm.sum(axis=1)

        # Print class-wise accuracy
        print_to_file("Class-wise Accuracy:")
        for i in range(len(classes)):
            print_to_file(f"Class {classes[i]}: {class_accuracy[i]:.2f}")

        print_to_file(classification_report(all_label_gt, all_class_pred, target_names=classes))

        print_to_file("Balanced Accuracy Score:", BACC)
        print_to_file("Precision Score:", Prec)
        print_to_file("Recall Score:", Rec)
        print_to_file("F1 Score:", F1)
        print_to_file("Specificity Score:", SPEC)
        print_to_file("Cohen Kappa Score:", kappa)
        # Calculate evaluation metrics
        mae = mean_absolute_error(all_true_visual_acuity, all_reg_output)
        mse = mean_squared_error(all_true_visual_acuity, all_reg_output)
        rmse = np.sqrt(mse)
        r2 = r2_score(all_true_visual_acuity, all_reg_output)

        # Discretize the predicted and ground truth values into bins for Cohen's Kappa
        # bins = np.linspace(0, 2, num=5)  # Create 4 bins between 0 and 2
        # df['Predicted_Binned'] = pd.cut(all_reg_output, bins=bins, labels=False)
        # df['Ground_Truth_Binned'] = pd.cut(all_true_visual_acuity, bins=bins, labels=False)

        # Calculate Cohen's Kappa
        # kappa = cohen_kappa_score(df['Predicted_Binned'], df['Ground_Truth_Binned'])

        # Print the results
        print("________________________________")
        print_to_file("Regression Part")
        print("________________________________")

        print_to_file(f"Mean Absolute Error (MAE): {mae:.4f}")
        print_to_file(f"Mean Squared Error (MSE): {mse:.4f}")
        print_to_file(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print_to_file(f"R-squared (R²): {r2:.4f}")
        # print_to_file(f"Cohen's Kappa: {kappa:.4f}")

        # Binarize the ground truth labels
        classes = np.unique(all_label_gt)
        y_true_binarized = label_binarize(all_label_gt, classes=classes)

        # Get predicted probabilities for each class
        y_score = np.zeros((len(all_class_pred), len(classes)))
        for i in range(len(all_class_pred)):
            y_score[i, all_class_pred[i]] = 1  # Assigning a probability of 1 to the predicted class

        # Calculate ROC curve and AUC for each class
        fpr = {}
        tpr = {}
        roc_auc = {}
        for i in range(len(classes)):
            fpr[i], tpr[i], _ = roc_curve(y_true_binarized[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Micro-average ROC curve
        y_true_micro = y_true_binarized.ravel()
        y_score_micro = y_score.ravel()
        fpr_micro, tpr_micro, _ = roc_curve(y_true_micro, y_score_micro)
        roc_auc_micro = auc(fpr_micro, tpr_micro)

        # Macro-average ROC curve
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes))]))
        mean_tpr = np.zeros_like(all_fpr)

        for i in range(len(classes)):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        mean_tpr /= len(classes)
        fpr_macro = all_fpr
        tpr_macro = mean_tpr
        roc_auc_macro = auc(fpr_macro, tpr_macro)
        print_to_file("AUC Score:", roc_auc_macro)


        # Save ROC curves and AUC results
        plt.figure(figsize=(10, 7))
        colors = ['blue', 'orange', 'green', 'red', 'purple']  # Vibrant colors for different classes
        for i in range(len(classes)):
            plt.plot(fpr[i], tpr[i], color=colors[i % len(colors)], label=f'ROC curve (area = {roc_auc[i]:.2f}) for class {classes[i]}')

        # Plot micro-average ROC curve
        plt.plot(fpr_micro, tpr_micro, color='cyan', label=f'Micro-average ROC curve (area = {roc_auc_micro:.2f})', linestyle='--')

        # Plot macro-average ROC curve
        plt.plot(fpr_macro, tpr_macro, color='magenta', label=f'Macro-average ROC curve (area = {roc_auc_macro:.2f})', linestyle=':')

        # Adding the title and labels
        plt.title('Receiver Operating Characteristic (ROC) Curves for Multiclass Classification', fontsize=16)
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.legend(loc='lower right')
        plt.grid(True)

        # Save the plot to the output directory
        plt.savefig(os.path.join(output_dir, 'roc_curves.png'))
        plt.close()  # Close the figure





