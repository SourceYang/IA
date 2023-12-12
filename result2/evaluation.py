import torch
import matplotlib.pyplot as plt
import numpy as np


def pred_lab(model, dev_dataloader, target_idx=1):
    all_predictions = []
    labels = []

    with torch.no_grad():
        for batch in dev_dataloader:
            batch_data = batch['input_data']
            if target_idx == 1:
                batch_labels = batch['out1']
            elif target_idx == 2:
                batch_labels = batch['out2']
            elif target_idx == 3:
                batch_labels = batch['out3']
            else:
                print("Invalid target_idx")
            labels.append(batch_labels)
            
            batch_predictions = model(batch_data)
            all_predictions.append(batch_predictions)
        
    all_predictions = torch.cat(all_predictions, dim=0)
    all_labels = torch.cat(labels, dim=0)
    
    return all_predictions, all_labels


def plot_55(all_predictions, all_labels, title):
    plt.figure(figsize=(15, 10))
    for i in range(25):
        idx = np.random.randint(0, len(all_predictions))
        plt.subplot(5, 5, i+1)
        plt.plot(all_labels[idx], label = 'Actual')
        plt.plot(all_predictions[idx], label = 'Predicted')
        plt.xticks([])

    plt.suptitle(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{title}.pdf')
    plt.show()


def plot_RMSE(all_predictions1, all_labels1, all_predictions2, all_labels2, all_predictions3, all_labels3):
    errors1 = torch.abs(all_predictions1 - all_labels1).numpy()  # Shape: [num_samples, seq_len]
    rmse1 = np.sqrt(np.mean(errors1**2, axis=0))
    std1 = np.std(errors1, axis=0)
    print("rmse for position-position correlation:", rmse1)
    
    errors2 = torch.abs(all_predictions2 - all_labels2).numpy()
    rmse2 = np.sqrt(np.mean(errors2**2, axis=0))
    std2 = np.std(errors2, axis=0)
    print("rmse for position-orientation correlation:", rmse2)
    
    errors3 = torch.abs(all_predictions3 - all_labels3).numpy()
    rmse3 = np.sqrt(np.mean(errors3**2, axis=0))
    std3 = np.std(errors3, axis=0)
    print("rmse for orientation-orientation correlation:", rmse3)
    
    fig, ax = plt.subplots(1, 3, figsize=(20,4))
    
    ax[0].plot(rmse1)
    ax[0].fill_between(np.arange(20), rmse1 - std1, rmse1 + std1, alpha=0.2)
    ax[0].set_title('RMSE, Position-Position Function')
    ax[0].set_xlabel('$r$ bins')
    ax[0].set_xticks(np.arange(20), [str(i) for i in range(1, 21)])
    ax[0].set_yscale('log')
    
    ax[1].plot(rmse2)
    ax[1].fill_between(np.arange(20), rmse2 - std2, rmse2 + std2, alpha=0.2)
    ax[1].set_title('RMSE, Position-Orientation Function')
    ax[1].set_xlabel('$r$ bins')
    ax[1].set_xticks(np.arange(20), [str(i) for i in range(1, 21)])

    ax[2].plot(rmse3)
    ax[2].fill_between(np.arange(20), rmse3 - std3, rmse3 + std3, alpha=0.2)
    ax[2].set_title('RMSE, Orientation-Orientation Function')
    ax[2].set_xlabel('$r$ bins')
    ax[2].set_xticks(np.arange(20), [str(i) for i in range(1, 21)])
    
    plt.tight_layout()
    plt.show()

def calculate_aapd(predictions, target):
    predictions = predictions.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    
    absolute_percentage_difference = abs(predictions - target) / (abs(target) + 1e-8)
    aapd = np.mean(absolute_percentage_difference)
    return aapd.item()