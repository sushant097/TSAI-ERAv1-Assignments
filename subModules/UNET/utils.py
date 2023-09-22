import torch
import matplotlib.pyplot as plt
import numpy as np

def visualise_data(dataset):
    dataiter = iter(dataset)

    ind = 0
    fig = plt.figure(figsize=(20,10))
    for i in range(4):
        images, labels, one_hot_labels = next(dataiter)

        image = images 
        ind = ind + 1
        ax = fig.add_subplot(2, 4, ind)
        ax.set_title(f"\nOriginal",fontsize=12) 
        ax.imshow(np.transpose(image, (1, 2, 0)))
        plt.axis("off")
        
        ind = ind + 1
        ax = fig.add_subplot(2, 4, ind)
        ax.set_title(f"\nGround Truth",fontsize=12) 
        ax.imshow(labels)
        plt.axis("off")

        images, labels, one_hot_labels = next(dataiter)


def visualise_model_outputs(model, loader, num_imgs=2):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataiter = iter(loader)

    with torch.no_grad():
        ind = 0
        fig = plt.figure(figsize=(10,10))

        for i in range(num_imgs):
            images, labels, one_hot_labels = next(dataiter)
            labels = labels.to(torch.float)
            data, target = images.to(device), labels.to(device)

            output = model(data).squeeze()
            predicted_masks = torch.argmax(output, 1)
            predicted_masks = predicted_masks.cpu().numpy()
            
            ind = ind + 1
            ax = fig.add_subplot(num_imgs, 3, ind)
            ax.set_title("\n Label : Original",fontsize=12) 
            ax.imshow(np.transpose(images[0], (1, 2, 0))) 
            plt.axis("off")

            ind = ind + 1
            ax = fig.add_subplot(num_imgs, 3, ind)
            ax.set_title(f"\n Label : Ground Truth",fontsize=12)
            ax.imshow(labels[0])
            plt.axis("off")

            ind = ind + 1
            ax = fig.add_subplot(num_imgs, 3, ind)
            ax.set_title(f"\n Label : Predicted",fontsize=12)
            ax.imshow(predicted_masks[0])
            plt.axis("off")

            images, labels, one_hot_labels = next(dataiter)