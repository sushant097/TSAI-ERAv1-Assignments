import torch
import pandas as pd
import numpy as np
import gradio as gr
from PIL import Image
from torch.nn import functional as F
from collections import OrderedDict
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_lightning import LightningModule, Trainer, seed_everything
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as T
from resnet18 import LitResnet

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
new_line = '\n'
wrong_img = pd.read_csv('wrong_predictions.csv')
wrong_img_no = len(wrong_img)

model = LitResnet()
model.load_state_dict(torch.load("model.pth", map_location=torch.device('cpu')), strict=False)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

inv_normalize = T.Normalize(
    mean=[-0.50/0.23, -0.50/0.23, -0.50/0.23],
    std=[1/0.23, 1/0.23, 1/0.23])

grad_cams = [GradCAM(model=model, target_layers=[model.convblock3[i]], use_cuda=False) for i in range(5)]

def create_gradcam(input_tensor, label, target_layer):
    grad_cam = grad_cams[target_layer]
    targets = [ClassifierOutputTarget(label)]
    grayscale_cam = grad_cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :]
    return grayscale_cam


def inference(input_image, top_classes=3, show_cam=True, target_layers=[2, 3], transparency=0.5):
    orig_image = input_image
    input_image = transform(input_image)
     
    input_image = input_image.unsqueeze(0)
    output = model(input_image)
    
    softmax = torch.nn.Softmax(dim=0)
    o = softmax(output.flatten())

    confidences = {classes[i]: float(o[i]) for i in range(10)}
    confidences = dict(sorted(confidences.items(), key=lambda x:x[1],reverse=True))
    confidences = {i: confidences[i] for i in list(confidences)[:top_classes]}
    _, label = torch.max(output, 1)
    
    outputs = list()
    if show_cam:
        for layer in target_layers:
            grayscale_cam = create_gradcam(input_image, label, layer)
            output_image = show_cam_on_image(orig_image / 255, grayscale_cam, use_rgb=True, image_weight=transparency)
            outputs.append((output_image, f"Layer {layer - 5}"))    

    return outputs, confidences


examples = []
for i in range(10):
  examples.append([f'examples/{classes[i]}.jpg', 3, True,["-2","-1"],0.5])

demo_1 = gr.Interface(
    fn=inference,
    inputs=[
        gr.Image(shape=(32, 32), label="Uploaded Image").style(width=128, height=128),
        gr.Slider(1, 10, value=3, step=1, label="How many Top Predictions",
                  info="Display top classes?"),
        gr.Checkbox(label="Display GradCAM", value=True, info="Display GradCAM Images?"),
        gr.CheckboxGroup(["-5","-4", "-3", "-2", "-1"], value=["-2", "-1"], label="Conv Layers", type='index',
                         info="Layer you want to visualize GradCAM?",),
        gr.Slider(0, 1, value=0.5, label="Transparency", step=0.1,
                  info="Set Transparency of CAMs")
    ],
    outputs=[gr.Gallery(label="Output Images", columns=2, rows=2), gr.Label(label='Top Classes')],
    examples=examples
)


def show_misclassified(num_examples=10):
    result = list()
    for i in range(num_examples):
        j = np.random.randint(1,30)
        image = np.asarray(Image.open(f'Misclassified_images/{j}.jpg'))
        actual = classes[wrong_img.loc[j-1].at["actual"]]
        predicted = classes[wrong_img.loc[j-1].at["predicted"]]
        
        result.append((image, f"Actual:{actual}{new_line}Predicted:{predicted}"))
        
    return result


demo_2 = gr.Interface(
    fn=show_misclassified,
    inputs=[
        gr.Number(value=10, minimum=1, maximum=30, label="Uploaded number of images", precision=0,
                  info="Number of misclassified examples to show? (max 30)")
    ],
    outputs=[gr.Gallery(label="Misclassified Images (Actual / Predicted)", columns=5)]
)

demo = gr.TabbedInterface([demo_1, demo_2], ["CIFAR10 Classifier", "Mis-predicted Images"])
demo.launch(debug=True)
