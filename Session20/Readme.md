# Session 20

**Hugging Face Demo**: 

**Prompt: a dog as a astronaut**

**Bird Style**

![](images/bird_style.png)

**Matrix Style**

![](images/matrix_style.png)

**Egorey Style**

![](images/egorey_style.png)


**Pjablonski Style**

![](images/pjablonski_style.png)


**Fairy tale painting Style**

![](images/fairy_style.png)


**Dreamy Painting Style**

![](images/dreamy_style.png)


### Blue Loss
```python
def blue_loss(images):
    # How far are the blue channel values to 0.9:
    error = torch.abs(images[:,2] - 0.7).mean() # [:,2] -> all images in batch, only the blue channel
    return error
```
**output:**

![image](images/blue_loss.png)

### Custom Loss
```python
def custom_loss(image):
    # Calculate colorfulness metric (standard deviation of RGB channels)
    std_dev = torch.std(image, dim=(1, 2))
    loss = torch.mean(std_dev)
    return loss
```

**output:**

![image](images/custom_loss-output.png)

