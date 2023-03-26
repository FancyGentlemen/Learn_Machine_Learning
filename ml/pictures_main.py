import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import os

# Load the pre-trained model
model = torchvision.models.resnet50(pretrained=True)
model.eval()

# Define the transformation to apply to each image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Define the font and font size for the text overlay
font = ImageFont.truetype("arial.ttf", 24)

# Specify the directory containing the images
image_dir = input("Enter the path of the image directory: ")

# Loop over each image in the directory and process it
for filename in os.listdir(image_dir):
    # Load the image
    image_path = os.path.join(image_dir, filename)
    image = Image.open(image_path)
    
    # Apply the transformation to the image
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    
    # Feed the image tensor into the model and get the predicted class label
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = torch.max(output, 1)
        label = predicted.item()
    
    # Get the name of the predicted class from the pre-defined classes list
    classes = []
    with open("imagenet_classes.txt", "r") as f:
        for line in f:
            classes.append(line.strip())
    predicted_class = classes[label]
    
    # Draw the predicted class label onto the image
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), predicted_class, fill=(255, 255, 255), font=font)
    
    # Save the image with the predicted class label overlaid
    output_path = os.path.join(image_dir, "output_" + filename)
    image.save(output_path)
    print("Processed image:", filename)
