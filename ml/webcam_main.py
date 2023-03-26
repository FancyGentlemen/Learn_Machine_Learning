import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import json

# Load the pre-trained model
model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet50', pretrained=True)
model.eval()

# Define the classes of the model
#classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
#           'dog', 'frog', 'horse', 'ship', 'truck']
with open('iamgenet_classes.json') as json_file:
    classes = json.load(json_file)


# Define the transformation to apply to the input image
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Create a window to display the video stream and classification results
cv2.namedWindow("Webcam Classification")

# Start the video stream
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
    
    # Convert the frame to a PIL Image
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Apply the transformation pipeline to the input image
    img = transform(img).unsqueeze(0)
    
    # Pass the tensor through the model to get the predicted class probabilities
    with torch.no_grad():
        outputs = model(img)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
    
    # Get the class with the highest probability
    _, idx = torch.max(probs, 0)
    #label = str(idx.item())
    label = classes[str(idx)]
    
    # Overlay the classification results onto the frame
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Show the frame in the window
    cv2.imshow("Webcam Classification", frame)
    
    # Check if the user wants to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video stream and close the window
cap.release()
cv2.destroyAllWindows()
