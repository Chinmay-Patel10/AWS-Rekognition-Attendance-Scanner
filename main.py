import boto3
import cv2
from keys import access_key, secret_key
import os

# Create AWS Rekognition Client
reko_client = boto3.client('rekognition',
                           aws_access_key_id=access_key,
                           aws_secret_access_key=secret_key,
                           region_name='us-east-1')

# Set the target class
target_class = 'person'

# Load Pictures
pic_dir = 'test_pics/'
pictures = []

for filename in os.listdir(pic_dir):
    if filename.endswith('.jpg'):
        image_path = os.path.join(pic_dir, filename)
        image = cv2.imread(image_path)
        resized_image = cv2.resize(image, (400, 600))  # Resize image to 800 x 600
        pictures.append(image)  # Append resized image to the list
    
# Detect Objects and Draw Bounding Boxes
for image in pictures:
    H, W, _ = image.shape  # Find height and width inside the loop
    success, encoded_image = cv2.imencode('.jpeg', image)
    image_bytes = encoded_image.tobytes()
    response = reko_client.detect_labels(Image={'Bytes': image_bytes},
                                          MinConfidence=50)

    for label in response['Labels']:
        if label['Name'] == target_class:
            for instance in label['Instances']:
                bbox = instance['BoundingBox']
                x1 = int(bbox['Left'] * W)
                y1 = int(bbox['Top'] * H)
                width = int(bbox['Width'] * W)
                height = int(bbox['Height'] * H)
                print(x1, y1, width, height)
                cv2.rectangle(image, (x1, y1), (x1 + width, y1 + height), 
                              (0, 255, 0), 3)

    # Display the image with bounding boxes
    cv2.imshow('image', image)
    cv2.waitKey(0)  # Wait for a key press to close the window 
