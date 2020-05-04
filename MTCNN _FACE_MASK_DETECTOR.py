# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 02:53:59 2020

@author: insanepopeye
"""
from mtcnn.mtcnn import MTCNN 
import torch 
import torchvision.transforms as transforms
from PIL import Image
import cv2

def result(model_path): #function for getting the model prediction
    Checkpoint = torch.load(model_path,map_location='cpu') #change it to gpu if you have gpu
    model = Checkpoint['model']
    model.load_state_dict(Checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False
    
    return model.eval()

loaded_model = result("models/9.pth")

train_transforms = transforms.Compose([
                                       transforms.Resize((224,224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                                       ]) #transforming images into input tensor type
        
#triggering web cam for frames
cap = cv2.VideoCapture(0)
font_scale=1
thickness = 2
red = (0,0,255)
green = (0,255,0)
blue = (255,0,0)
font=cv2.FONT_HERSHEY_SIMPLEX
fps = int(cap.get(cv2.CAP_PROP_FPS))
# check if webcam has launched 
if (cap.isOpened()== False): 
  print("Can Not Find Webcam Input !!" )

detector = MTCNN()
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  _, frame = cap.read()
  if _ == True:
      result_list = detector.detect_faces(frame) #Detecting face from frames
      #Could detect multiple faces from each frames
      #print(faces)
      cv2.putText(frame, "FPS COUNT : "+ str(fps), (10,30), font, font_scale, blue, thickness)
      #cv2.putText(frame, "BUG FIXED", (10,80), font, font_scale, green, thickness)
      for result in result_list:
          x, y, w, h = result['box']
          croped_img = frame[y:y+h, x:x+w] #Cropping only facial portion from frame 
          face_portion = Image.fromarray(croped_img, mode = "RGB") #Converting the facial prtion into PIL image format 
          face_portion = train_transforms(face_portion) 
          face_portion = face_portion.unsqueeze(0)#transforming image to tensor structure
          prediction = loaded_model(face_portion)
          _, predicted = torch.max(prediction.data, 1)
          prediction = predicted.item()
              #print("prediction 1 is" + str(prediction))
          if prediction == 1:
              cv2.putText(frame, "No Mask", (x,y - 10), font, font_scale, red, thickness)
              cv2.rectangle(frame, (x, y), (x+w, y+h), red, 2)
              #print("prediction is No Mask" )
          elif prediction == 0:
              cv2.putText(frame, "Masked", (x,y - 10), font, font_scale, green, thickness)
              cv2.rectangle(frame, (x, y), (x+w, y+h), green, 2)
              #print("prediction is Masked")
  cv2.imshow('Latest Logic',frame)     
  if cv2.waitKey(25) & 0xFF == ord('q'):
          break
     
cap.release() #release webcam

cv2.destroyAllWindows() # destroy frame
