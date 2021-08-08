import torch
import clip
from PIL import Image
import numpy as np
import cv2
import glob, random
import imutils, os, time

class clip_class:

    def get_ramdom_path():
        file_path = './static/image/Real World/'

        first_path = os.listdir(file_path)
        class_path = random.choice(first_path)
        rfile = os.path.join(file_path, class_path)
        img_list = os.listdir(rfile)
        sel_img = random.choice(img_list)
        rfile= rfile+'/'+ sel_img
        return class_path, rfile

    def clip_predict(rfile):
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model_use = "ViT-B/32"
        # model_use = "RN50x16" # ViT-B/32, RN50x16
        model, preprocess = clip.load(model_use, device=device)
        image = preprocess(Image.open(rfile)).unsqueeze(0).to(device)

        class_list = ['Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 'Bottle', 'Bucket', 'Calculator', 'Calendar', 'Candles', 'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 'Desk_Lamp', 'Drill', 'Eraser', 'Exit_Sign', 'Fan', 'File_Cabinet', 'Flipflops', 'Flowers', 'Folder', 'Fork', 'Glasses', 'Hammer', 'Helmet', 'Kettle', 'Keyboard', 'Knives', 'Lamp_Shade', 'Laptop', 'Marker', 'Monitor', 'Mop', 'Mouse', 'Mug', 'Notebook', 'Oven', 'Pan', 'Paper_Clip', 'Pen', 'Pencil', 'Postit_Notes', 'Printer', 'Push_Pin', 'Radio', 'Refrigerator', 'Ruler', 'Scissors', 'Screwdriver', 'Shelf', 'Sink', 'Sneakers', 'Soda', 'Speaker', 'Spoon', 'Table', 'Telephone', 'ToothBrush', 'Toys', 'Trash_Can', 'TV', 'Webcam']

        start_time = time.time()
        text = clip.tokenize(class_list).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)

            logits_per_image, logits_per_text = model(image, text)
            probs = np.array(logits_per_image.softmax(dim=-1).cpu().numpy())

        label = np.argmax(probs[0])
        end_time = time.time()
        print(probs[0][label])
        print("inference time :", end_time-start_time,"seconds")

        return class_list[label], rfile
