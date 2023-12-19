from deepforest import main
from src.patches import crop_predictions
import open_clip
import torch
import glob
import tempfile
import os
from PIL import Image
import cv2
import numpy as np

class FieldGuide():
    def __init__(self, image_dir):
        self.detector = main.deepforest()
        self.detector.use_bird_release()

        #Classificaton model
        self.classifier, _ , self.preprocess_val = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
        self.tokenizer = open_clip.get_tokenizer('hf-hub:imageomics/bioclip')
        self.image_paths = glob.glob("{}/*".format(image_dir))[:2]
        self.tempdir = tempfile.TemporaryDirectory()
        self.root_dir = image_dir

    def query(self, prompt):
        """Search through image-level predictions for a prompt

        Args:
            prompt: dictionary of prompt: label
        Returns:
            responses:
        """
        responses = {}  
        for image_path in self.image_paths:
                crop = cv2.imread(image_path)
                response = self.classify(crop, prompt) 
                if response is None:
                    responses[image_path] = "Invalid crop"
                else:
                    responses[image_path] = list(prompt.keys())[np.argmax(response.numpy(), 1)[0]]

        return responses
    
    def find(self, prompt, buffer=20):
        """_summary_

        Args:
            image_paths (_type_): _description_

        Returns:
            
        """
        responses = {}  
        predicted_boxes = {}
        for image_path in self.image_paths:
            boxes = self.detector.predict_image(path=image_path)
            boxes = boxes[boxes["score"] > 0.2]

            # add a buffer to the boxes
            boxes["xmin"] = boxes["xmin"] - buffer
            boxes["ymin"] = boxes["ymin"] - buffer
            boxes["xmax"] = boxes["xmax"] + buffer
            boxes["ymax"] = boxes["ymax"] + buffer
            predicted_boxes[image_path] = boxes

            crops = self.extract_crops(boxes)
            responses[image_path] = {}
            for index, row in boxes.iterrows():
                response = self.classify(crops[index], prompt) 
                if response is None:
                    responses[image_path][index] = "Invalid crop"
                else:
                    responses[image_path][index] = list(prompt.keys())[np.argmax(response.numpy(), 1)[0]]
            
        return responses, predicted_boxes
    
    def draw_responses(self, boxes, responses, show=True):
        for image_path in responses:
            image_boxes = boxes[image_path]
            for index, row in image_boxes.iterrows():
                response = responses[image_path][index]
                #Draw bounding box
                image = cv2.imread(image_path)
                cv2.rectangle(image, (int(row["xmin"]), int(row["ymin"])), (int(row["xmax"]), int(row["ymax"])), (0, 255, 0), 2)
                #Draw response
                cv2.putText(image, response, (int(row["xmin"]), int(row["ymin"])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                if show:
                    cv2.imshow("image", image)
                    cv2.waitKey(0)
                else:
                    return image
                    
    def extract_crops(self, boxes):
        """Extract annotation crop from a list of predicted boxes

        Args:
            boxes (list): results of deepforest.main.predict* type methods
            buffer: distance in pixels to pad around the box
        """
        crops = crop_predictions(boxes, root_dir=self.root_dir, savedir=self.tempdir.name)

        return crops

    def classify(self, crop, prompt):
        # Taken from https://github.com/mlfoundations/open_clip/blob/main/docs/Interacting_with_open_clip.ipynb

        # Can this caption?
        #https://github.com/mlfoundations/open_clip/blob/main/docs/Interacting_with_open_coca.ipynb

        with torch.no_grad():
            # PIL image array
            try:
                # This is a temporary solution to deal with edge boxes cropping outside the image
                image = Image.fromarray(crop)
            except ValueError:
                return None
            crop_tensor = self.preprocess_val(image)
            batch = crop_tensor.unsqueeze(0)
            image_features = self.classifier.encode_image(batch)
            text = self.tokenizer(prompt)
            img_embedding, text_embedding, other = self.classifier(batch, text)
            probs = (100 * img_embedding @ text_embedding.T).softmax(dim=-1)
            print("Label probs:", probs)
        
        return probs

