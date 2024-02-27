
from deepforest import main

def predict(model, image_paths):
    """Predict bounding boxes for images
    Args:
        model (main.deepforest): A trained deepforest model.
        image_paths (list): A list of image paths.  
    Returns:
        list: A list of image predictions.
    """
    predictions = []
    for image_path in image_paths:
        prediction = model.predict_image(image_path)
        predictions.append(prediction)
    
    return predictions