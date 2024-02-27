from bioclip_example import FieldGuide
import glob
from matplotlib import pyplot as plt

image_dir = "/Users/benweinstein/Downloads/newmexico"
guide = FieldGuide.FieldGuide(image_dir)

# Box level predictions
responses, boxes = guide.find({"This crop contains a bird.": "bird", "This crop contains a mammal.": "mammal","This crop contains no animals.": "no_animal"})
guide.draw_responses(boxes=boxes, responses=responses, show=True)

# Image level predictions
responses = guide.query({"This image contains a bird.": "bird", "This image contains a mammal.": "mammal","This image contains no animals.": "no_animal"})
for image_path, response in responses.items():
    print("{}: {}".format(image_path, response))
    plt.title(response)
    plt.imshow(plt.imread(image_path))
