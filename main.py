from keras.models import load_model
import ntpath
import os 

class Global:
    dir_path = "/Users/ccrowe/Documents/gitrepos/visualizeCNN"#os.path.dirname(os.path.realpath(__file__))
    sentiment_model_path = "/Users/ccrowe/Documents/Thesis/facebook_api/Notebooks/Image_CNN/getCommentCount_simple.h5"

    def getImages():
        pathToImages = "/Users/ccrowe/Documents/Thesis/facebook_api/Notebooks/Image_CNN/images"
        imagePathData = next(os.walk(pathToImages))
        root = imagePathData[0]
        all_files = imagePathData[2]
        files_full_path = list(map(lambda x: os.path.join(root, x), all_files))
        return files_full_path
    images = getImages()

def getModel():
    model = load_model(Global.sentiment_model_path)
    return model
class CNN:
    model = getModel()

sample_image_path = os.path.join(Global.dir_path,"magie_hall.jpeg") #Global.images[0]
print(sample_image_path)

from keras.preprocessing import image
import numpy as np

img = image.load_img(sample_image_path, target_size=(64,64))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

import matplotlib.pyplot as plt

from keras import models

layer_outputs = [layer.output for layer in CNN.model.layers]
activation_model = models.Model(inputs=CNN.model.input, outputs=layer_outputs)


activations = activation_model.predict(img_tensor)
for activation_layer_index in range(2):#range(len(activations)):
    activation_layer = activations[activation_layer_index]
    import matplotlib.pyplot as plt
    for x in range(activation_layer.shape[3]):
        plt.matshow(activation_layer[0, :, :,x], cmap='viridis')

        save_path = os.path.join(Global.dir_path,str(activation_layer_index) + "-" + str(x) + "-" + ntpath.basename(sample_image_path))
        print(save_path)
        plt.savefig(save_path)



import sys
from PIL import Image
images = map(Image.open, list(map(lambda x: os.path.join(Global.dir_path, x),filter(lambda x: "jpg" in x, os.listdir(Global.dir_path)))))
widths, heights = zip(*(i.size for i in images))

total_width = sum(widths)
max_height = max(heights)

new_im = Image.new('RGB', (total_width, max_height))

x_offset = 0
for im in images:
  new_im.paste(im, (x_offset,0))
  x_offset += im.size[0]

new_im.save('test.jpg')
