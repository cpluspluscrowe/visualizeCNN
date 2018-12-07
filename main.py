from keras.models import load_model

class Global:
    sentiment_model_path = "/Users/ccrowe/Documents/Thesis/facebook_api/Notebooks/Image_CNN/getSentiment.h5"
    pathToImages = "/Users/ccrowe/Documents/Thesis/facebook_api/Notebooks/Image_CNN/images"
    def getImages():
        imagePathData = next(os.walk(Global.pathToImages))
        root = imagePathData[0]
        all_files = imagePathData[2]
        files_full_path = list(map(lambda x: os.path.join(root, x), all_files))
        return files_full_path
    images = getImages()

def getModel():
    model = load_model(Global.sentiment_model_path)

class CNN:
    model = getModel()


sample_image_path = Global.images[0]
print(sample_image_path)


from keras.preprocessing import image
import numpy as np

img = image.load_img(sample_image_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255.

import matplotlib.pyplot as plt

plt.imshow(img_tensor[0])
plt.show()


