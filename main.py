from keras.models import load_model
import ntpath
import os 
from shutil import copyfile

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

def getImageActivations(image_path):
    file = ntpath.basename(image_path)
    current_dir = Global.dir_path
    new_folder = file[0:file.find(".")-1]
    sample_image_path = os.path.join(current_dir,new_folder,file)
    new_folder_path = os.path.join(current_dir,new_folder)

    print(new_folder_path)
    if not os.path.exists(new_folder_path):
        os.mkdir(new_folder_path)
        copyfile(image_path, os.path.join(new_folder_path,file))
    from keras.preprocessing import image
    import numpy as np

    img = image.load_img(image_path, target_size=(64,64))
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

            save_path = os.path.join(new_folder_path,str(activation_layer_index) + "-" + str(x) + "-" + ntpath.basename(sample_image_path))
            print(save_path)
            plt.savefig(save_path)

for sample_image_path in Global.images[1:10]:
    getImageActivations(sample_image_path)
