from tensorflow import keras
import numpy as np
import urllib

model = keras.models.load_model('app/best_model_multi')

labels = [
  'Home Furnishing',
 'Baby Care',
 'Watches',
 'Home Decor & Festive Needs',
 'Kitchen & Dining',
 'Beauty and Personal Care',
 'Computers'
 ]

def PreprocessingImg(url_image='https://images.pexels.com/photos/190819/pexels-photo-190819.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1') :
  ### Fonction effectuant le preprocessing nécessaire sur une image ##
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(url_image, "url-image.jpg")
    image = keras.preprocessing.image.load_img("url-image.jpg", grayscale=False, color_mode="rgb", target_size=[224,224], interpolation="nearest")
    array = keras.preprocessing.image.img_to_array(image)
    return np.array(array)

def predict_pipeline(description=None, url_image=None):
    img_input = PreprocessingImg(url_image)
    img_input = np.array([img_input])
    text_input = np.array([description])
    preds = model.predict([text_input, img_input])
    return labels[np.argmax(preds)]