from tensorflow.compat.v1.keras.backend import get_session
import matplotlib.pyplot as plt
import yaml
import pandas as pd
import dill
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from IPython.core.debugger import set_trace
import sys, os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'../'))

from src.visualization.visualize import visualize_explanation
from src.predict import predict_instance, predict_and_explain
from src.data.preprocess import remove_text

import tensorflow as tf
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
import keras.backend as K
import numpy as np
import json
import shap

def setup_shap():
    '''
    Load relevant information and create a LIME Explainer
    :return: dict containing important information and objects for explanation experiments
    '''

    # Load relevant constants from project config file
    cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))
    shap_dict = {}
    shap_dict['NUM_SAMPLES'] = cfg['LIME']['NUM_SAMPLES']
    shap_dict['NUM_FEATURES'] = cfg['LIME']['NUM_FEATURES']
    shap_dict['IMG_PATH'] = cfg['PATHS']['IMAGES']
    shap_dict['RAW_DATA_PATH'] = cfg['PATHS']['RAW_DATA']
    shap_dict['IMG_DIM'] = cfg['DATA']['IMG_DIM']
    shap_dict['PRED_THRESHOLD'] = cfg['PREDICTION']['THRESHOLD']
    shap_dict['CLASSES'] = cfg['DATA']['CLASSES']
    shap_dict['CLASS_MODE'] = cfg['TRAIN']['CLASS_MODE']
    shap_dict['COVID_ONLY'] = cfg['LIME']['COVID_ONLY']
    KERNEL_WIDTH = cfg['LIME']['KERNEL_WIDTH']
    FEATURE_SELECTION = cfg['LIME']['FEATURE_SELECTION']

    # Load train and test sets
    try:
      shap_dict['TRAIN_SET'] = pd.read_csv(cfg['PATHS']['TRAIN_SET'])
      shap_dict['TEST_SET'] = pd.read_csv(cfg['PATHS']['TEST_SET'])
    except:
      print("issue")

    # Create ImageDataGenerator for test set
    test_img_gen = ImageDataGenerator(preprocessing_function=remove_text,
                                       samplewise_std_normalization=True, samplewise_center=True)
    test_generator = test_img_gen.flow_from_dataframe(dataframe=shap_dict['TEST_SET'], directory=cfg['PATHS']['RAW_DATA'],
        x_col="filename", y_col='label_str', target_size=tuple(cfg['DATA']['IMG_DIM']), batch_size=1,
        class_mode='categorical', validate_filenames=False, shuffle=False)
    shap_dict['TEST_GENERATOR'] = test_generator

    # Load trained model's weights
    shap_dict['MODEL'] = load_model(cfg['PATHS']['MODEL_TO_LOAD'], compile=False)

    return shap_dict

def explain_xray_shap(shap_dict, idx, layerToExplain, save_exp=True):
    '''
    Make a prediction and provide a LIME explanation
    :param lime_dict: dict containing important information and objects for explanation experiments
    :param idx: index of image in test set to explain
    :param save_exp: Boolean indicating whether to save the explanation visualization
    '''

    # Get 50 preprocessed image in test set
    shap_dict['TEST_GENERATOR'].reset()
    X = []
    y = []
    for i in range(50):
        x, o = shap_dict['TEST_GENERATOR'].next()
        x = np.squeeze(x, axis=0)
        X.append(x)
        y.append(o)
    #x = np.squeeze(x, axis=0)
    X = np.array(X)
    y = np.array(y)

    # Get the corresponding original image (no preprocessing)
    orig_img = cv2.imread(shap_dict['RAW_DATA_PATH'] + shap_dict['TEST_SET']['filename'][idx])
    new_dim = tuple(shap_dict['IMG_DIM'])
    orig_img = cv2.resize(orig_img, new_dim, interpolation=cv2.INTER_NEAREST)     # Resize image

    to_explain = X[[idx]]

    model = shap_dict['MODEL']

    # explain how the input to the 7th layer of the model explains the top two classes
    def map2layer(x, layer):
        feed_dict = dict(zip([model.layers[0].input], [x.copy()]))
        return K.get_session().run(model.layers[layer].input, feed_dict)

    e = shap.GradientExplainer(
        (model.layers[layerToExplain].input, model.layers[-1].output),
        map2layer(X, layerToExplain),
        local_smoothing=0 # std dev of smoothing noise
    )

    shap_values,indexes = e.shap_values(map2layer(to_explain, layerToExplain), ranked_outputs=2)

    # get the names for the classes
    # Rearrange index vector to reflect original ordering of classes in project config
    #set_trace()
    indexes = [[indexes[0][shap_dict['CLASSES'].index(c)] for c in shap_dict['TEST_GENERATOR'].class_indices]]
    index_names = np.vectorize(lambda x: shap_dict['CLASSES'][x])(indexes)

    # plot the explanations
    shap.image_plot(shap_values, to_explain, index_names, orig_img=orig_img)

    return

if __name__ == '__main__':
    shap_dict = setup_shap()
    tf.compat.v1.disable_v2_behavior()                    
    explain_xray_shap(shap_dict, 0, 12, save_exp=True) 