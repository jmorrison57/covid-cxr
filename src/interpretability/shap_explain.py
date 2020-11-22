import shap
import pandas as pd
import yaml
import os
import datetime
import dill
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import sys, os
sys.path.append(os.path.join(os.path.dirname(sys.path[0]),'../'))

from src.visualization.visualize import visualize_explanation
from src.predict import predict_instance, predict_and_explain
from src.data.preprocess import remove_text


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
    shap_dict['TRAIN_SET'] = pd.read_csv(cfg['PATHS']['TRAIN_SET'])
    shap_dict['TEST_SET'] = pd.read_csv(cfg['PATHS']['TEST_SET'])

    # Create ImageDataGenerator for test set
    test_img_gen = ImageDataGenerator(preprocessing_function=remove_text,
                                       samplewise_std_normalization=True, samplewise_center=True)
    test_generator = test_img_gen.flow_from_dataframe(dataframe=shap_dict['TEST_SET'], directory=cfg['PATHS']['RAW_DATA'],
        x_col="filename", y_col='label_str', target_size=tuple(cfg['DATA']['IMG_DIM']), batch_size=1,
        class_mode='categorical', validate_filenames=False, shuffle=False)
    shap_dict['TEST_GENERATOR'] = test_generator

    # Define the LIME explainer
    shap_dict['EXPLAINER'] = LimeImageExplainer(kernel_width=KERNEL_WIDTH, feature_selection=FEATURE_SELECTION,
                                                verbose=True)
    dill.dump(shap_dict['EXPLAINER'], open(cfg['PATHS']['LIME_EXPLAINER'], 'wb'))    # Serialize the explainer

    # Load trained model's weights
    shap_dict['MODEL'] = load_model(cfg['PATHS']['MODEL_TO_LOAD'], compile=False)

    return shap_dict


def explain_xray(shap_dict, idx, save_exp=True):
    '''
    Make a prediction and provide a LIME explanation
    :param shap_dict: dict containing important information and objects for explanation experiments
    :param idx: index of image in test set to explain
    :param save_exp: Boolean indicating whether to save the explanation visualization
    '''

    # Get i'th preprocessed image in test set
    shap_dict['TEST_GENERATOR'].reset()
    for i in range(idx + 1):
        x, y = shap_dict['TEST_GENERATOR'].next()
    x = np.squeeze(x, axis=0)

    # Get the corresponding original image (no preprocessing)
    orig_img = cv2.imread(shap_dict['RAW_DATA_PATH'] + shap_dict['TEST_SET']['filename'][idx])
    new_dim = tuple(shap_dict['IMG_DIM'])
    orig_img = cv2.resize(orig_img, new_dim, interpolation=cv2.INTER_NEAREST)     # Resize image

    # Make a prediction for this image and retrieve a LIME explanation for the prediction
    start_time = datetime.datetime.now()
    explanation, probs = predict_and_explain(x, shap_dict['MODEL'], shap_dict['EXPLAINER'],
                                      shap_dict['NUM_FEATURES'], shap_dict['NUM_SAMPLES'])
    print("Explanation time = " + str((datetime.datetime.now() - start_time).total_seconds()) + " seconds")


    # Get image filename and label
    img_filename = shap_dict['TEST_SET']['filename'][idx]
    label = shap_dict['TEST_SET']['label'][idx]

    # Rearrange prediction probability vector to reflect original ordering of classes in project config
    probs = [probs[0][shap_dict['CLASSES'].index(c)] for c in shap_dict['TEST_GENERATOR'].class_indices]

    # Visualize the LIME explanation and optionally save it to disk
    if save_exp:
        file_path = shap_dict['IMG_PATH']
    else:
        file_path = None
    if shap_dict['COVID_ONLY'] == True:
        label_to_see = shap_dict['TEST_GENERATOR'].class_indices['COVID-19']
    else:
        label_to_see = 'top'
    _ = visualize_explanation(orig_img, explanation, img_filename, label, probs, shap_dict['CLASSES'], label_to_see=label_to_see,
                          dir_path=file_path)
    return


if __name__ == '__main__':
    shap_dict = setup_shap()
    i = 0                                                       # Select i'th image in test set
    explain_xray(shap_dict, i, save_exp=True)                   # Generate explanation for image
