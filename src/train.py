import pandas as pd
import yaml
import os
import datetime
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from math import ceil
import tensorflow.summary as tf_summary
from tensorflow.keras.metrics import BinaryAccuracy, CategoricalAccuracy, Precision, Recall, AUC
from tensorflow.keras.models import save_model
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src.models.models import *
from src.visualization.visualize import *
from src.custom.metrics import F1Score
from src.data.preprocess import remove_text

def get_class_weights(histogram, class_multiplier=None):
    '''
    Computes weights for each class to be applied in the loss function during training.
    :param histogram: A list depicting the number of each item in different class
    :param class_multiplier: List of values to multiply the calculated class weights by. For further control of class weighting.
    :return: A dictionary containing weights for each class
    '''
    weights = [None] * len(histogram)
    for i in range(len(histogram)):
        weights[i] = (1.0 / len(histogram)) * sum(histogram) / histogram[i]
    class_weight = {i: weights[i] for i in range(len(histogram))}
    if class_multiplier is not None:
        class_weight = [class_weight[i] * class_multiplier[i] for i in range(len(histogram))]
    print("Class weights: ", class_weight)
    return class_weight


def random_minority_oversample(train_set):
    '''
    Oversample the minority class using the specified algorithm
    :param train_set: Training set image file names and labels
    :return: A new training set containing oversampled examples
    '''
    X_train = train_set[[x for x in train_set.columns if x != 'label']].to_numpy()
    if X_train.shape[1] == 1:
        X_train = np.expand_dims(X_train, axis=-1)
    Y_train = train_set['label'].to_numpy()
    sampler = RandomOverSampler(random_state=np.random.randint(0, high=1000))
    X_resampled, Y_resampled = sampler.fit_resample(X_train, Y_train)
    print("Train set shape before oversampling: ", X_train.shape, " Train set shape after resampling: ", X_resampled.shape)
    train_set_resampled = pd.DataFrame({'filename': np.squeeze(X_resampled, axis=1), 'label':Y_resampled})
    return train_set_resampled


def train_model(cfg, data, callbacks, verbose=1):
    '''
    Train a and evaluate model on given data.
    :param cfg: Project config (from config.yml)
    :param data: dict of partitioned dataset
    :param callbacks: list of callbacks for Keras model
    :param verbose: Verbosity mode to pass to model.fit_generator()
    :return: Trained model and associated performance metrics on the test set
    '''

    # If set in config file, oversample the minority class
    if cfg['TRAIN']['IMB_STRATEGY'] == 'random_oversample':
        data['TRAIN'] = random_minority_oversample(data['TRAIN'])

    # Create ImageDataGenerators
    train_img_gen = ImageDataGenerator(rotation_range=10, preprocessing_function=remove_text,
                                       samplewise_std_normalization=True, samplewise_center=True)
    val_img_gen = ImageDataGenerator(preprocessing_function=remove_text,
                                       samplewise_std_normalization=True, samplewise_center=True)
    test_img_gen = ImageDataGenerator(preprocessing_function=remove_text,
                                       samplewise_std_normalization=True, samplewise_center=True)

    # Create DataFrameIterators
    img_shape = tuple(cfg['DATA']['IMG_DIM'])
    y_col = 'label_str'
    class_mode = 'categorical'
    train_generator = train_img_gen.flow_from_dataframe(dataframe=data['TRAIN'], directory=cfg['PATHS']['TRAIN_IMGS'],
        x_col="filename", y_col=y_col, target_size=img_shape, batch_size=cfg['TRAIN']['BATCH_SIZE'], class_mode=class_mode)
    val_generator = val_img_gen.flow_from_dataframe(dataframe=data['VAL'], directory=cfg['PATHS']['VAL_IMGS'],
        x_col="filename", y_col=y_col, target_size=img_shape, batch_size=cfg['TRAIN']['BATCH_SIZE'], class_mode=class_mode)
    test_generator = test_img_gen.flow_from_dataframe(dataframe=data['TEST'], directory=cfg['PATHS']['TEST_IMGS'],
        x_col="filename", y_col=y_col, target_size=img_shape, batch_size=cfg['TRAIN']['BATCH_SIZE'], class_mode=class_mode,
        shuffle=False)

    # Apply class imbalance strategy. We have many more X-rays negative for COVID-19 than positive.
    if cfg['TRAIN']['IMB_STRATEGY'] == 'class_weight':
        class_multiplier = cfg['TRAIN']['CLASS_MULTIPLIER']
        histogram = np.bincount(np.array(train_generator.labels).astype(int))   # Get class distribution
        class_multiplier = [class_multiplier[cfg['DATA']['CLASSES'].index(c)] for c in test_generator.class_indices]
        class_weight = get_class_weights(histogram, class_multiplier)

    # Define metrics.
    covid_class_idx = test_generator.class_indices['COVID-19']   # Get index of COVID-19 class
    thresholds = 1.0 / len(cfg['DATA']['CLASSES'])      # Binary classification threshold for a class
    metrics = ['accuracy', CategoricalAccuracy(name='accuracy'),
               Precision(name='precision', thresholds=thresholds, class_id=covid_class_idx),
               Recall(name='recall', thresholds=thresholds, class_id=covid_class_idx),
               AUC(name='auc'),
               F1Score(name='f1score', thresholds=thresholds, class_id=covid_class_idx)]

    # Define the model.
    print('Training distribution: ', ['Class ' + list(test_generator.class_indices.keys())[i] + ': ' + str(histogram[i]) + '. '
           for i in range(len(histogram))])
    input_shape = cfg['DATA']['IMG_DIM'] + [3]
    if cfg['TRAIN']['CLASS_MODE'] == 'binary':
        histogram = np.bincount(data['TRAIN']['label'].astype(int))
        output_bias = np.log([histogram[i] / (np.sum(histogram) - histogram[i]) for i in range(histogram.shape[0])])
        model = dcnn_resnet(cfg['NN']['DCNN_BINARY'], input_shape, metrics, 2, output_bias=output_bias)
    else:
        n_classes = len(cfg['DATA']['CLASSES'])
        histogram = np.bincount(data['TRAIN']['label'].astype(int))
        output_bias = np.log([histogram[i] / (np.sum(histogram) - histogram[i]) for i in range(histogram.shape[0])])
        model = dcnn_resnet(cfg['NN']['DCNN_MULTICLASS'], input_shape, metrics, n_classes, output_bias=output_bias)

    # Train the model.
    steps_per_epoch = ceil(train_generator.n / train_generator.batch_size)
    val_steps = ceil(val_generator.n / val_generator.batch_size)
    history = model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=cfg['TRAIN']['EPOCHS'],
                                  validation_data=val_generator, validation_steps=val_steps, callbacks=callbacks,
                                  verbose=verbose, class_weight=class_weight)

    # Run the model on the test set and print the resulting performance metrics.
    test_results = model.evaluate_generator(test_generator, verbose=1)
    test_metrics = {}
    test_summary_str = [['**Metric**', '**Value**']]
    for metric, value in zip(model.metrics_names, test_results):
        test_metrics[metric] = value
        print(metric, ' = ', value)
        test_summary_str.append([metric, str(value)])
    return model, test_metrics, test_generator


def train_experiment(experiment='single_train', save_weights=True, write_logs=True):
    '''
    Defines and trains HIFIS-v2 model. Prints and logs relevant metrics.
    :param experiment: The type of training experiment. Choices are {'single_train'}
    :param save_weights: A flag indicating whether to save the model weights
    :param write_logs: A flag indicating whether to write TensorBoard logs
    :return: A dictionary of metrics on the test set
    '''

    # Load project config data
    cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))
    cur_date = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')

    # Load dataset file paths and labels
    data = {}
    data['TRAIN'] = pd.read_csv(cfg['PATHS']['TRAIN_SET'])
    data['VAL'] = pd.read_csv(cfg['PATHS']['VAL_SET'])
    data['TEST'] = pd.read_csv(cfg['PATHS']['TEST_SET'])

    # Set callbacks.
    early_stopping = EarlyStopping(monitor='val_loss', verbose=1, patience=cfg['TRAIN']['PATIENCE'], mode='min', restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001, verbose=1)
    callbacks = [early_stopping]
    if write_logs:
        log_dir = cfg['PATHS']['LOGS'] + "training\\" + cur_date
        tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)
        callbacks.append(tensorboard)

    # Conduct desired train experiment
    if experiment == 'single_train':
        model, test_metrics, test_generator = train_model(cfg, data, callbacks)

    # Visualization of test results
    test_predictions = model.predict_generator(test_generator, verbose=0)
    test_labels = test_generator.labels
    covid_idx = test_generator.class_indices['COVID-19']
    roc_img = plot_roc("Test set", test_labels, test_predictions, class_id=covid_idx, dir_path=None)
    cm_img = plot_confusion_matrix(test_labels, test_predictions, class_id=covid_idx, dir_path=None)

    # Log test set results and plots in TensorBoard
    if write_logs:
        writer = tf_summary.create_file_writer(logdir=log_dir)

        # Create table of test set metrics
        test_summary_str = [['**Metric**','**Value**']]
        thresholds = cfg['TRAIN']['THRESHOLDS']  # Load classification thresholds
        for metric in test_metrics:
            if metric in ['precision', 'recall'] and isinstance(metric, list):
                metric_values = dict(zip(thresholds, test_metrics[metric]))
            else:
                metric_values = test_metrics[metric]
            test_summary_str.append([metric, str(metric_values)])

        # Create table of model and train config values
        hparam_summary_str = [['**Variable**', '**Value**']]
        for key in cfg['TRAIN']:
            hparam_summary_str.append([key, str(cfg['TRAIN'][key])])
        if cfg['TRAIN']['CLASS_MODE'] == 'binary':
            for key in cfg['NN']['DCNN_BINARY']:
                hparam_summary_str.append([key, str(cfg['NN']['DCNN_BINARY'][key])])
        else:
            for key in cfg['NN']['DCNN_BINARY']:
                hparam_summary_str.append([key, str(cfg['NN']['DCNN_BINARY'][key])])

        # Write to TensorBoard logs
        with writer.as_default():
            tf_summary.text(name='Test set metrics', data=tf.convert_to_tensor(test_summary_str), step=0)
            tf_summary.text(name='Run hyperparameters', data=tf.convert_to_tensor(hparam_summary_str), step=0)
            tf_summary.image(name='ROC Curve (Test Set)', data=roc_img, step=0)
            tf_summary.image(name='Confusion Matrix (Test Set)', data=cm_img, step=0)

    # Save the model's weights
    if save_weights:
        model_path = os.path.splitext(cfg['PATHS']['MODEL_WEIGHTS'])[0] + cur_date + '.h5'
        save_model(model, model_path)
    return test_metrics


if __name__ == '__main__':
    cfg = yaml.full_load(open(os.getcwd() + "/config.yml", 'r'))
    results = train_experiment(experiment=cfg['TRAIN']['EXPERIMENT_TYPE'], save_weights=True, write_logs=True)