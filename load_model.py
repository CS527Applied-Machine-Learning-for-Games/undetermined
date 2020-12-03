import numpy as np
import cv2
import os
import keras
from keras.models import Model
from keras.layers import Input,Dense,Dropout,Flatten
from keras.applications import  ResNet50
from keras.callbacks import TensorBoard

def main():
    Width = 224
    Height = 224
    _input_image_size = [Width,Height,3]

    # file = 'D:\\PycharmProjects\\527\\data\\pic\\-1.png'
    file = 'C:\\Users\\zyx_hhxx\\Desktop\\data\\9\\initial.png'
    img = cv2.imread(file)
    # print(img)
    img = cv2.resize(img, (Width, Height))
    X = img.reshape(-1, Width, Height, 3)
    # print(X.shape)
    # print(type(X))

    pic_dir_out = 'out'
    sub_dir = 'model'
    pic_dir_mine = os.path.join(pic_dir_out, sub_dir)

    inputs = Input(shape = _input_image_size, dtype = 'float32', name = 'model_input')


    _model = ResNet50(include_top=False, weights='imagenet')
    feature_output = _model(inputs)
    feature_output = Flatten()(feature_output)

    dense = Dense(200, activation='relu')(feature_output)
    dense = Dropout(0.5)(dense)
    dense = Dense(100, activation='relu')(dense)
    dense = Dropout(0.5)(dense)
    # dense = Dense(91,activation='relu')(dense)
    # dense = Dropout(0.5)(dense)
    outputs = Dense(91, activation='sigmoid', name = 'final_output')(dense)

    model = Model(inputs = inputs, outputs = outputs)

    cm = 0  # change for continuing training
    cm_str = '' if cm == 0 else str(cm)
    cm2_str = '' if (cm + 1) == 0 else str(cm + 1)

    model.load_weights(os.path.join(pic_dir_mine, 'resnet50_based_' + cm2_str + '.h5'))
    results = model.predict(X)


if __name__ == '__main__':
    main()
