import keras
import cv2
import numpy as np
import h5py
import os
# from keras import backend as K
from keras.models import Model
from keras.layers import Input,Dense,Dropout,Flatten
from keras.applications import VGG19, ResNet50
from keras import losses


def get_name_list(filepath):  # 获取各个类别的名字
    pathDir = os.listdir(filepath)
    out = []
    for allDir in pathDir:
        if os.path.isdir(os.path.join(filepath, allDir)):
            child = allDir.encode('gbk').decode('gbk')  # .decode('gbk')是解决中文显示乱码问题
            out.append(child)
    return out


def eachFile(filepath):  # 将目录内的文件名放入列表中
    pathDir = os.listdir(filepath)
    out = []
    for allDir in pathDir:
        child = allDir.encode('gbk').decode('gbk')  # .decode('gbk')是解决中文显示乱码问题
        out.append(child)
    return out



def get_data(data_name, train_left=0.0, train_right=0.7, train_all=0.7, resize=True, data_format=None,
             t=''):  # 从文件夹中获取图像数据
    file_name = os.path.join(pic_dir_out, data_name + t + '_' + str(train_left) + '_' + str(train_right) + '_' + str(
        Width) + "X" + str(Height) + ".h5")
    print(file_name)

    if os.path.exists(file_name):  # 判断之前是否有存到文件中
        f = h5py.File(file_name, 'r')
        if t == 'train':
            X_train = f['X_train'][:]
            y_train = f['y_train'][:]
            f.close()
            return (X_train, y_train)
        elif t == 'test':
            X_test = f['X_test'][:]
            y_test = f['y_test'][:]
            f.close()
            return (X_test, y_test)
        else:
            return
    # data_format = conv_utils.normalize_data_format(data_format)
    pic_dir_set = eachFile(pic_dir_data)
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    label = 0
    for pic_dir in pic_dir_set:
        print( pic_dir_data + pic_dir)
        if not os.path.isdir(os.path.join(pic_dir_data, pic_dir)):
            continue
        pic_set = eachFile(os.path.join(pic_dir_data, pic_dir))
        pic_index = 0
        train_count = int(len(pic_set) * train_all)
        train_l = int(len(pic_set) * train_left)
        train_r = int(len(pic_set) * train_right)
        for pic_name in pic_set:
            if not os.path.isfile(os.path.join(pic_dir_data, pic_dir, pic_name)):
                continue
            img = cv2.imread(os.path.join(pic_dir_data, pic_dir, pic_name))
            #print img.shape
            if img is None:
                continue
            if (resize):
                img = cv2.resize(img, (Width, Height))
                img = img.reshape(-1, Width, Height, 3)
            if (pic_index < train_count):
                if t == 'train':
                    if (pic_index >= train_l and pic_index < train_r):
                        X_train.append(img)
                        y_train.append(label)
            else:
                if t == 'test':
                    X_test.append(img)
                    y_test.append(label)
            pic_index += 1
        if len(pic_set) != 0:
            label += 1

    f = h5py.File(file_name, 'w')
    if t == 'train':
        X_train = np.concatenate(X_train, axis=0)
        y_train = np.array(y_train)
        f.create_dataset('X_train', data=X_train)
        f.create_dataset('y_train', data=y_train)
        f.close()
        return (X_train, y_train)
    elif t == 'test':
        X_test = np.concatenate(X_test, axis=0)
        y_test = np.array(y_test)
        f.create_dataset('X_test', data=X_test)
        f.create_dataset('y_test', data=y_test)
        f.close()
        return (X_test, y_test)
    else:
        return


def get_X_train(data_name, resize = True,):
    file_name = os.path.join(pic_dir_out, data_name + 'train' + '_' + '.h5')
    # if already stored
    if os.path.exists(file_name):
        f = h5py.File(file_name, 'r')
        X_train = f['X_train'][:]
        y_train = f['y_train'][:]
        return (X_train, y_train)

    X_train = []
    pic_set = eachFile(os.path.join(pic_dir_data, pic_dir))
    # print(pic_set)
    # get_pics and resize
    for pic in pic_set:
        if not os.path.isfile(os.path.join(pic_dir_data, pic_dir, pic)):
            continue
        img = cv2.imread(os.path.join(pic_dir_data, pic_dir, pic))
        # print(img.shape)
        if img is None:
            continue
        if (resize):
            img = cv2.resize(img, (Width, Height))
            img = img.reshape(-1, Width, Height, 3)
        X_train.append(img)

    # get tags for  multi_label classification
    y_train = []
    tags_set = eachFile(os.path.join(pic_dir_data, tags_dir))
    # print(tags_set)
    for tags in tags_set:
        labels = [0] * 91
        angle = []
        score = []
        f = open(os.path.join(pic_dir_data, tags_dir, tags))
        lines = f.readlines()
        for line in lines:
            line.replace('\n', '')
            _angle, _score = line.split()
            _angle,_ = _angle.split('.')
            angle.append(int(_angle))
            score.append(int(_score))
        order = sorted(zip(score,angle))
        labels[order[-1][1]] = 1
        for i in range(len(order)-2, -1, -1):
            if order[i][0] / order[-1][0] >= 0.85:
                labels[order[i][1]] = 1
            else:
                break
        # print(labels)
        y_train.append(labels)
        f.close()

    # write x_train, y_train
    f = h5py.File(file_name, 'w')
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.array(y_train)
    f.create_dataset('X_train', data=X_train)
    f.create_dataset('y_train', data=y_train)
    f.close()

    return (X_train, y_train)

def main():
    global Width, Height, pic_dir_out, pic_dir_data, pic_dir, tags_dir
    Width = 224
    Height = 224

    pic_dir_out = 'out'
    sub_dir = 'model'
    pic_dir_data = 'data'
    pic_dir = 'pic'
    tags_dir = 'tags'

    pic_dir_mine = os.path.join(pic_dir_out, sub_dir)

    (X_train, y_train) = get_X_train("angry_birds")
    X_train = X_train.reshape(-1, Width, Height, 3)
    y_train = y_train.reshape(-1, 91)



    # input size and batch size
    _input_image_size = [Width,Height,3]
    batch_size = len(X_train)
    epochs = 5000


    inputs = Input(shape = _input_image_size, dtype = 'float32', name = 'model_input')
    # vgg_based = VGG19(include_top = False, weights = 'imagenet', input_shape = input_image_size)
    # feature_output = vgg_based(inputs)

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
    model.summary()
    model.compile(optimizer='adagrad', loss='binary_crossentropy', metrics=['accuracy'])
    tbCallBack = keras.callbacks.TensorBoard(log_dir='Graph1', histogram_freq=1, write_graph=True, write_images=True)

    cm = 0  # change for continuing training
    cm_str = '' if cm == 0 else str(cm)
    cm2_str = '' if (cm + 1) == 0 else str(cm + 1)
    if cm >= 1:
        model.load_weights(os.path.join(pic_dir_mine, 'resnet50_based_' + cm_str + '.h5'))
    history = model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, shuffle = True)
    model_history = model.fit(X_train,
                              y_train,
                              batch_size=batch_size,
                              epochs=epochs,
                              verbose=1,
                              validation_split=0.2,callbacks=[tbCallBack])
    # model.save_weights(os.path.join(pic_dir_mine, 'resnet50_based_' + cm2_str + '.h5'))



    # acc of each epoch
    accy = history.history['acc']
    np_accy = np.array(accy)
    print(np_accy)

    # pred and groundtruth comparision
    results = model.predict(X_train)
    results = np.round(results)
    pred = results.astype(np.int)
    ground_truth = y_train.astype(np.int)
    np.savetxt('comparision.txt', np.concatenate((results, ground_truth),1))

    # acc on each angle
    temp = 1 - np.sum(abs(pred - ground_truth),axis=0) / results.shape[0]
    print(temp)
    print(len(temp))
    np.savetxt('acc_on_each_angle.txt', temp)

if __name__ == '__main__':
    main()