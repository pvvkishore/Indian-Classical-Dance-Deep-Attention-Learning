#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 15:23:17 2024

@author: root
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 13 13:42:30 2023

@author: root
"""

import os
import tensorflow as tf
import cv2
import numpy as np
batch_size = 3
img_height = 256
img_width = 256
#input_shape = (256, 256, 3)
dataset_path = os.listdir('Dance Dataset_2024')
dance_types = os.listdir('Dance Dataset_2024/Frame_1')
dance_types = sorted(dance_types)
print(dance_types)
#%%
""" Store all the image_class of frame_1 list """
def load_images_from_folder(folder_path_1):
    images_Frame_1 = []
    labels_Frame_1 = []
    class_labels = {}
    class_label_counter = 0
    
    for frame_name, class_label in enumerate(os.listdir(folder_path_1)):
        class_path = os.path.join(folder_path_1, class_label)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                img_path = os.path.join(class_path, filename)
                if os.path.isfile(img_path):
                    img = cv2.imread(img_path)
                    img = cv2.resize(img,(img_height,img_width))
                    images_Frame_1.append(img/255.0)
                    labels_Frame_1.append(class_label_counter)
            class_label_counter += 1         
    return np.array(images_Frame_1), np.array(labels_Frame_1),class_labels

folder_path_1 = "/home/user/Desktop/Classical Dance Identification_DL/Dance Dataset_2024/Frame_1"
images_Frame_1,labels_Frame_1,class_labels = load_images_from_folder(folder_path_1)
#%%
""" Store all the image_class of frame_1 list """
def load_images_from_folder(folder_path_1):
    images_Frame_2 = []
    labels_Frame_2 = []
    class_labels = {}
    class_label_counter = 0
    
    for frame_name, class_label in enumerate(os.listdir(folder_path_1)):
        class_path = os.path.join(folder_path_1, class_label)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                img_path = os.path.join(class_path, filename)
                if os.path.isfile(img_path):
                    img = cv2.imread(img_path)
                    img = cv2.resize(img,(img_height,img_width))
                    images_Frame_2.append(img/255.0)
                    labels_Frame_2.append(class_label_counter)
            class_label_counter += 1         
    return np.array(images_Frame_2), np.array(labels_Frame_2),class_labels

folder_path_1 = "/home/user/Desktop/Classical Dance Identification_DL/Dance Dataset_2024/Frame_2"
images_Frame_2,labels_Frame_2,class_labels = load_images_from_folder(folder_path_1)
#%%
""" Store all the image_class of frame_1 list """
def load_images_from_folder(folder_path_1):
    images_Frame_3 = []
    labels_Frame_3 = []
    class_labels = {}
    class_label_counter = 0
    
    for frame_name, class_label in enumerate(os.listdir(folder_path_1)):
        class_path = os.path.join(folder_path_1, class_label)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                img_path = os.path.join(class_path, filename)
                if os.path.isfile(img_path):
                    img = cv2.imread(img_path)
                    img = cv2.resize(img,(img_height,img_width))
                    images_Frame_3.append(img/255.0)
                    labels_Frame_3.append(class_label_counter)
            class_label_counter += 1         
    return np.array(images_Frame_3), np.array(labels_Frame_3),class_labels

folder_path_1 = "/home/user/Desktop/Classical Dance Identification_DL/Dance Dataset_2024/Frame_3"
images_Frame_3,labels_Frame_3,class_labels = load_images_from_folder(folder_path_1)
#%%
""" Store all the image_class of all frames list """
def load_images_from_folder(folder_path_1):
    images_Frame_1 = []
    labels_Frame_1 = []
    class_labels = {}
    class_label_counter = 0
    
    for frame_name, class_label in enumerate(os.listdir(folder_path_1)):
        class_path = os.path.join(folder_path_1, class_label)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                img_path = os.path.join(class_path, filename)
                if os.path.isfile(img_path):
                    img = cv2.imread(img_path)
                    img = cv2.resize(img,(img_height,img_width))
                    images_Frame_1.append(img/255.0)
                    labels_Frame_1.append(class_label_counter)
            class_label_counter += 1         
    return np.array(images_Frame_1), np.array(labels_Frame_1),class_labels

folder_path_1 = "/home/user/Desktop/Classical Dance Identification_DL/Dance Dataset_2024/Frame_all"
images_Frame_All,labels_Frame_1,class_labels = load_images_from_folder(folder_path_1)
#%%
# import matplotlib.pyplot as plt
# fig, axes = plt.subplots(3,3,figsize=(10,10))
# for i in range(3):
#     for j in range(3):
#         axes[i,j].imshow(images[j+i])
#         axes[i,j].axis('off')
#         print(i)
# cv2.imshow('Enhanced',images[10])
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#%%
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, Add, GlobalAveragePooling2D, Dense, Concatenate, Multiply
def single_head_single_frame_attention_block(x,filters,block_name = 'shsf_attention'):
    f = filters
    x1 = Conv2D(f,(1,1),strides = 1,padding = 'same', activation = 'relu')(x)
    x2 = Conv2D(f,(1,1),strides = 1,padding = 'same', activation = 'relu')(x)
    x3 = Conv2D(f,(1,1),strides = 1,padding = 'same', activation = 'relu')(x)
    x12 = Multiply()([x1,x2])
    x = Multiply()([x12,x3])
    x = Conv2D(f,(1,1), strides = 1, padding = 'same', activation = 'relu')(x)
    x = Add()([x,x1])
    x = Add()([x,x3])
    x = Conv2D(f,(1,1),strides = 1, padding = 'same', activation = 'relu')(x)
    x_branch = Conv2D(1,(1,1),strides = 1, padding = 'same', activation = 'relu')(x)
    x_activation_branch = Conv2D(f,(1,1),strides = 1, padding = 'same', activation = 'sigmoid')(x)
    x = Multiply()([x_branch,x_activation_branch])
    x = Conv2D(f,(1,1),strides = 1, padding = 'same', activation = 'relu')(x)
    return x
#%%
def identity_block(x, filters, kernel_size=3, block_name='identity_block'):
    f1, f2, f3 = filters
    
    x_shortcut = x

    x = Conv2D(f1, (1, 1), name=block_name + '_conv1')(x)
    x = BatchNormalization(name=block_name + '_bn1')(x)
    x = ReLU()(x)

    x = Conv2D(f2, kernel_size, padding='same', name=block_name + '_conv2')(x)
    x = BatchNormalization(name=block_name + '_bn2')(x)
    x = ReLU()(x)

    x = Conv2D(f3, (1, 1), name=block_name + '_conv3')(x)
    x = BatchNormalization(name=block_name + '_bn3')(x)

    x = Add()([x, x_shortcut])
    x = ReLU()(x)

    return x
#%%
def conv_block(x, filters, kernel_size=3, stride=2, block_name='conv_block'):
    f1, f2, f3 = filters

    x_shortcut = x

    x = Conv2D(f1, (1, 1), strides=stride, name=block_name + '_conv1')(x)
    x = BatchNormalization(name=block_name + '_bn1')(x)
    x = ReLU()(x)

    x = Conv2D(f2, kernel_size, padding='same', name=block_name + '_conv2')(x)
    x = BatchNormalization(name=block_name + '_bn2')(x)
    x = ReLU()(x)

    x = Conv2D(f3, (1, 1), name=block_name + '_conv3')(x)
    x = BatchNormalization(name=block_name + '_bn3')(x)

    x_shortcut = Conv2D(f3, (1, 1), strides=stride, name=block_name + '_shortcut_conv')(x_shortcut)
    x_shortcut = BatchNormalization(name=block_name + '_shortcut_bn')(x_shortcut)

    x = Add()([x, x_shortcut])
    x = ReLU()(x)

    return x
#%%
def build_resnet50_model(input_shape=(256, 256, 3), num_classes=8, model_name='resnet50'):
    input_tensor_source_1 = Input(shape=input_shape, name=f'{model_name}_input_source1')
    input_tensor_source_2 = Input(shape=input_shape, name=f'{model_name}_input_source2')
    input_tensor_source_3 = Input(shape=input_shape, name=f'{model_name}_input_source3')
    input_tensor_source_4 = Input(shape=input_shape, name=f'{model_name}_input_source4')
    
    # Frame_1 Source_1 Resnet 50 Branch
    x1 = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name=f'{model_name}_conv1')(input_tensor_source_1)
    x1 = BatchNormalization(name=f'{model_name}_bn_conv1')(x1)
    x1 = ReLU()(x1)

    x1 = conv_block(x1, [64, 64, 256], stride=1, block_name=f'{model_name}_conv_block2')
    x1 = identity_block(x1, [64, 64, 256], block_name=f'{model_name}_identity_block2_1')
    x1 = identity_block(x1, [64, 64, 256], block_name=f'{model_name}_identity_block2_2')
    x1a_64 = single_head_single_frame_attention_block(x1,filters = 256,block_name = f'{model_name}_shsf_attention')
    x1 = Multiply()([x1,x1a_64])
    print("Dimensions at the end of res_block1:", x1.shape)
    
    x1 = conv_block(x1, [128, 128, 512], block_name=f'{model_name}_conv_block3')
    x1 = identity_block(x1, [128, 128, 512], block_name=f'{model_name}_identity_block3_1')
    x1 = identity_block(x1, [128, 128, 512], block_name=f'{model_name}_identity_block3_2')
    x1 = identity_block(x1, [128, 128, 512], block_name=f'{model_name}_identity_block3_3')
    x1a_32 = single_head_single_frame_attention_block(x1,filters = 512, block_name = f'{model_name}_shsf_attention')
    x = Multiply()([x1,x1a_32])
    print("Dimensions at the end of res_block2:", x1.shape)

    x1 = conv_block(x1, [256, 256, 1024], block_name=f'{model_name}_conv_block4')
    x1 = identity_block(x1, [256, 256, 1024], block_name=f'{model_name}_identity_block4_1')
    x1 = identity_block(x1, [256, 256, 1024], block_name=f'{model_name}_identity_block4_2')
    x1 = identity_block(x1, [256, 256, 1024], block_name=f'{model_name}_identity_block4_3')
    x1 = identity_block(x1, [256, 256, 1024], block_name=f'{model_name}_identity_block4_4')
    x1 = identity_block(x1, [256, 256, 1024], block_name=f'{model_name}_identity_block4_5')
    x1a_16 = single_head_single_frame_attention_block(x1,filters = 1024, block_name = f'{model_name}_shsf_attention')
    x1 = Multiply()([x1,x1a_16])
    print("Dimensions at the end of res_block3:", x1.shape)
    
    x1 = conv_block(x1, [512, 512, 2048], block_name=f'{model_name}_conv_block5')
    x1 = identity_block(x1, [512, 512, 2048], block_name=f'{model_name}_identity_block5_1')
    x1 = identity_block(x1, [512, 512, 2048], block_name=f'{model_name}_identity_block5_2')
    x1a_8 = single_head_single_frame_attention_block(x1,filters = 2048, block_name = f'{model_name}_shsf_attention')
    x1 = Multiply()([x1,x1a_8])
    print("Dimensions at the end of res_block4:", x1.shape)
    
    # Frame_2 Source_2 Resnet 50 Branch
    x1 = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name=f'{model_name}_2_conv1')(input_tensor_source_2)
    x1 = BatchNormalization(name=f'{model_name}_bn_2_conv1')(x1)
    x1 = ReLU()(x1)

    x1 = conv_block(x1, [64, 64, 256], stride=1, block_name=f'{model_name}_conv_2_block2')
    x1 = identity_block(x1, [64, 64, 256], block_name=f'{model_name}_identity_2_block2_1')
    x1 = identity_block(x1, [64, 64, 256], block_name=f'{model_name}_identity_2_block2_2')
    x2a_64 = single_head_single_frame_attention_block(x1,filters = 256,block_name = f'{model_name}_2_shsf_attention')
    #x1 = Multiply()([x1,x2a_64])
    print("Dimensions at the end of res_block1:", x1.shape)
    
    x1 = conv_block(x1, [128, 128, 512], block_name=f'{model_name}_conv_2_block3')
    x1 = identity_block(x1, [128, 128, 512], block_name=f'{model_name}_identity_2_block3_1')
    x1 = identity_block(x1, [128, 128, 512], block_name=f'{model_name}_identity_2_block3_2')
    x1 = identity_block(x1, [128, 128, 512], block_name=f'{model_name}_identity_2_block3_3')
    x2a_32 = single_head_single_frame_attention_block(x1,filters = 512, block_name = f'{model_name}_2_shsf_attention')
    #x = Multiply()([x1,x1a_32])
    print("Dimensions at the end of res_block2:", x1.shape)

    x1 = conv_block(x1, [256, 256, 1024], block_name=f'{model_name}_conv_2_block4')
    x1 = identity_block(x1, [256, 256, 1024], block_name=f'{model_name}_identity_2_block4_1')
    x1 = identity_block(x1, [256, 256, 1024], block_name=f'{model_name}_identity_2_block4_2')
    x1 = identity_block(x1, [256, 256, 1024], block_name=f'{model_name}_identity_2_block4_3')
    x1 = identity_block(x1, [256, 256, 1024], block_name=f'{model_name}_identity_2_block4_4')
    x1 = identity_block(x1, [256, 256, 1024], block_name=f'{model_name}_identity_2_block4_5')
    x2a_16 = single_head_single_frame_attention_block(x1,filters = 1024, block_name = f'{model_name}_2_shsf_attention')
    #x1 = Multiply()([x1,x1a_16])
    print("Dimensions at the end of res_block3:", x1.shape)
    
    x1 = conv_block(x1, [512, 512, 2048], block_name=f'{model_name}_conv_2_block5')
    x1 = identity_block(x1, [512, 512, 2048], block_name=f'{model_name}_identity_2_block5_1')
    x1 = identity_block(x1, [512, 512, 2048], block_name=f'{model_name}_identity_2_block5_2')
    x2a_8 = single_head_single_frame_attention_block(x1,filters = 2048, block_name = f'{model_name}_2_shsf_attention')
    #x1 = Multiply()([x1,x1a_8])
    print("Dimensions at the end of res_block4:", x1.shape)
    
    # Frame_3 Source_3 Resnet 50 Branch
    x1 = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name=f'{model_name}_3_conv1')(input_tensor_source_3)
    x1 = BatchNormalization(name=f'{model_name}_bn_3_conv1')(x1)
    x1 = ReLU()(x1)

    x1 = conv_block(x1, [64, 64, 256], stride=1, block_name=f'{model_name}_conv_3_block2')
    x1 = identity_block(x1, [64, 64, 256], block_name=f'{model_name}_identity_3_block2_1')
    x1 = identity_block(x1, [64, 64, 256], block_name=f'{model_name}_identity_3_block2_2')
    x3a_64 = single_head_single_frame_attention_block(x1,filters = 256,block_name = f'{model_name}_3_shsf_attention')
    #x1 = Multiply()([x1,x1a_64])
    print("Dimensions at the end of res_block1:", x1.shape)
    
    x1 = conv_block(x1, [128, 128, 512], block_name=f'{model_name}_3_conv_block3')
    x1 = identity_block(x1, [128, 128, 512], block_name=f'{model_name}_identity_3_block3_1')
    x1 = identity_block(x1, [128, 128, 512], block_name=f'{model_name}_identity_3_block3_2')
    x1 = identity_block(x1, [128, 128, 512], block_name=f'{model_name}_identity_3_block3_3')
    x3a_32 = single_head_single_frame_attention_block(x1,filters = 512, block_name = f'{model_name}_3_shsf_attention')
    #x = Multiply()([x1,x1a_32])
    print("Dimensions at the end of res_block2:", x1.shape)

    x1 = conv_block(x1, [256, 256, 1024], block_name=f'{model_name}_3_conv_block4')
    x1 = identity_block(x1, [256, 256, 1024], block_name=f'{model_name}_identity_3_block4_1')
    x1 = identity_block(x1, [256, 256, 1024], block_name=f'{model_name}_identity_3_block4_2')
    x1 = identity_block(x1, [256, 256, 1024], block_name=f'{model_name}_identity_3_block4_3')
    x1 = identity_block(x1, [256, 256, 1024], block_name=f'{model_name}_identity_3_block4_4')
    x1 = identity_block(x1, [256, 256, 1024], block_name=f'{model_name}_identity_3_block4_5')
    x3a_16 = single_head_single_frame_attention_block(x1,filters = 1024, block_name = f'{model_name}_3_shsf_attention')
    #x1 = Multiply()([x1,x1a_16])
    print("Dimensions at the end of res_block3:", x1.shape)
    
    x1 = conv_block(x1, [512, 512, 2048], block_name=f'{model_name}_3_conv_block5')
    x1 = identity_block(x1, [512, 512, 2048], block_name=f'{model_name}_identity_3_block5_1')
    x1 = identity_block(x1, [512, 512, 2048], block_name=f'{model_name}_identity_3_block5_2')
    x3a_8 = single_head_single_frame_attention_block(x1,filters = 2048, block_name = f'{model_name}_3_shsf_attention')
    #x1 = Multiply()([x1,x1a_8])
    print("Dimensions at the end of res_block4:", x3a_8.shape)
    
    # Compute Multi Frame Multi_Head_Attention and Generate an average Attention Map
    xa_64_avg = tf.reduce_mean(tf.stack([x1a_64,x2a_64,x3a_64],axis=-1),axis = -1)
    xa_32_avg = tf.reduce_mean(tf.stack([x1a_32,x2a_32,x3a_32],axis=-1),axis = -1)
    xa_16_avg = tf.reduce_mean(tf.stack([x1a_16,x2a_16,x3a_16],axis=-1),axis = -1)
    xa_8_avg = tf.reduce_mean(tf.stack([x1a_8,x2a_8,x3a_8],axis=-1),axis = -1)
    
    # Source_4 Resnet 50 Branch
    x1 = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name=f'{model_name}_4_conv1')(input_tensor_source_4)
    x1 = BatchNormalization(name=f'{model_name}_bn_4_conv1')(x1)
    x1 = ReLU()(x1)

    x1 = conv_block(x1, [64, 64, 256], stride=1, block_name=f'{model_name}_conv_4_block2')
    x1 = identity_block(x1, [64, 64, 256], block_name=f'{model_name}_identity_4_block2_1')
    x1 = identity_block(x1, [64, 64, 256], block_name=f'{model_name}_identity_4_block2_2')
    x1 = Multiply()([x1,xa_64_avg])
    print("Dimensions at the end of res_block1:", x1.shape)
    
    x1 = conv_block(x1, [128, 128, 512], block_name=f'{model_name}_conv_4_block3')
    x1 = identity_block(x1, [128, 128, 512], block_name=f'{model_name}_identity_4_block3_1')
    x1 = identity_block(x1, [128, 128, 512], block_name=f'{model_name}_identity_4_block3_2')
    x1 = identity_block(x1, [128, 128, 512], block_name=f'{model_name}_identity_4_block3_3')
    x1 = Multiply()([x1,xa_32_avg])
    print("Dimensions at the end of res_block2:", x1.shape)

    x1 = conv_block(x1, [256, 256, 1024], block_name=f'{model_name}_conv_4_block4')
    x1 = identity_block(x1, [256, 256, 1024], block_name=f'{model_name}_identity_4_block4_1')
    x1 = identity_block(x1, [256, 256, 1024], block_name=f'{model_name}_identity_4_block4_2')
    x1 = identity_block(x1, [256, 256, 1024], block_name=f'{model_name}_identity_4_block4_3')
    x1 = identity_block(x1, [256, 256, 1024], block_name=f'{model_name}_identity_4_block4_4')
    x1 = identity_block(x1, [256, 256, 1024], block_name=f'{model_name}_identity_4_block4_5')
    x1 = Multiply()([x1,xa_16_avg])
    print("Dimensions at the end of res_block3:", x1.shape)
    
    x1 = conv_block(x1, [512, 512, 2048], block_name=f'{model_name}_conv_4_block5')
    x1 = identity_block(x1, [512, 512, 2048], block_name=f'{model_name}_identity_4_block5_1')
    x1 = identity_block(x1, [512, 512, 2048], block_name=f'{model_name}_identity_4_block5_2')
    x1 = Multiply()([x1,xa_8_avg])
    print("Dimensions at the end of res_block4:", x3a_8.shape)
    
    x = GlobalAveragePooling2D(name=f'{model_name}_global_avg_pooling')(x1)

    x = Dense(num_classes, activation='softmax', name=f'{model_name}_output')(x)

    # Model with intermediate outputs
    model = tf.keras.models.Model(
        inputs=[input_tensor_source_1,input_tensor_source_2,input_tensor_source_3,input_tensor_source_4],
        outputs = x,
        name=model_name
    )
#outputs=[output_res_block1, output_res_block2, output_res_block3, output_res_block4, x]
    return model
#%%
# Build three similar ResNet-50 models for three different dataset sources
input_shape = (256, 256, 3)
num_classes = 8

model_combined_sources = build_resnet50_model(
    input_shape=input_shape, num_classes=num_classes, model_name='resnet50_combined_sources')
# Compile the combined model for training
optimizer = tf.keras.optimizers.Adam(learning_rate=0.000001)
model_combined_sources.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Display the summary of the combined model
model_combined_sources.summary()
#%%
# Convert labels to one-hot encoding
import numpy as np
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.utils import to_categorical
labels_one_hot_F1 = to_categorical(labels_Frame_1, num_classes=8)
labels_one_hot_F2 = to_categorical(labels_Frame_2, num_classes=8)
labels_one_hot_F3 = to_categorical(labels_Frame_3, num_classes=8)
train_data_source1 = images_Frame_1
train_data_source2 = images_Frame_2
train_data_source3 = images_Frame_3
train_data_source4 = images_Frame_All
train_labels_source1 = labels_one_hot_F1
train_labels_source2 = labels_one_hot_F2
train_labels_source3 = labels_one_hot_F3
#%%
# Train the combined model
epochs = 15
batch_size = 1

# Replace the following with your actual training data and parameters
history = model_combined_sources.fit(
    [train_data_source1, train_data_source2, train_data_source3,train_data_source4],
    train_labels_source1,
    epochs=epochs,
    batch_size=batch_size
    )
#validation_data=([val_data_source1, val_data_source2, val_data_source3], val_labels)  
#%%
# import keras
# model = combined_model
# model.save('combined_model.keras')
# #model.save('model')
# %%
import matplotlib.pyplot as plt
acc = history.history['accuracy']
#val_acc = history.history['val_accuracy']

loss = history.history['loss']
#al_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, loss, label='Training Loss')
plt.legend(loc='upper right')
plt.title('Training and Loss')
plt.show()
# %% TEST the model
# Load Test Data from all the frames
""" Store all the image_class of all frames list """
def load_images_from_folder(folder_path_1):
    images_Frame_1 = []
    labels_Frame_1 = []
    class_labels = {}
    class_label_counter = 0
    
    for frame_name, class_label in enumerate(os.listdir(folder_path_1)):
        class_path = os.path.join(folder_path_1, class_label)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                img_path = os.path.join(class_path, filename)
                if os.path.isfile(img_path):
                    img = cv2.imread(img_path)
                    img = cv2.resize(img,(img_height,img_width))
                    images_Frame_1.append(img/255.0)
                    labels_Frame_1.append(class_label_counter)
            class_label_counter += 1         
    return np.array(images_Frame_1), np.array(labels_Frame_1),class_labels

folder_path_1 = "/home/user/Desktop/Classical Dance Identification_DL/Dance Dataset_2024/Test_Data"
images_Frames_Test,labels_Frame_Test,class_labels = load_images_from_folder(folder_path_1)
#%%
predictions = model_combined_sources.predict([images_Frames_Test, images_Frames_Test, images_Frames_Test,images_Frames_Test])
# %%
test_loss, test_accuracy = model_combined_sources.evaluate([images_Frames_Test, images_Frames_Test, images_Frames_Test,images_Frames_Test])
print('Test accuracy:', test_accuracy)
print('Test Loss:', test_loss)
#%%
def show_batch(images, labels, predictions=None):
    plt.figure(figsize=(10, 10))
    # min = images.numpy().min()
    # max = images.numpy().max()
    # delta = max - min

    for i in range(24):
        plt.subplot(6, 6, i + 1)
        plt.imshow(images[i])
        if predictions is None:
            plt.title(labels_Frame_Test[i])
        else:
            if labels_Frame_Test[i] == predictions[i]:
                color = 'g'
            else:
                color = 'r'
            plt.title(predictions[i], color=color)
        plt.axis("off")
predictions_max = tf.argmax(predictions, axis=-1)
for test_images, test_labels in zip(images_Frames_Test,labels_Frame_Test):
    show_batch(test_images, test_labels, tf.cast(predictions_max, tf.int32))