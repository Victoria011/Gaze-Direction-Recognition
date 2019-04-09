import Gazedetect
import dataGen
import glob
import dlib
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image

def open_img(name, flag, color = 'RGB'):
    """ Open an image 
    Args:
        name    : Name of the sample
        color   : Color Mode (RGB/BGR/GRAY)
    """
    if flag == 0: # img
        filename = os.path.join('../train_1/x', name)
    elif flag == 1: # gtMap left
        filename = os.path.join('train_1/left/', name)
    elif flag == 2: # gtMap right
        filename = os.path.join('train_1/right/', name)
    elif flag == 3: # gtMap right
        filename = os.path.join('train_1/gt_img/left/rotate/', name)
    elif flag == 4: # gtMap right
        filename = os.path.join('train_1/gt_img/right/rotate/', name)
        
    img = cv2.imread(filename)
    if color == 'RGB':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    elif color == 'BGR':
        return img
    elif color == 'GRAY':
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
    else:
        print('Color mode supported: RGB/BGR. If you need another mode do it yourself :p')

def _argmax(tensor):
    resh = tf.reshape(tensor, [-1]) # flatten into 1-D 64*64
    argmax = tf.argmax(resh,0)
    # NOTE: return row, col i.e. y coord, x coord
    return (argmax // tensor.get_shape().as_list()[0], argmax % tensor.get_shape().as_list()[0])

def _version(version):
	if version == 1:
		print('Dataset preparation version 1: 9 grid 64*64 coord')
	if version == 2:
		print('Dataset preparation version 1: 3*6 grid 64*64 coord')
	if version == 3:
		print('Dataset preparation version 1: center at exact eye center')


img_dir = '../train_1/x'
gt_dir = 'train_1/batch/y'
dataset = dataGen.DataGenerator(img_dir,gt_dir)
gazedetector = Gazedetect.Gazedetector('shape_predictor_68_face_landmarks.dat',enable_cuda=False)

count = 0
input_file = open('../dataset.txt', 'r')
for line in input_file:
    count+=1
    # if count == 3:
    #     break
    if line in ['\n', '\r\n']:
        print('READING end of file')
        break
    line = line.strip()
    line = line.split(' ')
    name = line[0]
    gtMap = line[1]
    direction = int(line[2]) 
    # eyes = list(map(int,line[3:])) # [11, 11, 11, 11]
    img = open_img(name,0)
    gtMap_l = open_img(gtMap,1,color = 'GRAY')
    gtMap_r = open_img(gtMap,2,color = 'GRAY')
    print(img.shape)
    print(gtMap_l.shape)

    # generate hm
    # filename = 'train_1/left/'+str(count)+'.png'
    # hm=dataset.generate_hm('xx.txt', direction = direction, size = 64)
    # img = Image.fromarray((hm[:,:,0]*255).astype(np.uint8))
    # img.save(filename,ext='png')

    # filename1 = 'train_1/right/'+str(count)+'.png'
    # img = Image.fromarray((hm[:,:,1]*255).astype(np.uint8))
    # img.save(filename1,ext='png')

    # rotate
    # image1,gtMap_l,gtMap_r = dataset._rotate(img, gtMap_l, gtMap_r,max_rotation = 30, rand=False)
    # tmp_l = tf.convert_to_tensor(gtMap_l, np.float32)
    # tmp_r = tf.convert_to_tensor(gtMap_r, np.float32)

    # img_path = str(count+301) + '.png'
    # filename1 = 'train_1/img/rotate/' + img_path
    # img = Image.fromarray(image1.astype(np.uint8))
    # img.save(filename1,ext='png')

    # filename2 = 'train_1/gt_img/left/rotate/' + img_path
    # img = Image.fromarray(gtMap_l.astype(np.uint8))
    # img.save(filename2,ext='png')

    # filename3 = 'train_1/gt_img/right/rotate/' + img_path
    # img = Image.fromarray(gtMap_r.astype(np.uint8))
    # img.save(filename3,ext='png')
    
    # flip
    # image2,gtMap_l,gtMap_r,r = dataset._flip(img,gtMap_l,gtMap_r, rand=False)
    # tmp_l = tf.convert_to_tensor(gtMap_l, np.float32)
    # tmp_r = tf.convert_to_tensor(gtMap_r, np.float32)

    # img_path = str(count+602) + '.png'
    # filename1 = 'train_1/img/flip/' + img_path
    # img = Image.fromarray(image2.astype(np.uint8))
    # img.save(filename1,ext='png')

    # filename2 = 'train_1/gt_img/left/flip/' + img_path
    # img = Image.fromarray(gtMap_l.astype(np.uint8))
    # img.save(filename2,ext='png')

    # filename3 = 'train_1/gt_img/right/flip/' + img_path
    # img = Image.fromarray(gtMap_r.astype(np.uint8))
    # img.save(filename3,ext='png')

    # color
    # image3,gtMap_l,gtMap_r = dataset._color_augment(img,gtMap_l,gtMap_r,rand=False)
    # tmp_l = tf.convert_to_tensor(gtMap_l, np.float32)
    # tmp_r = tf.convert_to_tensor(gtMap_r, np.float32)

    # img_path = str(count+903) + '.png'
    # filename1 = 'train_1/img/color/' + img_path
    # img = Image.fromarray(image3.astype(np.uint8))
    # img.save(filename1,ext='png')

    # filename2 = 'train_1/gt_img/left/color/' + img_path
    # img = Image.fromarray(gtMap_l.astype(np.uint8))
    # img.save(filename2,ext='png')

    # filename3 = 'train_1/gt_img/right/color/' + img_path
    # img = Image.fromarray(gtMap_r.astype(np.uint8))
    # img.save(filename3,ext='png')

    # combination
    image,gtMap_l,gtMap_r = dataset._rotate(img, gtMap_l, gtMap_r)
    image,gtMap_l,gtMap_r,r = dataset._flip(image,gtMap_l,gtMap_r)
    image,gtMap_l,gtMap_r = dataset._color_augment(image,gtMap_l,gtMap_r)
    tmp_l = tf.convert_to_tensor(gtMap_l, np.float32)
    tmp_r = tf.convert_to_tensor(gtMap_r, np.float32)

    img_path = str(count+1204) + '.png'
    filename1 = 'train_1/img/combination/' + img_path
    img = Image.fromarray(image.astype(np.uint8))
    img.save(filename1,ext='png')

    filename2 = 'train_1/gt_img/left/combination/' + img_path
    img = Image.fromarray(gtMap_l.astype(np.uint8))
    img.save(filename2,ext='png')

    filename3 = 'train_1/gt_img/right/combination/' + img_path
    img = Image.fromarray(gtMap_r.astype(np.uint8))
    img.save(filename3,ext='png')
    # max coord
    l_y,l_x = _argmax(tmp_l)
    r_y,r_x = _argmax(tmp_r)
    l_y,l_x,r_y,r_x = tf.Session().run([l_y,l_x,r_y,r_x])
    print('Before saving',l_x,l_y,r_x,r_y)
    pts1=[[l_x,l_y],[r_x,r_y]]

    if r == 1:
        if direction == 1:
            direction = 3
        elif direction == 3:
            direction = 1
        elif direction == 4:
            direction = 6
        elif direction == 6:
            direction = 4
        elif direction == 7:
            direction = 9
        elif direction == 9:
            direction = 7

    dataset.write_txt(img_path, img_path, direction,pts1,'test.txt')

    # just validation
    # gtMap_l = open_img(img_path,3,color = 'GRAY')
    # gtMap_r = open_img(img_path,4,color = 'GRAY')
    # tmp_l = tf.convert_to_tensor(gtMap_l, np.float32)
    # tmp_r = tf.convert_to_tensor(gtMap_r, np.float32)
    # l_y,l_x = _argmax(tmp_l)
    # r_y,r_x = _argmax(tmp_r)
    # l_y,l_x,r_y,r_x = tf.Session().run([l_y,l_x,r_y,r_x])
    # print('After saving',l_y,l_x,r_y,r_x)

    
    
input_file.close()

