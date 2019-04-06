# For testing different code section
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

gazedetector = Gazedetect.Gazedetector('shape_predictor_68_face_landmarks.dat',enable_cuda=False)
# path_imgs = 'Dissertation/local_Dataset/video_frame/front/frame1.png'
# img = dlib.load_rgb_image(path_imgs)
# gtMap = cv2.resize(img, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
# filename = 'Dissertation/local_Dataset/video_frame/front/frame12.png'
# plt.imsave(filename, gtMap)
# pts = gazedetector.getPts(path_imgs)
# print(pts)
# files = sorted(glob.glob(path_imgs + '/*.png'))
# fig = plt.figure(figsize=plt.figaspect(.5))
# count = 0
# for names in files:
#     # print(names)
#     img = dlib.load_rgb_image(names)
#     count+=1
#     filename = 'Dissertation/local_Dataset/video_frame/front/new/frame'+str(count)+'.png'
#     output = Gazedetector.detectGaze(img,filename)

    # pred,map,img = Gazedetector.detectGaze(img)
    # for j in range(0,5):
    #     resized_map = dlib.resize_image(map[j,:,:].cpu().data.numpy(),rows=256,cols=256)
    #     ax = fig.add_subplot(5,2,2*j+1)
    #     ax.imshow(img)
    #     ax.axis('off')
    #     ax = fig.add_subplot(5, 2, 2*j+2)
    #     ax.imshow(resized_map)
    #     ax.axis('off')
    # plt.pause(.1)
    # plt.draw()

# video processing
# video_path = 'Dissertation/local_Dataset/Sessions/10/UserFrontal_C_Prudence.avi'
# gazedetector.process_video(video_path,True,'Dissertation/local_Dataset/video_frame/10/UserFrontal_C_Prudence')

# procrustes transform imgs in a dir
# img_dir = 'Dissertation/local_Dataset/video_frame/5/Frontal_C_Obadiah'
# out_dir = 'Dissertation/local_Dataset/video_frame/5/trainC'
# gazedetector.transform(img_dir = img_dir, out_dir = out_dir)

# heatmap generation
# init
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



img_dir = '../train_1/x'
gt_dir = 'train_1/batch/y'
dataset = dataGen.DataGenerator(img_dir,gt_dir)
# hm = dataset.generate_hm('xx.txt', direction = 1, size = 64, pts=[[11,11],[11,11]])
# print(hm.shape)
# # print(hm[11:20,:12,0])
# # print(hm[11:20,:12,1])
# tmp1 = tf.convert_to_tensor(hm, np.float32)
# l_y,l_x = _argmax(tmp1[:,:,0])
# r_y,r_x = _argmax(tmp1[:,:,1])
# l_y,l_x,r_y,r_x = tf.Session().run([l_y,l_x,r_y,r_x])
# print(l_y,l_x,r_y,r_x)

# # hm = round(hm)
# # print(hm[11:20,:12,0])

# img = Image.fromarray((hm[:,:,0]*255).astype(np.uint8))
# img.save("train_1/test.png",ext='png')

# img = Image.fromarray((hm[:,:,1]*255).astype(np.uint8))
# img.save("train_1/test1.png",ext='png')

# im1 = cv2.imread('train_1/test.png')
# img1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
# print(im1.shape)
# tmp1 = tf.convert_to_tensor(img1, np.float32)
# l_y,l_x = _argmax(tmp1)
# # r_y,r_x = _argmax(tmp1[:,:,1])
# l_y,l_x = tf.Session().run([l_y,l_x])
# print(l_y,l_x)

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
    # image2,gtMap_l,gtMap_r = dataset._flip(img,gtMap_l,gtMap_r, rand=False)
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
    # image,gtMap_l,gtMap_r = dataset._rotate(img, gtMap_l, gtMap_r)
    # image,gtMap_l,gtMap_r = dataset._flip(image,gtMap_l,gtMap_r)
    # image,gtMap_l,gtMap_r = dataset._color_augment(image,gtMap_l,gtMap_r)
    # tmp_l = tf.convert_to_tensor(gtMap_l, np.float32)
    # tmp_r = tf.convert_to_tensor(gtMap_r, np.float32)

    # img_path = str(count+1204) + '.png'
    # filename1 = 'train_1/img/combination/' + img_path
    # img = Image.fromarray(image.astype(np.uint8))
    # img.save(filename1,ext='png')

    # filename2 = 'train_1/gt_img/left/combination/' + img_path
    # img = Image.fromarray(gtMap_l.astype(np.uint8))
    # img.save(filename2,ext='png')

    # filename3 = 'train_1/gt_img/right/combination/' + img_path
    # img = Image.fromarray(gtMap_r.astype(np.uint8))
    # img.save(filename3,ext='png')
    # # max coord
    # l_y,l_x = _argmax(tmp_l)
    # r_y,r_x = _argmax(tmp_r)
    # l_y,l_x,r_y,r_x = tf.Session().run([l_y,l_x,r_y,r_x])
    # print('Before saving',l_x,l_y,r_x,r_y)
    # pts1=[[l_x,l_y],[r_x,r_y]]
    # dataset.write_txt(img_path, img_path, direction,pts1,'test.txt')

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
# img = cv2.imread('train_1/batch/gt_3/1.png')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# # single img
# img_name = 'frame1.png'
# out_name = 'Dissertation/local_Dataset/video_frame/front/frame11.png'
# dataset.generate_hm(img_name = img_name, direction = 5, pts=[[91, 125], [156, 123]], out_name = out_name)
# # all imgs in dir
# out_dir = 'train_1/batch/gt'
# dataset.generate_hm(img_dir = img_dir, size = 64, pts=None, out_dir = out_dir):

# train data generate
# img_dir = 'train_1/x'
# gt_dir = 'train_1/groundtruth'
# dataset = dataGen.DataGenerator(img_dir, gt_dir,'dataset.txt')
# dataset.generate_set(rand=True)
# # train_set = DataGenerator.get_train()
# # valid_set = DataGenerator.get_valid()
# # batch_set = DataGenerator.get_batch(4,'train')

# generator = dataset._aux_generator(4, 4, normalize = True, sample_set = 'train')
# for i in range(2):
# 	x_batch, y_batch, w_batch = next(generator)

# demo
# load video

# split into frames + save

# load video frames

# pass to network get output img

# calculate max point x y coord

# decide gaze direction


