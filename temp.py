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
from train_launcher import process_config

# params = process_config('config.cfg')

# gazedetector = Gazedetect.Gazedetector('shape_predictor_68_face_landmarks.dat',params,enable_cuda=False)
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

# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


# dets = detector(image)
# shape = predictor(image,dets[0])
# coords = np.zeros((68, 2), dtype='float')
# for i in range(0,68):
#     coords[i] = (float(shape.part(i).x),float(shape.part(i).y))

# img_dir = 'train_x/'
# gt_dir = ['train_x/left','train_x/right']
# dataset = dataGen.DataGenerator(img_dir,gt_dir)

# for i in range(1,302):
#     filename = '../train_1/128_x/'+str(i)+'.png'
#     image = cv2.imread(filename)
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#     im_l,im_r = dataset.crop_im(image,points=None,size=(64,64))
# # im = dataset._transform(image,coords, filename = None)
#     image = cv2.cvtColor(im_l, cv2.COLOR_BGR2RGB)
#     filename = 'train_x/left/'+str(i)+'.png'
#     cv2.imwrite(filename, image)
#     filename = 'train_x/right/'+str(i)+'.png'
#     image = cv2.cvtColor(im_r, cv2.COLOR_BGR2RGB)
#     cv2.imwrite(filename, image)

    

# heatmap generation
# init
# def open_img(name, flag, color = 'RGB'):
#     """ Open an image 
#     Args:
#         name    : Name of the sample
#         color   : Color Mode (RGB/BGR/GRAY)
#     """
#     if flag == 0: # img
#         filename = os.path.join('../train_1/x', name)
#     elif flag == 1: # gtMap left
#         filename = os.path.join('train_1/left/', name)
#     elif flag == 2: # gtMap right
#         filename = os.path.join('train_1/right/', name)
#     elif flag == 3: # gtMap right
#         filename = os.path.join('train_1/gt_img/left/rotate/', name)
#     elif flag == 4: # gtMap right
#         filename = os.path.join('train_1/gt_img/right/rotate/', name)
        
#     img = cv2.imread(filename)
#     if color == 'RGB':
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         return img
#     elif color == 'BGR':
#         return img
#     elif color == 'GRAY':
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         return img
#     else:
#         print('Color mode supported: RGB/BGR. If you need another mode do it yourself :p')

# def _argmax(tensor):
#     resh = tf.reshape(tensor, [-1]) # flatten into 1-D 64*64
#     argmax = tf.argmax(resh,0)
#     # NOTE: return row, col i.e. y coord, x coord
#     return (argmax // tensor.get_shape().as_list()[0], argmax % tensor.get_shape().as_list()[0])



# img_dir = 'train_1/same/img'
# gt_dir = ['train_1/same/gt_img/left','train_1/same/gt_img/right']
# dataset = dataGen.DataGenerator(img_dir,gt_dir)
# dataset.generate_set(rand = True)

# _, y, _, _ = next(dataset._aux_generator(batch_size = 4, stacks = 4, sample_set = 'train'))
# for i in range(4):
#     t = y[:,3,:,:,:]
#     t = t[i]
#     tmp1 = tf.convert_to_tensor(t, np.float32)
#     l_y,l_x = _argmax(tmp1[:,:,0])
#     r_y,r_x = _argmax(tmp1[:,:,1])
#     l_y,l_x,r_y,r_x = tf.Session().run([l_y,l_x,r_y,r_x])
#     print(l_y,l_x,r_y,r_x)

# coord = [[[0,0],[0,0]],[[25,26],[25,26]],[[35, 24],[35, 24]],[[42,29],[42,29]],[[25,30],[25,30]],[[32,31],[32,31]],[[43, 33],[43, 33]],[[25, 38],[25, 38]],[[33, 42],[33, 42]],[[39, 38],[39, 38]]]
# hm = dataset.generate_hm('xx.txt', direction = 1, size = 64, pts=coord[1])
# print(hm.shape)
# print(hm[23:28,23:28,0])
# img = Image.fromarray((hm[:,:,0]*255).astype(np.uint8))
# img.save("train_x/test.png",ext='png')
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
# input_file = open(self.train_data_file, 'r')
#         for line in input_file:
#             if line in ['\n', '\r\n']:
#                 print('READING end of file')
#                 break
#             line = line.strip()
#             line = line.split(' ')
#             name = line[0]
#             gtMap = line[1]
#             direction = int(line[2]) #Note: str type
#             eyes = list(map(int,line[3:]))
#             w = [1] * len(self.direction)
#             if eyes != [-1] * len(eyes):
#                 eyes = np.reshape(eyes, (-1,2))
#                 # w = [1] * eyes.shape[0]
#                 for i in range(eyes.shape[0]):
#                     if np.array_equal(eyes[i], [-1,-1]):
#                         w[0] = 0 # w[1] TODO w len 1/2?
#                 self.data_dict[name] = {'gtMap' : gtMap, 'direction' : direction, 'eyes' : eyes, 'weights' : w}
#                 self.train_table.append(name)
#         input_file.close()

# cropping to 128 size
# count = 0
# input_file = open('../dataset.txt', 'r')
# for line in input_file:
#     count+=1
#     if line in ['\n', '\r\n']:
#         print('READING end of file')
#         break
#     line = line.strip()
#     line = line.split(' ')
#     name = line[0]
#     img = open_img(name, 0)
# # files = sorted(glob.glob('../train_1/x' + '/*.png'))
# # count = 0
# # for name in files:
#     # count+=1
#     # img = cv2.imread(name)
#     # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     im1 = dataset.crop_im(img,points=None,size=(128,128))
#     img = Image.fromarray(im1)
#     filename = 'train_1/128_x/'+str(count)+'.png'
#     img.save(filename,ext='png')
# input_file.close()
# coord = [[[0,0],[0,0]],[[13, 30],[45, 30]],[[17, 30],[48, 30]],[[19,30],[49, 30]],[[14, 32],[45, 33]],[[16, 32],[49, 32]],[[20, 31],[52, 32]],[[13, 34],[45, 34]],[[16, 35],[47, 37]],[[18, 34],[51, 34]]]

coord = [[[0,0],[0,0]],[[25,26],[25,26]],[[35, 24],[35, 24]],[[42,29],[42,29]],[[25,30],[25,30]],[[32,31],[32,31]],[[43, 33],[43, 33]],[[25, 38],[25, 38]],[[33, 42],[33, 42]],[[39, 38],[39, 38]]]
            

for i in range(31, 302):
    with open('64.txt', 'a') as file:
        # if i < 31:
        #     direction = 0
        #     pts = coord[0]
        if i < 61:
            direction = 1
            pts = coord[1]
        elif i < 91:
            direction = 2
            pts = coord[2]
        elif i < 121:
            direction = 3
            pts = coord[3]
        elif i < 151:
            direction = 4
            pts = coord[4]
        elif i < 182:
            direction = 5
            pts = coord[5]
        elif i < 212:
            direction = 6
            pts = coord[6]
        elif i < 242:
            direction = 7
            pts = coord[7]
        elif i < 272:
            direction = 8
            pts = coord[8]
        elif i < 302:
            direction = 9
            pts = coord[9]
        line1='left/'+str(i)+'.png '+'left/'+str(i)+'.png '+str(direction)+' '+str(pts[0][0])+' '+str(pts[0][1])+' '+str(pts[1][0])+' '+str(pts[1][1])+'\n'
        line2='right/'+str(i)+'.png '+'right/'+str(i)+'.png '+str(direction)+' '+str(pts[0][0])+' '+str(pts[0][1])+' '+str(pts[1][0])+' '+str(pts[1][1])+'\n'
        
            # print('writing: '+line)
        file.write(line1)
        file.write(line2)

