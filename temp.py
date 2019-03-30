import Gazedetect
import dataGen
import glob
import dlib
import matplotlib.pyplot as plt
import numpy as np
import cv2

gazedetector = Gazedetect.Gazedetector('Dissertation/Action-Units-Heatmaps/shape_predictor_68_face_landmarks.dat',enable_cuda=False)
path_imgs = 'Dissertation/local_Dataset/video_frame/front/frame1.png'
img = dlib.load_rgb_image(path_imgs)
gtMap = cv2.resize(img, dsize=(64, 64), interpolation=cv2.INTER_CUBIC)
filename = 'Dissertation/local_Dataset/video_frame/front/frame12.png'
plt.imsave(filename, gtMap)
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
# video_path = 'Dissertation/local_Dataset/Sessions/4/UserFrontal_C_Prudence.avi'
# Gazedetector.process_video(video_path,True,'Dissertation/local_Dataset/video_frame/4/UserFrontal_C_Prudence')

# procrustes transform imgs in a dir
# img_dir = 'Dissertation/local_Dataset/video_frame/4/Frontal_C_Prudence'
# out_dir = 'Dissertation/local_Dataset/video_frame/4/train'
# Gazedetector.transform(img_dir = img_dir, out_dir = out_dir)

# heatmap generation
# img_dir = 'Dissertation/local_Dataset/video_frame/front'
# gt_dir = 'Dissertation/local_Dataset/video_frame/front/groundtruth'
# dataset = dataGen.DataGenerator(img_dir,gt_dir)
# # single img
# img_name = 'frame1.png'
# out_name = 'Dissertation/local_Dataset/video_frame/front/frame11.png'
# dataset.generate_hm(img_name = img_name, direction = 5, pts=[[91, 125], [156, 123]], out_name = out_name)
# # all imgs in dir
# out_dir = 'Dissertation/local_Dataset/video_frame/front/groundtruth'
# DataGenerator.generate_hm(img_dir = img_dir, out_dir = out_dir)

# train data generate
# img_dir = 'Dissertation/local_Dataset/video_frame/front'
# gt_dir = 'Dissertation/local_Dataset/video_frame/front/groundtruth'
# dataset = dataGen.DataGenerator(img_dir, gt_dir,'dataset.txt')
# dataset.generate_set(rand=True)
# train_set = DataGenerator.get_train()
# valid_set = DataGenerator.get_valid()
# batch_set = DataGenerator.get_batch(4,'train')

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


