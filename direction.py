import AUmaps
import glob
import dlib
import matplotlib.pyplot as plt
AUdetector = AUmaps.AUdetector('shape_predictor_68_face_landmarks.dat',enable_cuda=False)

# load video
path_video = 'test_video.avi'
# split into frames + save

# load video frames
path_imgs = 'video_imgs'
files = sorted(glob.glob(path_imgs + '/*.png'))
fig = plt.figure(figsize=plt.figaspect(.5))
for names in files:
    print(names)
    img = dlib.load_rgb_image(names)
    pred,map,img = AUdetector.detectAU(img)
    for j in range(0,5):
        resized_map = dlib.resize_image(map[j,:,:].cpu().data.numpy(),rows=256,cols=256)
        ax = fig.add_subplot(5,2,2*j+1)
        ax.imshow(img)
        ax.axis('off')
        ax = fig.add_subplot(5, 2, 2*j+2)
        ax.imshow(resized_map)
        ax.axis('off')
    plt.pause(.1)
    plt.draw()