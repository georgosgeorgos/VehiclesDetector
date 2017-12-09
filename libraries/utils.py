import cv2
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def save_p(data, name):
    with open(name, 'wb') as f:
        pickle.dump(data, f)
def load_p(name):
    with open(name, 'rb') as f:
        data = pickle.load(f)
    return data

def plot_image(img, title="Original", flag=False, size=(10,10), dpi=100, cmap=None):
    plt.figure(num=None, figsize=size, dpi=dpi, facecolor='w', edgecolor='k')
    if cmap != None:
        plt.imshow(img, cmap=cmap)
    else:
        plt.imshow(img)
    plt.title(title)
    
    if flag:
        mpimg.imsave("output_images/" + title, img, format="jpg")

def plot_2images(img, converted, title1='Original', title2='Converted', cmap="gray", flag1=False, flag2=False):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    f.subplots_adjust(hspace = .2, wspace=.05)
    ax1.imshow(img)
    ax1.set_title(title1, fontsize=30)
    ax2.imshow(converted, cmap=cmap)
    ax2.set_title(title2, fontsize=30)
    
    if flag1 == True:
        mpimg.imsave("output_images/" + title1, img, format="jpg")
    if flag2 == True:
        mpimg.imsave("output_images/" + title2, converted, format="jpg")

def plot_N_images(files, pipeline, flag=False):

    fig, axs = plt.subplots(len(files), 2, figsize=(20,20))
    fig.subplots_adjust(hspace = .2, wspace=.05)
    for f in range(len(files)):
        file = files[f]
        image = mpimg.imread("./test_images/" + file)
        img=image.copy()
        # draw with default scales=[1.0, 1.6]
        draw_img, boxes=pipeline(image, flag=True)
        
        axs[f, 0].imshow(image)
        axs[f,1].imshow(draw_img)
        if flag:
            mpimg.imsave("output_images/" + file, draw_img, format="jpg")

def grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return gray
    
def hsvscale(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    return hsv

def hlsscale(img):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    return hls

def luvscale(img):
    luv = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    return luv

def labscale(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    return lab

def yuvscale(img):
    yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    return yuv

def gaussian_blur(img, kernel_size):
    '''gaussian smoothing'''
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def color_space(img, flag="gray"):
    if flag=="gray":
        res=grayscale(img)
    elif flag=="hsv":
        res=hsvscale(img)
    elif flag=="hls":
        res=hlsscale(img)
    elif flag=="luv":
        res=luvscale(img)
    elif flag=="lab":
        res=labscale(img)
    elif flag=="yuv":
        res=yuvscale(img)
    return res