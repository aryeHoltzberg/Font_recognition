import numpy as np
from PIL import Image
import glob
import cv2
from tqdm import tqdm


BASE_PATH = 'datasets/simple/dataset-0/train'

def get_stat(img):
    mean = [0,0,0]
    std = [0,0,0]
    for chanel in range(3):
        mean[chanel] = np.mean(img[:,:,chanel])/255
        std[chanel] = np.std(img[:,:,chanel])/255
    return mean, std
def get_stat_all_imgs(all_imgs):
    count = 0
    total_mean = [0,0,0]
    total_std = [0,0,0]
    for img_path in tqdm(all_imgs):
        try:
            img = cv2.imread(img_path)
            mean ,std = get_stat(img)
            total_mean = np.add(mean,total_mean)
            total_std = np.add(std,total_std)
            count += 1
        except:
            continue
    final_mean = [0,0,0]
    final_std = [0,0,0]
    for i in range(3):
        final_mean[i] = total_mean[i]/count
        final_std[i] = total_std[i]/count
    return final_mean , final_std

def main():
    dir_list = glob.glob(BASE_PATH + "/*")
    all_imgs = []
    for dir in dir_list:
        img_list = glob.glob(dir + "/*")
        all_imgs.extend(img_list)
    final_mean,final_std = get_stat_all_imgs(all_imgs)
    np.save('final_mean',final_mean)
    np.save('final_std',final_std)
    print(final_mean,final_std)


if __name__ == '__main__':
    main()

