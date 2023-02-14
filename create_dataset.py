import h5py
import cv2
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import os
import traceback
import glob


SPLITTER = '--*--'
SPLITTER2 = '**-**'
PATH_SPLITTER = '*-*-*-'
# DATASET_PATH = 'datasets/datasets_06'
DATASET_PATH = 'submit_file/dataset'

FONT_INDEXES = {  
    'Titillium_Web':0,
     'Alex_Brush':1,
     'Open_Sans':2,
     'Sansation':3,
    'Ubuntu_Mono':4
}


def _cut_bb(image,bb):
    mask1 = np.zeros(image.shape)
    mask2 = np.ones(image.shape)
    poly = bb.T.astype(np.int32)
    cv2.fillPoly(mask1,[poly],1)
    cv2.fillPoly(mask2,[poly],0)
    poly_copied = np.multiply(mask1,image)
    top_left = int(min(bb[0,:])),int(min(bb[1,:]))
    bottom_right = int(max(bb[0,:])),int(max(bb[1,:]))
    background_color = np.sum(poly_copied) / ((bottom_right[0]-top_left[0])*(bottom_right[1]-top_left[1]))
    background = mask2*background_color
    poly_copied += background
    return poly_copied

def cut_bb(image,bb):
    img_shape = image.shape
    char = np.zeros(img_shape)
    for i in range(3):
        char[:,:,i] = _cut_bb(image[:,:,i],bb)
    top_left = int(max(0,min(bb[0,:]))),int(max(0,min(bb[1,:])))
    bottom_right = int(min(img_shape[1],max(bb[0,:]))),int(min(img_shape[0],max(bb[1,:])))
    char =  char[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]]
    return char

def save_char_img(im_name,db,phase):
    img = db['data'][im_name][:]
    # gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_attrs = db['data'][im_name].attrs
    fonts = img_attrs['font']
    chars_BB = img_attrs['charBB']
    fails = []
    for i , font in enumerate(fonts):
        font = str(font.decode("utf-8")).replace(' ','_')
        bb = chars_BB[:,:,i]
        char = cut_bb(img, bb)
        dir_path = os.path.join(DATASET_PATH,phase,font)
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
        img_num = len(os.listdir(dir_path))
        img_name = f'{img_num}.jpg'
        file_name = os.path.join(dir_path,img_name)
        try:
            cv2.imwrite(file_name, char)
        except Exception as e:
            fails.append(im_name)
        print(fails)
word_count = 0
char_count = 0 
def save_word_img(im_name,db,phase):
    global word_count
    global char_count
    img = db['data'][im_name][:]
    # gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_attrs = db['data'][im_name].attrs
    if phase != 'predict':
        fonts = img_attrs['font']
        font_idx = 0
    words_BB = img_attrs['wordBB']
    words = img_attrs['txt']
    assert words_BB.shape[2] == len(words) ,  f'{words_BB.shape[2]} == {len(words)} , {words}'
    fails = []
    for i, word in enumerate(words):
        if phase != 'predict':
            font = fonts[i]
            # font = fonts[font_idx]
            # font = str(font.decode("utf-8")).replace(' ','_')
            font = font.replace(' ','_')
        bb = words_BB[:,:,i]
        word_img = cut_bb(img,bb)
        if phase != 'predict':
            dir_path = os.path.join(DATASET_PATH,phase,font)
        else:
            dir_path = os.path.join(DATASET_PATH,'test')

        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
        if phase != 'predict':
            img_num = len(os.listdir(dir_path))
            img_name = f'{img_num}.jpg'
            font_idx += len(word)
        else:
            word = str(word.decode("utf-8"))
            char_count += len(word)
            word = word.replace('/',PATH_SPLITTER)
            img_name = f'_{SPLITTER}{im_name}{SPLITTER2}{word}{SPLITTER}{i}.jpg'
        file_name = os.path.join(dir_path,img_name)
        while os.path.exists(file_name):
            print("exist")
            img_name = '_' + im_name
            file_name = os.path.join(dir_path,img_name)
        try:
            word_count += 1
            cv2.imwrite(file_name, word_img)
            imgs_c = len(glob.glob(dir_path + "/*.jpg"))
            assert imgs_c == word_count , (imgs_c ,word_count)
        except Exception:
            if phase == 'predict':
                traceback.print_exc()
                print(file_name)
                print(word_img.shape)
                raise Exception()
            fails.append(im_name)
def main():
    db  = h5py.File('submit_file/SynthText_test.h5','r')
    im_names = list(db['data'].keys())
    for im_name in tqdm(im_names,desc = 'predict'):
        save_word_img(im_name,db,'predict')
    print(word_count)
    print(char_count)
    # db  = h5py.File('Project/SynthText/results/AlexBrush.h5','r')
    # im_names = list(db['data'].keys())
    # for im_name in tqdm(im_names,desc = 'predict'):
    #     save_word_img(im_name,db,'predict')
    # for im_name in tqdm(im_names[601:900],desc = 'val'):
    #     save_word_img(im_name,db,'val')
    # for im_name in tqdm(im_names[901:],desc = 'test'):
    #     save_word_img(im_name,db,'test')
    # files_count = {}
    # for phase in ['train', 'val', 'test']:
    #     dir_path = os.path.join(DATASET_PATH,phase)
    #     count = sum([len(files) for r, d, files in os.walk(dir_path)])
    #     files_count[phase] = count
    # total_count = sum(files_count.values())
    # for key, value in files_count.items():
    #     print(f'number of images in {key} are : {value} = {value/total_count}%')
        

if __name__ == '__main__':
    main()
