from PIL import Image, ImageFont, ImageDraw 
import argparse
import numpy as np
import glob
from tqdm import tqdm
import requests
from numpy import random
import scipy.stats as ss
import numpy as np
import os

FAIL_COUNT = 0

class Words():
    def __init__(self):
        self.words = self.__get_list_of_words()
        self.font_sizes = range(10,110,10)
        x = np.arange(-5, 5)
        xU, xL = x + 0.5, x - 0.5 
        prob = ss.norm.cdf(xU, scale = 3) - ss.norm.cdf(xL, scale = 3)
        prob = prob / prob.sum() # normalize the probabilities so their sum is 1
        self.prob = prob
        

    def __get_list_of_words(self):
        response = requests.get(
            'https://www.mit.edu/~ecprice/wordlist.10000',
            timeout=10
        )

        string_of_words = response.content.decode('utf-8')

        list_of_words = string_of_words.splitlines()

        return list_of_words

    def get_word(self):
        return random.choice(self.words)

    def get_font(self):
        return random.choice(self.font_sizes,p=self.prob)

            

BASE_PATH = '/media/arye/Elements/Arye/SynthText/bg_data/bg_img/'
FONTS =  {  
    'Titillium_Web':'/home/arye/Desktop/Arye/CV-101/final_project/fonts/titillium-web/TitilliumWeb-Regular.ttf',
     'Alex_Brush':'/home/arye/Desktop/Arye/CV-101/final_project/fonts/alex-brush/AlexBrush-Regular.ttf',
     'Open_Sans':'/home/arye/Desktop/Arye/CV-101/final_project/fonts/open-sans/OpenSans-Regular.ttf',
     'Sansation':'/home/arye/Desktop/Arye/CV-101/final_project/fonts/sansation/Sansation-Regular.ttf',
    'Ubuntu_Mono':'/home/arye/Desktop/Arye/CV-101/final_project/fonts/ubuntumono/UbuntuMono-Regular.ttf'
}

DATABASE_PATH = 'datasets/simple'



def get_locs(image:Image,grid_dim = 3):
    w,h = image.size
    locs = []
    for i in range(grid_dim):
        for j in range(grid_dim):
            loc = (int(w/grid_dim)*i,int(h/grid_dim)*j)
            locs.append(loc)
    return locs

def get_color():
    return tuple(np.random.choice(range(255), size=3))

def update_bb(bb,loc):
    return(
        bb[0]+loc[0],
        bb[1]+loc[1],
        bb[2]+loc[0],
        bb[3]+loc[1],
    )


def save_img(img,dir_path):
    global FAIL_COUNT
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    img_num = len(os.listdir(dir_path))
    img_name = f'{img_num}.jpg'
    file_name = os.path.join(dir_path,img_name)
    try:
        img.save(file_name)
    except Exception as e:
        FAIL_COUNT += 1

def create_folder():
    if not os.path.isdir(DATABASE_PATH):
        os.makedirs(DATABASE_PATH)
    ds_num = len(os.listdir(DATABASE_PATH))
    dir_name = f'dataset-{ds_num}'
    dir_path = os.path.join(DATABASE_PATH,dir_name)
    os.makedirs(dir_path)
    return dir_path

def update_database(file,dir_path,words):
    global FONTS
    global FAIL_COUNT
    font_size = words.get_font()
    try:
        image = Image.open(file)
        image.convert("RGB")
        color = get_color()
        text = words.get_word()
        locs = get_locs(image,grid_dim=2)
        for font_name, font in FONTS.items():
                img_copy = image.copy()
                image_editable = ImageDraw.Draw(img_copy)
                font = ImageFont.truetype(font, font_size)
                bb = font.getbbox(text)
                for i,loc in enumerate(locs):
                    image_editable.text(loc, text, color, font=font)
                    bb_new = update_bb(bb,loc)
                    img_crop = img_copy.crop(bb_new)
                    path = os.path.join(dir_path,font_name)
                    save_img(img_crop,path)
    except:
        FAIL_COUNT += 1
        return


def main(args):
    dir_path = create_folder()
    words = Words()
    file_list = glob.glob(BASE_PATH + "/*")
    for file in tqdm(file_list[:args.num_of_imgs]):
        update_database(file,dir_path,words)
    print(f'finish with {FAIL_COUNT} fails')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Genereate Synthetic Scene-Text Images')
    parser.add_argument('--num-of-images',action='store',dest='num_of_imgs',default=1,type=int)
    # parser.add_argument('--num-of-images',action='store',dest='num_of_imgs',default=1,type=int)
    main(parser.parse_args())
