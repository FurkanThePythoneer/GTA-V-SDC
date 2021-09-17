import numpy as np
import pandas as pd
import cv2
import albumentations as albu
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.efficientnet import *
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from models import SDC_V0_Model

from keys.get_keys import *
from keys.direct_keys import *
from grab_screen import grab_screen

import random
import time
from statistics import mode, mean
#--------------------------
GAME_WIDTH = 800
GAME_HEIGHT = 620

WIDTH  = 480
HEIGHT = 270

models_dir = 'models'
model_path = models_dir + '/effnetv2-m/' + 'effnetv2_m_480_270_v0-1.pth'

w = [1,0,0,0,0,0,0,0,0]
s = [0,1,0,0,0,0,0,0,0]
a = [0,0,1,0,0,0,0,0,0]
d = [0,0,0,1,0,0,0,0,0]
wa = [0,0,0,0,1,0,0,0,0]
wd = [0,0,0,0,0,1,0,0,0]
sa = [0,0,0,0,0,0,1,0,0]
sd = [0,0,0,0,0,0,0,1,0]
nk = [0,0,0,0,0,0,0,0,1]

augment = albu.Compose([
    albu.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ToTensorV2()])



model = SDC_V0_Model(num_classes=9, pretrained=False)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
print('Model loaded succesfully!')

for i in reversed(range(5)):
    print(i+1)
    time.sleep(1)    


def process_screen(screen):
    #screen = grab_screen(region=(0, 40, GAME_WIDTH, GAME_HEIGHT))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
    screen = cv2.resize(screen, (WIDTH, HEIGHT)) # (270, 480, 3)
    screen = screen.reshape(WIDTH,HEIGHT,3) # (480, 270, 3)
    screen = augment(image=screen)['image'] # (3, 480, 270)
    screen = np.reshape(screen, (-1, 3, WIDTH, HEIGHT)) # torch.Size([1, 3, 480, 270])

    return screen


def main():


    while (True):
        last_time = time.time()
        screen = grab_screen(region=(0, 40, GAME_WIDTH, GAME_HEIGHT))
        screen = process_screen(screen)

        paused = False
        if not paused:

            # prediction:
            with torch.no_grad():
                prediction = model(screen)

            prediction = np.argmax(F.softmax(prediction, -1)) # now it's scalar: 0, 1, 2 or 3 etc..
            
            if prediction == 0:
                straight()
                choice_picked = 'straight'

            elif prediction == 1:
                reverse()
                choice_picked = 'reverse'
            
            elif prediction == 2:
                left()
                choice_picked = 'left'

            elif prediction == 3:
                right()
                choice_picked = 'right'

            elif prediction == 4:
                forward_left()
                choice_picked = 'forward+left'

            elif prediction == 5:
                forward_right()
                choice_picked = 'forward+right'

            elif prediction == 6:
                reverse_left()
                choice_picked = 'reverse+left'

            elif prediction == 7:
                reverse_right()
                choice_picked = 'reverse+right'
            
            elif prediction == 8:
                no_keys()
                choice_picked = 'nokeys'

            print('Loop took {} seconds'.format(round(time.time()-last_time,3)))
            
            #print('loop took {} seconds. Choice: {}'.format( round(time.time()-last_time, 3), choice_picked))

            keys = key_check()
            print(keys)
            if 'Q' in keys:
                print('Breaking')
                break


            #cv2.imshow('SDC-WHAT_AI_SEES', screen_)
            #if cv2.waitKey(25) & 0xFF == ord('q'):
            #    cv2.destroyAllWindows()
            #    break


if __name__ == '__main__':
    main()