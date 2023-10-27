import pygame, sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2

WINDOWSIZEX = 800
WINDOWSIZEY = 720
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BOUNDARYINC = 5
IMAGESAVE = False
MODEL = load_model("my_model.h5")
LABELS = {0:"ZERO", 1:"ONE", 2:"TWO", 3:"THREE", 4:"FOUR", 5:"FIVE", 6:"SIX", 7:"SEVEN", 8:"EIGHT", 9:"NINE"}   
pygame.init()

FONT = pygame.font.Font("freesansbold.ttf", 18)
DISPLAYSURFACE = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))

pygame.display.set_caption("Writing Board")

PREDICT = True
iswriting = False
image_count = 1
number_x = []
number_y = []

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        
        if event.type == MOUSEMOTION and iswriting:
            x, y = event.pos
            pygame.draw.circle(DISPLAYSURFACE, WHITE, (x, y), 4, 0)   
            number_x.append(x)
            number_y.append(y)
            
        if event.type == MOUSEBUTTONDOWN:
            iswriting= True
            
        if event.type == MOUSEBUTTONUP:
            iswriting = False
            number_x = sorted(number_x)         
            number_y = sorted(number_y)
            
            rectangle_min_x, rectangle_max_x = max(number_x[0] - BOUNDARYINC, 0), min(WINDOWSIZEX, number_x[-1] + BOUNDARYINC)
            rectangle_min_y, rectangle_max_y = max(number_y[0] - BOUNDARYINC, 0), min(number_y[-1] + BOUNDARYINC, WINDOWSIZEX) 
            
            number_x = []
            number_y = []
            
            image_array = np.array(pygame.PixelArray(DISPLAYSURFACE))[rectangle_min_x:rectangle_max_x, rectangle_min_y:rectangle_max_y].T.astype(np.float32)
            
            if IMAGESAVE:
                cv2.imwrite("image.png")
                image_count += 1
                
            if PREDICT:
                image = cv2.resize(image_array, (28, 28))
                image = np.pad(image, (10, 10), 'constant', constant_values = 0)
                image = cv2.resize(image, (28,28))/255
                label = str(LABELS[np.argmax(MODEL.predict(image.reshape(1, 28, 28, 1)))])
                text_surface = FONT.render(label, True, RED, WHITE)
                text_rectangle = text_surface.get_rect()
                text_rectangle.left, text_rectangle.bottom = rectangle_min_x, rectangle_max_y
                DISPLAYSURFACE.blit(text_surface, text_rectangle)
                
            if event.type == KEYDOWN:
                if event.unicode == "n":
                    DISPLAYSURFACE.fill(BLACK)
                    
        pygame.display.update()
                
            