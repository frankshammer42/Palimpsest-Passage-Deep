#-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.
 #File Name : bg_remove.py
 #Creation Date : 31-07-2019
 #Created By : Rui An
#_._._._._._._._._._._._._._._._._._._._._.
import cv2
import time
import numpy as np
import os

background_set_time = 5
cap = cv2.VideoCapture(0) 
bg_storage_path = "../background/"
pixel_diff_threshold = 150 
fgbg = cv2.createBackgroundSubtractorMOG2()

def check_legal_choice(choice):
    if (choice == 'N' or choice == 'P' or choice == 'Q'): 
        return True 
    else: 
        return False 


def set_up_environment(bg_name):
    print("Please Make Sure No One is In Front Of the Camera")
    print("Will Take the Background Photo in 5 Seconds")
    counter = 0
    while (counter<5):
        print(counter)
        counter += 1
        time.sleep(1)
    print("Take New Background Photo")
    ret, frame = cap.read()
    while(True):
        cv2.imshow('Press y to save the file',frame) 
        if cv2.waitKey(1) & 0xFF == ord('y'): #save on pressing 'y'
            cv2.imwrite('../background/' + bg_name, frame)
            cv2.destroyAllWindows()
            break


def extract_foreground(current_background, result_file_name):
    bg_path = bg_storage_path + current_background
    bg_img = cv2.imread(bg_path) 
    # bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)
    bg_img_denoised = cv2.fastNlMeansDenoisingColored(bg_img, None, 50, 50, 7, 21)
    bg_height, bg_width, channel = bg_img.shape 
    channel = 4
    # # Create White Images
    white_image = np.zeros(shape=[bg_height, bg_width, channel], dtype=np.uint8)
    print("Start Extract Foreground")
    # while(True):
    ret, frame = cap.read()
    # bw_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # frame_diff = np.abs(frame - bg_img) 
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_denoised = cv2.fastNlMeansDenoisingColored(frame, None, 50, 50, 7, 21)
    frame_diff = np.abs(frame_denoised - bg_img_denoised)
    frame_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)
    height, width= frame_diff.shape
    # # fgmask = fgbg.apply(frame)
    # # print(fgmask.shape)
    for i in range(height):
        for j in range(width):
            pixel_diff = frame_diff[i][j] 
            # pixel_diff = fgmask[i][j]
            if (pixel_diff >pixel_diff_threshold):
                white_image[i][j] = [frame[i][j][0], frame[i][j][1], frame[i][j][2], 255]
            # to_extract = False 
            # for k in range(3):
                # if (pixel_diff[k] > pixel_diff_threshold):
                    # to_cover = False
                    # break
            # if (to_cover):
                # frame[i][j] = [255, 255, 255] 
    cv2.imwrite(result_file_name, white_image)
    print("finished_extracting")
    # cv2.imshow('fuckthis', white_image)
    # white_image = np.zeros(shape=[bg_height, bg_width, channel], dtype=np.uint8)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
        # break


#DEVELOPMENT MODULES
def take_picture_every(seconds, number_of_images, current_background):
    path = "../raw_images/"
    progress = 0
    while (progress != number_of_images):
        time.sleep(seconds)
        file_name = path + str(progress) + ".png"
        extract_foreground(current_background, file_name)
        progress += 1
    print("Finish Taking Pictures")


def additive_blending():
    source_imgs = []
    for file in os.listdir("../raw_images"):
        if file.endswith(".png"):
            file_path = "../raw_images/" + file
            source_imgs.append(cv2.imread(file_path))
    result_img = cv2.addWeighted(source_imgs[0], 0.6, source_imgs[1], 0.4, 0)
    for i in range(2, 5):
        result_img = cv2.addWeighted(result_img, 0.6, source_imgs[i], 0.4, 0)
    height, width, _  = result_img.shape
    for i in range(height): 
        for j in range(width):
            if (np.count_nonzero(result_img[i][j]) == 0):
                result_img[i][j] = [255, 255, 255]
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2BGRA)
    cv2.imwrite("../result/result.png", result_img)


def main():
    print("-----------------Running Palimpsest Passage------------------")
    take_picture_every(2, 5, bg_file_name)
        

if __name__=="__main__":
    main()
