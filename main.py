#-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.
 #File Name : bg_remove.py
 #Creation Date : 31-07-2019
 #Created By : Rui An
#_._._._._._._._._._._._._._._._._._._._._.
import os
from io import BytesIO
import numpy as np
from PIL import Image

import tensorflow as tf 
import sys
import datetime
import cv2
import time

cap = cv2.VideoCapture(0) 

class DeepLabModel(object):
    """Class to load deeplab model and run inference."""

    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()

        graph_def = None
        graph_def = tf.GraphDef.FromString(open(tarball_path + "/frozen_inference_graph.pb", "rb").read()) 

        if graph_def is None:
          raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
          tf.import_graph_def(graph_def, name='')

        self.sess = tf.Session(graph=self.graph)

    def run(self, image):
        """Runs inference on a single image.

        Args:
          image: A PIL.Image object, raw input image.

        Returns:
          resized_image: RGB image resized from original input image.
          seg_map: Segmentation map of `resized_image`.
        """
        start = datetime.datetime.now()

        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]

        end = datetime.datetime.now()

        diff = end - start
        print("Time taken to evaluate segmentation is : " + str(diff))

        return resized_image, seg_map


def drawSegment(baseImg, matImg, outputFilePath):
    width, height = baseImg.size
    dummyImg = np.zeros([height, width, 4], dtype=np.uint8)
    for x in range(width):
              for y in range(height):
                  color = matImg[y,x]
                  (r,g,b) = baseImg.getpixel((x,y))
                  if color == 0:
                      dummyImg[y,x,3] = 0
                  else :
                      dummyImg[y,x] = [r,g,b,255]
    img = Image.fromarray(dummyImg)
    img.save(outputFilePath)


def run_visualization(filepath, MODEL, outputFilePath):
    """Inferences DeepLab model and visualizes result."""
    try:
            # print("Trying to open : " + sys.argv[1])
    	# f = open(sys.argv[1])
    	jpeg_str = open(filepath, "rb").read()
    	orignal_im = Image.open(BytesIO(jpeg_str))
    except IOError:
      print('Cannot retrieve image. Please check file: ' + filepath)
      return

    print('running deeplab on image %s...' % filepath)
    resized_im, seg_map = MODEL.run(orignal_im)

    # vis_segmentation(resized_im, seg_map)
    drawSegment(resized_im, seg_map, outputFilePath)


def extract_foreground(result_file_name, MODEL, progress):
    print("Start Taking Photo")
    time.sleep(1)
    ret, frame = cap.read()
    cv2.imwrite(result_file_name, frame)
    print("Finish Writing Image " + result_file_name)
    extract_input_path = result_file_name
    result_name = "./extracted_images/" + str(progress) + ".png" 
    run_visualization(extract_input_path, MODEL, result_name)


def take_picture_every(seconds, number_of_images, MODEL):
    path = "./raw_images/"
    progress = 0
    while (progress != number_of_images):
        time.sleep(seconds)
        file_name = path + str(progress) + ".jpg"
        extract_foreground(file_name, MODEL, progress)
        progress += 1
    print("Finish Taking Pictures")


def additive_blending(number_of_images):
    source_imgs = []
    for file in os.listdir("./extracted_images"):
        if file.endswith(".png"):
            file_path = "./extracted_images/" + file
            source_imgs.append(cv2.imread(file_path))
    result_img = cv2.addWeighted(source_imgs[0], 0.8, source_imgs[1], 0.2, 0)
    for i in range(2, number_of_images):
        result_img = cv2.addWeighted(result_img, 0.8, source_imgs[i], 0.2, 0)
    height, width, _  = result_img.shape
    # for i in range(height): 
        # for j in range(width):
            # if (np.count_nonzero(result_img[i][j]) == 0):
                # result_img[i][j] = [255, 255, 255]
    cv2.imwrite("./result/result.png", result_img)


def main(): 
    # inputFilePath = sys.argv[1]
    # outputFilePath = sys.argv[2]
    # if inputFilePath is None or outputFilePath is None:
      # print("Bad parameters. Please specify input file path and output file path")
      # exit()

    modelType = "mobile_net_model"
    MODEL = DeepLabModel(modelType)
    print('model loaded successfully : ' + modelType)
    take_picture_every(2, 10, MODEL)
    additive_blending(10)
    # run_visualization(inputFilePath, MODEL, outputFilePath)


if __name__=="__main__":
    main()

