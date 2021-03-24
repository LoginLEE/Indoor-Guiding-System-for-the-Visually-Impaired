#########
#ZED imports
import sys
import numpy as np
import pyzed.sl as sl
import cv2
#########
#Retinanet_video imports
import keras
from PIL import Image
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.gpu import setup_gpu
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import cv2
import os
import time
#########
#Set GPU visible to script
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#Set GPU ID and run set_up
gpu = 0
setup_gpu(gpu)
#########
#Set label name for retinanet detection
labels_to_names = {0: 'obstacle', 1: 'vehicle'}
#########

def detect_image(model, image):
    #Assumes image is in RGB format
    prediction_result = []
    ####
    image, scale = resize_image(image)
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    boxes /= scale
    box_count = 0
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        #scores have been sorted. So breaking is fine
        if score < 0.4:
            break
        box_count += 1
        b = box.astype(int)
        #class_score = "{} {:.3f}".format(labels_to_names[label], score)
        pred_class = str(labels_to_names[label])
        mid_point = [(b[0] + b[2])//2, (b[1] + b[3])//2]
        prediction_result.append([b, pred_class, mid_point])
    ####
    return prediction_result

help_string = "[s] Save side by side image [d] Save Depth, [n] Change Depth format, [p] Save Point Cloud, [m] Change Point Cloud format, [q] Quit"
prefix_point_cloud = "Cloud_"
prefix_depth = "Depth_"
path = "./"

count_save = 0
mode_point_cloud = 0
mode_depth = 0
point_cloud_format_ext = ".ply"
depth_format_ext = ".png"

def point_cloud_format_name(): 
    global mode_point_cloud
    if mode_point_cloud > 3:
        mode_point_cloud = 0
    switcher = {
        0: ".xyz",
        1: ".pcd",
        2: ".ply",
        3: ".vtk",
    }
    return switcher.get(mode_point_cloud, "nothing") 
  
def depth_format_name(): 
    global mode_depth
    if mode_depth > 2:
        mode_depth = 0
    switcher = {
        0: ".png",
        1: ".pfm",
        2: ".pgm",
    }
    return switcher.get(mode_depth, "nothing") 

def save_point_cloud(zed, filename) :
    print("Saving Point Cloud...")
    tmp = sl.Mat()
    zed.retrieve_measure(tmp, sl.MEASURE.DEPTH)
    saved = (tmp.write(filename + depth_format_ext) == sl.ERROR_CODE.SUCCESS)
    if saved :
        print("Done")
    else :
        print("Failed... Please check that you have permissions to write on disk")

def save_depth(zed, filename) :
    print("Saving Depth Map...")
    tmp = sl.Mat()
    zed.retrieve_measure(tmp, sl.MEASURE.XYZRGBA)
    saved = (tmp.write(filename + point_cloud_format_ext) == sl.ERROR_CODE.SUCCESS)
    if saved :
        print("Done")
    else :
        print("Failed... Please check that you have permissions to write on disk")

def save_sbs_image(zed, filename) :

    image_sl_left = sl.Mat()
    zed.retrieve_image(image_sl_left, sl.VIEW.LEFT)
    image_cv_left = image_sl_left.get_data()

    image_sl_right = sl.Mat()
    zed.retrieve_image(image_sl_right, sl.VIEW.RIGHT)
    image_cv_right = image_sl_right.get_data()

    sbs_image = np.concatenate((image_cv_left, image_cv_right), axis=1)

    cv2.imwrite(filename, sbs_image)
    
def process_key_event(zed, key) :
    global mode_depth
    global mode_point_cloud
    global count_save
    global depth_format_ext
    global point_cloud_format_ext

    if key == 100 or key == 68:
        save_depth(zed, path + prefix_depth + str(count_save))
        count_save += 1
    elif key == 110 or key == 78:
        mode_depth += 1
        depth_format_ext = depth_format_name()
        print("Depth format: ", depth_format_ext)
    elif key == 112 or key == 80:
        save_point_cloud(zed, path + prefix_point_cloud + str(count_save))
        count_save += 1
    elif key == 109 or key == 77:
        mode_point_cloud += 1
        point_cloud_format_ext = point_cloud_format_name()
        print("Point Cloud format: ", point_cloud_format_ext)
    elif key == 104 or key == 72:
        print(help_string)
    elif key == 115:
        save_sbs_image(zed, "ZED_image" + str(count_save) + ".png")
        count_save += 1
    else:
        a = 0

def print_help() :
    print(" Press 's' to save Side by side images")
    print(" Press 'p' to save Point Cloud")
    print(" Press 'd' to save Depth image")
    print(" Press 'm' to switch Point Cloud format")
    print(" Press 'n' to switch Depth format")

def main() :
    #load retinanet model and set output_video_path
    model_path = "inference_models/ret_B_1_epoch70.h5"
    output_video_path = "/home/khangminsoo/Desktop/vehicle_taskV1/testing_data/ret_B_1_E70.mp4"
    ret_model = models.load_model(model_path, backbone_name='resnet50')

    #Settings for writing output video
    #
    ##################

    # Create a ZED camera object
    zed = sl.Camera()

    # Set configuration parameters
    input_type = sl.InputType()
    if len(sys.argv) >= 2 :
        input_type.set_from_svo_file(sys.argv[1])
    init = sl.InitParameters(input_t=input_type)
    init.camera_resolution = sl.RESOLUTION.HD1080
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE
    init.coordinate_units = sl.UNIT.CENTIMETER

    #init.depth_minimum_distance = 0.1

    # Open the camera
    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS :
        print(repr(err))
        zed.close()
        exit(1)

    # Display help in console
    print_help()

    # Set runtime parameters after opening the camera
    runtime = sl.RuntimeParameters()
    #runtime.sensing_mode = sl.SENSING_MODE.STANDARD
    ##Changed mode from Standard to Fill
    runtime.sensing_mode = sl.SENSING_MODE.FILL
    # Prepare new image size to retrieve half-resolution images
    image_size = zed.get_camera_information().camera_resolution
    image_size.width = image_size.width /2
    image_size.height = image_size.height /2

    # Declare your sl.Mat matrices
    image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    depth_image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    point_cloud = sl.Mat()

    #Capturing output to video
    #video_FourCC    = cv2.VideoWriter_fourcc(*"mp4v")
    #video_fps       = 
    #video_size      = 
    #out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)

    key = ' '
    depth_measurement = sl.Mat()
    while key != 113 :
        err = zed.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS :
            # Retrieve the left image, depth image in the half-resolution
            ## Raw image from left_camera
            ## maybe sl.MEM.GPU not a good idea since GPU mem is taken up by obj detector
            zed.retrieve_measure(depth_measurement, sl.MEASURE.DEPTH)
            depth_matrix = depth_measurement.get_data()

            zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
            image_ocv = image_zed.get_data()
            #RGB image from left camera ready for object detection
            pred_result = detect_image(ret_model, image_ocv)
            
            if(len(pred_result) > 0):
                #There are objects detected
                depth_theshold_max = 500
                depth_threshold_min = 0
                valid_values = depth_matrix[np.where((depth_matrix<depth_theshold_max)&(depth_matrix>=depth_threshold_min))]
                valid_max = np.max(valid_values)
                valid_min = np.min(valid_values)
                depth_matrix[np.where(depth_matrix>=500)] = valid_max
                depth_matrix[np.where(depth_matrix<0)] = valid_min
                cv2.resize(depth_matrix, image_ocv.shape, interpolation = cv2.INTER_CUBIC)
                #######
                #if the depth matrix is exactly twice in dimension then can try this instead:
                """
                for res in pred_result:
                    depth_val = depth_measurement.get_value(res[2][0]*2, res[2][1]*2)
                    res.append(depth_val)
                """
                #######
                #else
                for res in pred_result:
                    depth_val = depth_matrix[res[2][0]][res[2][1]]
                    res.append(depth_val)

            print("\nOutput: ")
            print(pred_result)
            continue

            new_matrix = np.uint8(new_matrix)

            cv2.imwrite("org_L.jpg", image_ocv)
            cv2.imwrite("test.jpg", new_matrix)
            print("value: ", depth_measurement.get_value(540,961))
            exit()
            

            cv2.imshow("Image", image_ocv)
            cv2.imshow("Depth", depth_image_ocv)
            depth_val = depth_measurement.get_value(10,10)
            #print("Depth of camera at (10,10): ", depth_val)
            #print("Max: ", np.max(depth_image_ocv))
            #print("Min: ", np.min(depth_image_ocv))
            key = cv2.waitKey(10)

            process_key_event(zed, key)

    cv2.destroyAllWindows()
    zed.close()

    print("\nFINISH")

if __name__ == "__main__":
    main()
