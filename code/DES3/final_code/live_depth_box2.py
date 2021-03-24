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
import numpy as np
import time

labels_to_names = {0: 'GuidingCar', 1: 'TissueBox', 2: 'Shoes', 3: 'WaterBottle'}

###############################################################################################################################

def detect_image(model, image, result_list):
    #Assumes image is in RGB format
    prediction_result = []
    draw_img = image.copy()
    ####
    image, scale = resize_image(image)
    #print("Image shape: ", image.shape)
    #print("Output: ",  model.predict_on_batch(np.expand_dims(image, axis=0)))
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    boxes /= scale
    box_count = 0
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
        #scores have been sorted. So breaking is fine
        if score < 0.4:
            break
        box_count += 1
        b = box.astype(int)
        color = label_color(label)
        draw_box(draw_img,b,color=color)
        caption = "{} {:.3f}".format(labels_to_names[label], score)
        draw_caption(draw_img, b, caption)
        #class_score = "{} {:.3f}".format(labels_to_names[label], score)
        pred_class = str(labels_to_names[label])
        mid_point = [(b[0] + b[2])//2, (b[1] + b[3])//2]
        prediction_result.append([b, pred_class, mid_point])
        if(pred_class=="GuidingCar"):
            result_list[0] = mid_point
        elif(pred_class=="TissueBox"):
            result_list[1] = mid_point
        elif(pred_class=="WaterBottle"):
            result_list[2] = mid_point
    ####
    cv2.imwrite("test2.jpg", draw_img)
    return (draw_img, prediction_result)

###############################################################################################################################

def check_max_height(depth_img):
    percentile_marker = np.percentile(depth_img,1)
    print("raw max value: ", np.max(depth_img))
    print("maximum height value: ", percentile_marker)
    #org_depth = depth_img.copy()
    depth_img[np.where(depth_img<=percentile_marker)] = 0
    depth_img[np.where(depth_img>percentile_marker)] = 255
    cv2.imwrite("test4.jpg", np.uint8(depth_img))
    y, x = np.where(depth_img == 0)
    #print("X: ", x, " Y: ", y)
    print([min(x), min(y), max(x), max(y)])
    return([(min(x) + (max(x)-min(x))//2), (min(y) + (max(y)-min(y))//2)])

###############################################################################################################################

def main() :
    user_height = 170
    object_height_location = ["Below weist", "At weist level", "Above weight level"]
    #display_output = True
    #load retinanet model and set output_video_path
    model_path = "inference_model/resnet50_csv_03.h5"
    #output_video_path = "output_video/output.mp4"

    #video_FourCC    = cv2.VideoWriter_fourcc(*"mp4v")
    #out = cv2.VideoWriter(output_path, video_FourCC, 4, ()) 

    ret_model = models.load_model(model_path, backbone_name='resnet50')

    #Settings for writing output video
    #
    ##################


    #Capturing output to video
    #video_FourCC    = cv2.VideoWriter_fourcc(*"mp4v")
    #video_fps       = 
    #video_size      = 
    #out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)

    frame = cv2.imread("/home/khangminsoo/Desktop/zed_test/captured_files/testRun_capture/testRun_capture_1.jpg")
    #vid = cv2.VideoCapture(2)
    #if not vid.isOpened():
    #    raise IOError("Couldn't open webcam or video")
    #video_FourCC    = cv2.VideoWriter_fourcc(*"mp4v")
    #video_fps       = vid.get(cv2.CAP_PROP_FPS)
    #video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)//2),
#                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    if True:
        model_output={"User_position":[],"Object_coord":[]}
        #return_value, frame = vid.read()
        #if not return_value:
        #    break
        if(frame is not None):
            # Retrieve the left image, depth image in the half-resolution
            ## Raw image from left_camera
            ## maybe sl.MEM.GPU not a good idea since GPU mem is taken up by obj detector
            #####depth_matrix = depth_measurement.get_data()
            print("original frame shape: ", frame.shape)
            frame_org = frame.copy()
            frame = frame[100:,150:,:]
            image_ocv = frame
            #zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
            #image_ocv = image_zed.get_data()
            #RGB image from left camera ready for object detection

            #print("Depth image dimension: ", depth_matrix.shape)
            #print(check_max_height(depth_matrix))
            #####model_output["User_position"] = check_max_height(depth_matrix)
            #cv2.imwrite("test.jpg", image_ocv)
          
            
            depth_matrix = np.load("/home/khangminsoo/Desktop/zed_test/captured_files/testRun_capture/testRun_capture_1.npy")  
            print("depth matrix shape: ", depth_matrix.shape)
            output_img = image_ocv
            result_output = [[],[],[],[]]
            output_img, pred_result = detect_image(ret_model, image_ocv, result_output)
            
            if(len(pred_result) > 0):
                #There are objects detected
                depth_theshold_max = 500
                depth_threshold_min = 0
                valid_values = depth_matrix[np.where((depth_matrix<depth_theshold_max)&(depth_matrix>=depth_threshold_min))]
                valid_max = np.max(valid_values)
                valid_min = np.min(valid_values)
                depth_matrix[np.where(depth_matrix>=500)] = valid_max
                depth_matrix[np.where(depth_matrix<0)] = valid_min
                depth_matrix = cv2.resize(depth_matrix, (frame_org.shape[1], frame_org.shape[0]), interpolation = cv2.INTER_CUBIC)
                depth_matrix = depth_matrix[100:,150:]
                print("adjusted depth matrix: ", depth_matrix.shape)
                #######
                #if the depth matrix is exactly twice in dimension then can try this instead:
                """
                for res in pred_result:
                    depth_val = depth_measurement.get_value(res[2][0]*2, res[2][1]*2)
                    res.append(depth_val)
                """
                #######
                #else
                for res in result_output:
                    
                    if(len(res) != 0):
                         depth_val = depth_matrix[res[1]][res[0]]
                         depth_val = 300 - depth_val
                         res.append(depth_val)
                         #depth_val = depth_matrix[res[2][1]][res[2][0]]
                    else:
                         continue
                    print("Depth1: ", depth_val)
                    print("Res: ", res)
                    print("img shape: ", image_ocv.shape)
                    print("Depth dim: ", depth_matrix.shape)
                    #depth_matrix[res[2][1]-50:res[2][1]+50,res[2][0]-50:res[2][0]+50] = 0
                    #print("Depth2: ", depth_matrix[res[2][1]][res[2][0]])
                    #res.append(depth_val)

            print("\nOutput: ")
            #print(result_output) #detection result for Edmund
            max_y, max_x = check_max_height(depth_matrix.copy())
            print("max height detected at: ", [max_x, max_y])  #User position detected
            depth_matrix2 = depth_matrix.copy()
            depth_matrix3 = depth_matrix.copy()
            depth_matrix3 = depth_matrix3[50:-100,300:-50]

            print("depth matrix 3: ", np.percentile(depth_matrix3,3))
            
            obstacle_threshold = np.percentile(depth_matrix3,3)
            depth_matrix3[np.where(depth_matrix3<obstacle_threshold)] = 0
            depth_matrix3[np.where(depth_matrix3>=obstacle_threshold)] = 255

            y, x = np.where(depth_matrix3 == 0)
            x = 300 + min(x) + (max(x)-min(x))//2
            y = 50 + min(y) + (max(y)-min(y))//2
            temp_frame = frame.copy()
            #temp_frame = temp_frame[100:-6,300:,:]
            temp_frame[y-30:y+30, x-30:x+30,:] = 0
            cv2.imwrite("testingDepthImg2.jpg", temp_frame)
            #[(min(x) + (max(x)-min(x))//2), (min(y) + (max(y)-min(y))//2)])

            #This is for obstacle coord. Note that the depth for obstacle has been omitted from calc and set to 300
            result_output[3] = [x,y,300]
            print("Final output: ", result_output)

            #Final masking test, output to newframeimage.jpg
            with open("Final_obj_det_result.txt", 'w') as f:
                for testid in result_output:
                    for data in testid:
                        f.writelines(str(data)+"\n")
                    frame[testid[1]-30:testid[1]+30, testid[0]-30:testid[0]+30,:] = 0
            cv2.imwrite("newFrameImage.jpg", frame)

            cv2.imwrite("obstacle_depth.jpg", depth_matrix3)
            cv2.imwrite("testing_temp.jpg", frame[100:-60,300:,:])
            depth_matrix[max_x-30:max_x+30,max_y-30:max_y+30] = 0
            #print("Output shape: ", output_img.shape)
            depth_matrix2 *= (255.0/depth_matrix2.max())
            depth_matrix *= (255.0/depth_matrix.max())
            depth_matrix2 = np.uint8(depth_matrix2)
            depth_matrix = np.uint8(depth_matrix)
            cv2.imwrite("test3.jpg",depth_matrix2)
            cv2.imwrite("test.jpg", depth_matrix)
#            cv2.imwrite("Output.jpg", image_ocv)
            #output_img = np.asarray(output_img)
            #output_img = np.uint8(output_img)
            #cv2.imwrite("Output.jpg", image_ocv)


    cv2.destroyAllWindows()
#    zed.close()

    print("\nFINISH")


###############################################################################################################################

def detect_video(model_path, video_path, output_path=""):

    model = models.load_model(model_path, backbone_name='resnet50')

    vid = cv2.VideoCapture(video_path)
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")
    video_FourCC    = cv2.VideoWriter_fourcc(*"mp4v")
    video_fps       = vid.get(cv2.CAP_PROP_FPS)
    video_size      = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))//2,
                        int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    isOutput = True if output_path != "" else False
    if isOutput:
        print("!!! TYPE:", type(output_path), type(video_FourCC), type(video_fps), type(video_size))
        out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
    accum_time = 0
    curr_fps = 0
    fps = "FPS: ??"
    prev_time = timer()
    while True:
        return_value, frame = vid.read()
        if not return_value:
            break
        #print(frame.shape)
        frame = frame[:,:frame.shape[1]//2,:]
        #print(frame.shape)
        #exit()
        image = Image.fromarray(frame)
        image = np.asarray(image.convert('RGB'))
        #image = read_image_bgr(frame)
        #image = preprocess_image(image[:,:,::-1])

        draw = image.copy()
        #draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
        image, scale = resize_image(image)
        boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        boxes /= scale
        box_count = 0
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted so we can break
            if score < 0.4:
                break
            box_count += 1
            color = label_color(label)

            b = box.astype(int)
            draw_box(draw, b, color=color)

            caption = "{} {:.3f}".format(labels_to_names[label], score)
            draw_caption(draw, b, caption)

        print("Found {} boxes".format(box_count))
        result = np.asarray(draw)
        result = np.uint8(result)
        curr_time = timer()
        exec_time = curr_time - prev_time
        prev_time = curr_time
        accum_time = accum_time + exec_time
        curr_fps = curr_fps + 1
        if accum_time > 1:
            accum_time = accum_time - 1
            fps = "FPS: " + str(curr_fps)
            curr_fps = 0
        cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.50, color=(255, 0, 0), thickness=2)
        #cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        #cv2.imshow("result", result)
        #print("Dimension of result: ", result.shape)
        if isOutput:
            out.write(result)
            print("result shape: ", result.shape)
            print("Video size: ", video_size)
            print("isoutput: ", isOutput)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

if __name__=="__main__":
    main()
