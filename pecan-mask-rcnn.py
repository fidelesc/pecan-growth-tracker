import os
import re
import json
import pyautogui
import detectron2
import cv2
import random
import numpy as np
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.catalog import register_coco_instances
from detectron2.evaluation import COCOEvaluator
import glob
from detectron2.data import build_detection_test_loader
from fnmatch import fnmatch
from scipy.spatial import distance
import PySimpleGUI as sg


def mousePoints(event, x, y, flags, params):
    """
    Mouse click callback function for selecting two points on an image.

    Parameters:
        event (int): The type of mouse event.
        x (int): The x-coordinate of the mouse click.
        y (int): The y-coordinate of the mouse click.
        flags (int): The state of the mouse buttons and keyboard modifiers.
        params: Optional parameters passed to the function.

    Returns:
        None.
    """
    global counter, point_matrix, rsz_img
    
    # Check if the left mouse button was double-clicked
    if event == cv2.EVENT_LBUTTONDBLCLK:
        # Store the point in the point matrix
        point_matrix[counter] = x, y
        # Draw a red circle on the image at the clicked point
        cv2.circle(rsz_img, (x, y), radius=3, color=(0, 0, 255), thickness=-1)
        # Increment the counter
        counter += 1


# Set the theme of the GUI
sg.theme("DarkTeal2")

# Define the layout of the GUI
layout = [
    [sg.T("")],
    [sg.Text("Choose a folder: "), sg.Input(), sg.FolderBrowse(key="-PATH-")],
    [sg.Button("Submit")],
    [sg.Button("Close")]
]

# Create the GUI window
window = sg.Window('My Pecan Browser', layout, size=(600,150))

while True:
    # Read events from the window
    event, values = window.read()
    
    # If the window is closed or the user clicks the "Exit" button, break out of the loop
    if event == sg.WIN_CLOSED or event == "Exit":
        break
    
    # If the user clicks the "Submit" button, get the selected folder path
    elif event == "Submit":
        path = values["-PATH-"] + "/"
        if path == "/":
            # If the user did not select a folder, print a message and continue the loop
            sg.Print("Please input folder with the images!")
            continue
        else:
            # If a folder was selected, break out of the loop
            break
    
    # If the user clicks the "Close" button, close the window and continue the loop
    elif event == "Close":
        window.close()

# Close the GUI window
window.close()

# Print a message to indicate the start of the script
print("started")

# Initialize list to store file paths
files = []

# Loop over all files in the directory and subdirectories
for path, subdirs, filelist in os.walk(path):
    for name in filelist:
        # Check if the file is an image file
        if fnmatch(name, "*.jpg") or fnmatch(name, "*.JPG") or fnmatch(name, "*.jpeg") or fnmatch(name, "*.JPEG") or fnmatch(name, "*.png") or fnmatch(name, "*.PNG"):
            # Get the file path and add it to the list
            filename = os.path.join(path, name)
            files.append(filename)

# Get the paths to the models
Big_cell_model = "models/big-shell/model_final.pth"
big_embryo1_model = "models/big-embryo/model_final.pth"
big_pecan_model = "models/big-shuck/model_final.pth"
small_cell = "models/small-shell/model_final.pth"
small_pecan1_model = "models/small-shuck/model_final.pth"

# Create an array with the model paths
models = [Big_cell_model, big_embryo1_model, big_pecan_model, small_cell, small_pecan1_model]


# Create a list to store predictor objects
predictors = []

# Loop through each model in the models list
for model in models:

    # Create a configuration object
    cfg = get_cfg()

    # Load the configuration file for the model
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml"))

    # Set the number of classes
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    # Load the model weights
    cfg.MODEL.WEIGHTS = model

    # Set the threshold for score during testing
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8

    # Set the threshold for non-maximum suppression during testing
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.05

    # Create a predictor object with the configuration
    predictor = DefaultPredictor(cfg)

    # Add the predictor to the list of predictors
    predictors.append(predictor)

# Get the total number of files to process
total_files = len(files)

print("Starting inference on {} images".format(total_files))
ccount = 0
for imageName in files:
    ccount += 1
    checkfile = imageName[0:-4]+".csv"
    
    if os.path.isfile(checkfile):
        #print(checkfile)
        continue

    print("Progress: {}/{}".format(ccount, total_files))
    print(imageName)
    
    counter = 0
    resize_value = 600
    
    point_matrix = np.zeros((2,2),np.int)
    
    img = cv2.imread(imageName)


    max_value = np.max(img.shape)
    dim = resize_value/max_value

    new_w = int(img.shape[1]*dim)
    new_h = int(img.shape[0]*dim)
    
    rsz_img = cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_AREA)
    
    while counter < 2:
        # Showing original image
        cv2.imshow("Preview", rsz_img)
    
        # Mouse click event on original image
        cv2.setMouseCallback("Preview", mousePoints)
        
        # Refreshing window all time
        cv2.waitKey(1)         

    cv2.imshow("Preview", rsz_img)
    cv2.waitKey(1)   
    
    p0 = np.array([[point_matrix[0][0], point_matrix[0][1]]])
    p1 = np.array([[point_matrix[1][0], point_matrix[1][1]]])
    
    dist =distance.cdist(p0, p1, 'euclidean')
    print("Distance: ", dist)

    pixel_count = dist/dim

    print(pixel_count)

    pixel_size = pixel_count/10

    area_mm2 = pixel_size[0][0]**2

    print("px per mm2: ", area_mm2)

    
    for i in range(len(predictors)):
        # apply the predictor to the image
        outputs = predictors[i](img)
    
        contours = []
    
        # find the contour of each segment detected 
        for pred_mask in outputs['instances'].pred_masks:
            mask = pred_mask.cpu().detach().numpy().astype('uint8')
            contour, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            contours.append(contour[0]) 
    
        inferences = []

        img_copy = img.copy()
        
        # draw the contours and print their area in pixel
        for contour in contours:

            area = cv2.contourArea(contour)
            polygon_mm2 = round(area/area_mm2, 2)
    
            pixel = 30
            Red = np.random.randint(255)
            Blue = np.random.randint(255)
            Green = np.random.randint(255)
            index = np.where(contour[:,0,1] == np.min(contour[:,0,1]))[0][0]
            x_text = contour[index,0,0] - 3*pixel
            y_text = contour[index,0,1] - pixel
            
            text = str(polygon_mm2) + " mm2"
            cv2.drawContours(img_copy, [contour], -1, (Blue,Green,Red), 8)
            cv2.putText(img_copy, text, (x_text,y_text), cv2.FONT_HERSHEY_SIMPLEX, 2,(Blue,Green,Red), 8, cv2.LINE_AA)
            
            inference_area = 100*(area/area_mm2)
            inferences.append(inference_area)
    
        # show the detection related to each model       
        cv2.namedWindow("Output", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Output",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        cv2.imshow("Output",img_copy)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        check = [i]
        np.savetxt(checkfile, check, delimiter=',')
        
        save_name = imageName[0:-4]+"_"+str(i)+".csv"
        save_name_img = imageName[0:-4]+"_"+str(i)+".jpg"
        np.savetxt(save_name, inferences, delimiter=',')
        cv2.imwrite(save_name_img, img_copy)

