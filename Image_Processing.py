import cv2
import numpy as np
import os
# import math
# from collections import Counter
# import argparse

# DEFINING THE TEST SET AND FILE TO SAVE IMAGES
path_to_dataset = "l2-ip-images/test/corrupted"
path_to_results = "l2-ip-images/test/results"

# Add the images to an array
img_array = []
for image in os.listdir(path_to_dataset):
    img = cv2.imread(os.path.join(path_to_results, image))
    img_array.append(img)

# Find the width and height of the images
height, width, channel = img_array[0].shape
size = (width, height)
# print(size)

def noise_removal(frame):
    # Median filtering with 3 neighbours to remove salt and pepper noise
    # Performs the best on salt and pepper noise and well on gaussian noise, but smooths a lot
    neighbourhood = 3
    median = cv2.medianBlur(frame, neighbourhood)

    # Mean smoothing instead here still leaves gaussian noise
    # mean = cv2.blur(frame, (neighbourhood, neighbourhood), borderType=cv2.BORDER_DEFAULT)

    # Also bilateral filtering maintains detail compared to gaussian and mean
    # Edges preserved better than gaussian filtering, but still leaves noise
    # sigma_r = 0.1
    # sigma_s = 2
    # bilateral = cv2.bilateralFilter(frame, -1, sigma_r, sigma_s, borderType=cv2.BORDER_REPLICATE)

    # Non-local means is usually the best, but still leaves noise here
    # neighbourhood = 7
    # window = 21
    # filter = 16
    # nlm_img = cv2.fastNlMeansDenoising(frame, h = filter, templateWindowSize = neighbourhood, searchWindowSize=window)

    return median


# CONTRAST ADJUSTMENT
def contrast_adjustment(frame):
    # We can use laplacian of gaussian filtering for edge sharpening to improve the image detail
    # Subtract the laplacian filtered image from the original image
    # Could use bilateral or non-zero means here instead of gaussian
    # Or could use a band pass filter and apply this similarly to laplacian to enhance the edges
    # gaussian = cv2.GaussianBlur(frame, (21, 21), 0)
    # laplacian_applied = cv2.Laplacian(gaussian, cv2.CV_64F)
    # laplacian = np.uint8(laplacian_applied)
    # laplacian_filter = frame - laplacian

    # Apply an edge sharpening kernel
    # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    # edge_sharpening = cv2.filter2D(frame, -1, kernel)

    # Since we have a lot of bright range values, we don't want to perform a logarithmic transform
    # We want to decrease the dynamic range of dark regions so we could use exponential transform
    # Or apply contrast enhancement with power law transform (gamma correction) since the image is overexposed in areas
    # Act like exponential transform to change the overall brightness
    # Make everything a little darker - slight gamma correction as the image we currently have is slightly overexposed
    # gamma = 0.8
    # power_law = ((np.power(frame/255, gamma)) * 255).astype('uint8')

    # Construct a histogram to adjust the distribution of values (global image transform)
    # We can perform a contrast stretch, histogram equalisation or localised histogram equalisation
    # hist = cv2.equalizeHist(frame)

    # Contrast stretch and slightly limit the upper and lower values to closer match the image
    contrast = np.empty(frame.shape, dtype=np.uint8)
    cv2.normalize(frame, contrast, alpha=20, beta=250, norm_type=cv2.NORM_MINMAX)

    # Apply CLAHE histogram stretch
    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(6, 6))
    clahe_applied = clahe.apply(contrast)

    # We could have used a fourier transform to turn off all the low frequencies to better define the edges
    # Similarly to laplacian i.e. a high pass filter and butterworth to avoid ringing, but no need as no ringing
    return clahe_applied


# FINDING CORNERS
def corners(frame, list):
    # Apply a gaussian filter
    blur = cv2.GaussianBlur(frame, (5, 5), 0)
    # Colour the edges white and everything else black
    edges = cv2.Canny(blur, 75, 200)

    # Use corner Harris to find the corners of each image
    dst = cv2.cornerHarris(edges, 2, 3, 0.04)
    dst_norm = np.empty(dst.shape, dtype=np.float32)
    cv2.normalize(dst, dst_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Add each corner to the same list
    for i in range(dst_norm.shape[0]):
        for j in range(dst_norm.shape[1]):
            if int(dst_norm[i, j]) > 254:
                co = []
                co.append(i)
                co.append(j)
                list.append(co)

    return frame, list


# DEWARPING
def dewarping(frame):
    # Using the image corners we found earlier, create a pespective transform
    new = cv2.getPerspectiveTransform(np.array([[17, 386], [21, 20], [943, 7], [964, 375]], np.float32), np.array([[0, height], [0, 0], [width, 0], [width, height]], np.float32))
    # Transform each image
    warped = cv2.warpPerspective(frame, new, (width, height))
    return warped


# Create an empty list and add in each corner we find with corner Harris
list = []
for image_name in os.listdir(path_to_dataset):
    img = cv2.imread(os.path.join(path_to_dataset, image_name), 0)
    result, list2 = corners(img, list)
    list = list2

# A function to remove the most repeated coordinates from our list
def remove_values(corner_list, val):
    return [value for value in corner_list if value != val]


# Sort the list and print the 12 most repeated coordinates (our four corners) including our 4 corners
list2 = sorted(list, key=lambda k: [k[1], k[0]])
src1 = []
x = 12
while x != 0:
    max_list = max(list2, key=list.count)
    src1.append(max_list)
    list2 = remove_values(list2, max_list)
    x = x-1
    # print(src1)

# PERFORM THE IMAGE PROCESSING
for image_name in os.listdir(path_to_dataset):
    # Read in a new image every loop
    img = cv2.imread(os.path.join(path_to_dataset, image_name), 0)

    # Pass the image through the functions defined above
    # The order does not really matter
    result = contrast_adjustment(img)
    result = noise_removal(result)
    result = dewarping(result)

    # Save the images to the "l2-ip-images/test/results" folder. PNG is a lossless format
    # The images are saved with the same file name
    cv2.imwrite(path_to_results + "/" + image_name, result)

# CONVERT THE GENERATED SAVED FRAMES TO A VIDEO
img_array2 = []
for image2 in os.listdir(path_to_results):
    img = cv2.imread(os.path.join(path_to_results, image2))
    img_array2.append(img)

# Define the frame rate, size, file name and format for the video
fourcc = cv2.VideoWriter_fourcc(*"DIVX")
output = cv2.VideoWriter("test-output.avi", fourcc, 3, size)

# Create the video
for i in range(len(img_array2)):
    output.write(img_array2[i])
output.release()

# Now pass the created video to yolo.py to check object detection performance
