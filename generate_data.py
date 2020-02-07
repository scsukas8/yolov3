# Imports
import cv2
import os
import sys
import numpy as np
import random

# Load base images
root = "" #"/Users/seancsukas/Documents/Workspace/bananagrams/"
file = "Background"
baseImgPath = root + file
baseImgs = []
NUM_BASE = 5
for i in range(NUM_BASE):
    baseImgs.append(cv2.imread(baseImgPath + f"{i+1}.jpg"))


base_y, base_x = (0,0)
size_y, size_x = (300,300)
transparent = np.zeros((size_y, size_x, 4), np.uint8)

imgs = {}
for letter in ["A", "B", "C", "D"]:
    img = cv2.imread(root + letter + ".jpg")
    img = cv2.resize(img, (size_y, size_x))
    
    tr = transparent.copy()
    tr[base_y: base_y + size_y, base_x : base_x + size_x, 0:3] = img
    tr[:,:, 3] = 255
    
    imgs[letter] = tr



def overlay_image(larger_image, smaller_image, x_pos, y_pos):
    small_y, small_x, _ = smaller_image.shape
    larger_image[y_pos : y_pos + small_y, x_pos : x_pos + small_x] = smaller_image
    return larger_image



def show_and_destroy(img, name="Noname"):
    cv2.imshow(name, img)
    ch = cv2.waitKey()
    cv2.destroyAllWindows()
    ch = cv2.waitKey(1)

"""
def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result
"""

# https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))


# Generate randomized image
def generate_image(base_img, tiles):
    
    # Create mask
    mask = np.zeros((base_img.shape[0], base_img.shape[1], 4), np.uint8)
    
    i = 0
    darknet_labels = []
    for k,v in tiles.items():
        
        # Perform random rotation
        degrees = random.randrange(0, 360)
        rot_img = rotate_bound(v, degrees)
        
        # Add item to mask in random location
        offset_y = random.randrange(rot_img.shape[0], mask.shape[0] - rot_img.shape[0])
        offset_x = random.randrange(rot_img.shape[1], mask.shape[1] - rot_img.shape[1])
        mask = overlay_image(mask, rot_img, offset_x, offset_y)
        
        darknet_label = [i,
            (offset_x + rot_img.shape[1] / 2) / base_img.shape[1],
            (offset_y + rot_img.shape[0] / 2) / base_img.shape[0],
            rot_img.shape[1] / base_img.shape[1],
            rot_img.shape[0] / base_img.shape[0]]
        darknet_labels.append(darknet_label)
        i += 1
    
    base_img = base_img.copy()
    mask_inv = cv2.bitwise_not(mask[:,:,3])
    base_img = cv2.bitwise_or(base_img, base_img, mask=mask_inv)
    base_img = cv2.bitwise_or(base_img, mask[:,:,0:3])
    
    return base_img[:,:,0:3], darknet_labels

output_dir = "/Users/seancsukas/Documents/Workspace/yolov3/"
# Check that data dir exists
directory = "" + "data/bananas/"
if not os.path.exists(directory):
    os.makedirs(directory)

im_dir = directory + "images/"
if not os.path.exists(im_dir):
    os.makedirs(im_dir)

lab_dir = directory + "labels/"
if not os.path.exists(lab_dir):
    os.makedirs(lab_dir)


images_list = []

for i in range(1000):
    if i % 100 == 0:
        print("Iteration Number: %s" % i)
    
    # Generate an image and save it
    baseImg = baseImgs[random.randrange(0, NUM_BASE)]
    generated_img, darknet_labels = generate_image(baseImg, imgs)
    cv2.imwrite(im_dir + f"image_{i:04d}.png", generated_img)
    with open(lab_dir + f"image_{i:04d}.txt", "w") as f:
        label_str = ""
        for label in darknet_labels:
            label_str += " ".join(str(x) for x in label) + "\n"
        f.write(label_str)
    
    images_list.append(im_dir + f"image_{i:04d}.png")

training_percent = 0.8
with open(directory + "banana_train.txt", "w") as f:
    f.write("\n".join(images_list[:int(len(images_list) * training_percent)]))

with open(directory + "banana_test.txt", "w") as f:
    f.write("\n".join(images_list[int(len(images_list) * training_percent):]))



# show_and_destroy(generated_img, "generated_img")