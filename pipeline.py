import numpy as np
import cv2
import glob
import pickle
import os
import argparse
from sklearn.externals import joblib
from skimage.feature import hog
import math

IMG_X = 1280
IMG_Y = 720

def extract_features(img, scaler, x0, y0, x1, y1):
    img = img[y0:y1, x0:x1]
    img = cv2.resize(img, (64, 64))
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    h, l, s = cv2.split(img)
    l_hog = hog(l, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1))
    s_hog = hog(l, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1))
    return scaler.transform([np.concatenate([l_hog, s_hog])])[0]

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thick)
    # Return the image copy with boxes drawn
    return imcopy

def annotate_image(clf, scaler, img):
    found_matches = []
    sizes = [64,96,128,192]
    for s in sizes:
        for i in range(math.floor(IMG_X/s-1)*2):
            for j in range(math.floor(IMG_Y/s-1)*2):
                x0 = math.floor(s/2*i)
                x1 = x0+s
                y0 = math.floor(s/2*j)
                y1 = y0+s
                features = extract_features(img, scaler, x0, y0, x1, y1)
                res = clf.predict([features])
                if res:
                    found_matches.append([x0,y0,x1,y1])
    img = draw_boxes(img, found_matches)
    return img

def output_test_images():
    '''
    Process the test images and write out the result.
    '''
    clf, scaler = joblib.load('classifier.pkl')
    test_images = glob.glob('test_images/*.jpg')
    for img_file in test_images:
        print("Processing: ", img_file)
        img = cv2.imread(img_file)
        img = annotate_image(clf, scaler, img)
        cv2.imwrite('output_images/'+os.path.basename(img_file), img)

def main():
    parser = argparse.ArgumentParser(description='Lane Lines Detection')
    subparsers = parser.add_subparsers(dest="cmd")
    subparsers.required = True
    output_test_parser = subparsers.add_parser('output-test-images')
    args = parser.parse_args()

    if args.cmd=="output-test-images":
        print("Generating annotated test images")
        output_test_images()

if __name__ == '__main__':
    main()