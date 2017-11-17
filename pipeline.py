from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label
from skimage.feature import hog
from sklearn.externals import joblib
import argparse
import cv2
import glob
import math
import numpy as np
import os
import pickle

IMG_X = 1280
IMG_Y = 720

def extract_features(img, x0, y0, x1, y1):
    img = img[y0:y1, x0:x1]
    img = cv2.resize(img, (64, 64))
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv)
    y_hog = hog(y, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2))
    u_hog = hog(u, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2))
    v_hog = hog(v, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2))
    res = np.concatenate([y_hog, u_hog, v_hog])
    return res

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

def annotate_image(clf, img, prev_heats=[], threshold=3, return_heat=False):
    found_matches = []
    sizes = [64,96,128,192]
    for s in sizes:
        x0 = 0
        x1 = s
        while x1<=IMG_X:
            # We don't care about the upper half
            y0 = IMG_Y//2
            while y0+s<=IMG_Y:
                y1 = y0+s
                features = extract_features(img, x0, y0, x1, y1)
                res = clf.predict([features])
                if res:
                    found_matches.append([[x0,y0],[x1,y1]])
                y0 = y0+s//4
            x0 = x0+s//4
            x1 = x0+s
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    add_heat(heat,found_matches)
    summed_heat = heat.copy()
    for h in prev_heats:
        summed_heat += h
    thresholded_heat = apply_threshold(summed_heat,threshold)
    heatmap = np.clip(thresholded_heat, 0, 255)
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)
    if return_heat:
        return draw_img, heat
    else:
        return draw_img

def annotate_video(src, dst, use_prev_n_heats=6, threshold=4):
    clf = joblib.load('classifier.pkl')
    prev_heats = []
    def process_image(img):
        nonlocal prev_heats
        # The pipeline works on BGR images
        color_corrected = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        result, heat = annotate_image(
            clf, color_corrected,
            prev_heats=prev_heats,
            return_heat=True,
            threshold=(len(prev_heats)+1)*threshold)
        prev_heats.append(heat)
        prev_heats = prev_heats[-use_prev_n_heats:]
        return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    clip_in = VideoFileClip(src)
    clip_out = clip_in.fl_image(process_image)
    clip_out.write_videofile(dst, audio=False)

def output_test_images():
    '''
    Process the test images and write out the result.
    '''
    clf = joblib.load('classifier.pkl')
    test_images = glob.glob('test_images/*.jpg')
    for img_file in test_images:
        print("Processing: ", img_file)
        img = cv2.imread(img_file)
        img = annotate_image(clf, img)
        cv2.imwrite('output_images/'+os.path.basename(img_file), img)

def main():
    parser = argparse.ArgumentParser(description='Lane Lines Detection')
    subparsers = parser.add_subparsers(dest="cmd")
    subparsers.required = True
    output_test_parser = subparsers.add_parser('output-test-images')
    ann_vid_parser = subparsers.add_parser('annotate-video')
    ann_vid_parser.add_argument('--src', type=str, help="source file", required=True)
    ann_vid_parser.add_argument('--dst', type=str, help="target file", required=True)
    args = parser.parse_args()

    if args.cmd=="output-test-images":
        print("Generating annotated test images")
        output_test_images()
    elif args.cmd=='annotate-video':
        print("Annotating Video")
        annotate_video(args.src, args.dst)

if __name__ == '__main__':
    main()
