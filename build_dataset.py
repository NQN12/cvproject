import argparse
import cv2 
import os

# Set up argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to output directory")
args = vars(ap.parse_args())

# Create output directory if it doesn't exist
if not os.path.exists(args["output"]):
    os.makedirs(args["output"])

# Initialize webcam
video = cv2.VideoCapture(0)
total = 0

# Main loop
while True:
    ret, frame = video.read()
    if not ret:
        break

    cv2.imshow("video", frame)
    key = cv2.waitKey(1) & 0xFF

    # Capture frame on 'k' key press
    if key == ord("k"):
        p = os.path.sep.join([args["output"], "{}.png".format(str(total).zfill(5))])
        cv2.imwrite(p, frame)
        total += 1

    # Exit on 'q' key press
    elif key == ord("q"):
        break

print("[INFO] {} face images stored".format(total))
video.release()
cv2.destroyAllWindows()
