# Britney Clark
# CSC580: Applying Machine Learning and Neural Networks
# Critical Thinking 1, option 1
# Dr. Joseph Issa
# May 22, 2022

from PIL import Image, ImageDraw
import face_recognition
import numpy
import cv2

# Load the jpg file into a numpy array
image = face_recognition.load_image_file("PXL_20210320_223402323.jpg")
pilImage = Image.fromarray(image)
pilImage.show(image)

# Find all the faces in the image # Use the following Python pseudocode as guidance for your solution.
face_locations = face_recognition.face_locations(image)
numberOfFaces = len(face_locations)
print("Found {} face(s) in this picture.".format(numberOfFaces))

# Load the image into a Python Image Library object so that you can draw on top of it and display it
print(face_locations)

for (x1, y1, x2, y2) in face_locations:
    # Print the location of each face in this image. Each face is a list of co-ordinates in (top, right, bottom, left) order.
    print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(x1, y1, x2, y2))
    # Draw a box around the face     drawHandle = PIL.ImageDraw.Draw(pilImage)     drawHandle.rectangle([left, top, right, bottom], outline="red")
    draw = ImageDraw.Draw(pilImage, mode=None)
    draw.rectangle([y2, x1, y1, x2], outline=(255, 0, 0), width=4)

# Display the image on screen
pilImage.show(draw)

