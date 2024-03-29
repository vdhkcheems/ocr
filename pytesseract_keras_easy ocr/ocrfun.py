import matplotlib.pyplot as plt
import cv2
from PIL import Image

def display(im_path):
    dpi = 80
    im_data = plt.imread(im_path)

    height, width  = im_data.shape[:2]
    figsize = width / float(dpi), height / float(dpi)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    ax.axis('off')
    ax.imshow(im_data, cmap='gray')

    plt.show()

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def noise_removal(image):
    import numpy as np
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return (image)

def thin_font(image):
    import numpy as np
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return (image)

def thick_font(image):
    import numpy as np
    image = cv2.bitwise_not(image)
    kernel = np.ones((2,2),np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return (image)

def bbox_keras(results, image):
    img = cv2.imread(image)
    for text, bbox in results[0]:
        # Extract coordinates
        top_left = (int(bbox[0][0]), int(bbox[0][1]))
        bottom_right = (int(bbox[2][0]), int(bbox[2][1]))
        
        # Draw rectangle on image
        cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
        
        # Put text near the bounding box
        cv2.putText(img, text, (int(bbox[0][0]), int(bbox[0][1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return (img)

def bbox_tess(result, image):
    img = cv2.imread(image)
    lines = result.split("\n")
    lines.pop()
    for line in lines:
        content = line.split(" ")
        x1 = int(content[1])
        y1 = int(content[2])
        x2 = int(content[3])
        y2 = int(content[4])

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return (img)

def bbox_easy(result, image):
    img = cv2.imread(image)
    for x in result:
        top_left = tuple(map(int,x[0][0]))
        bottom_right = tuple(map(int, x[0][2]))

        cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)

        cv2.putText(img, x[1], (int(x[0][0][0]), int(x[0][0][1]) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return (img)