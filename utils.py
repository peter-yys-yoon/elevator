
import opt
import sys

print(sys.path)
import cv2
import numpy as np


from elev.opt import WHITE

def draw_header(img, img_id, color, msg=''):

    imgW, imgH = img.shape[1], img.shape[0]
    HEADER_height = int(img.shape[1] * 0.05)
    mask = np.zeros((HEADER_height, imgW, 3), dtype=np.uint8)

    mask[:, :, :] = color
    msg = f'Frame: {str(img_id).rjust(4)}, {msg}'

    header = cv2.addWeighted(
        img[0:HEADER_height, 0:imgW, :], 0.4, mask, 0.6, 0)

    txt_size, baseLine1 = cv2.getTextSize(msg, cv2.FONT_HERSHEY_DUPLEX, 2, 2)
    p1_ = (10, 10+txt_size[1]+10)
    img[0:HEADER_height, 0:imgW, :] = header[:, :, :]
    cv2.putText(img, msg, p1_, cv2.FONT_HERSHEY_DUPLEX,
                2, WHITE, 2)  # point is left-bottom