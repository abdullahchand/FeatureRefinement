import cv2

def convert_to_1D(mask):
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    resized_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return resized_mask