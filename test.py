import cv2
import numpy as np

ix, iy = -1, -1

def draw_circle(event, x, y, flags, param):
    global ix, iy
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img, (x,y), 100, (255,0,0), -1)
        ix, iy = x,y


image = np.zeros((512,512,3), np.uin8)
cv2.namedWindow("image")
cv2.SetMouseCallback("image", draw_circle)

while(1):
    cv2.imshow("image", ig)
    k = cv2.waitKey(20) & 0xFF
    if k == 27: break
    elif k == ord('a'): print ix,iy
cv2.destroyAllWindows()
