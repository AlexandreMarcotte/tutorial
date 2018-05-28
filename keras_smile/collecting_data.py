import numpy as np
import cv2
import os
#
cap = cv2.VideoCapture(0)
base_path = r'C:\Users\alexa\Documents\CODING\2018\tutorial\keras_smile/'
i = 0
side = 'right_hand'

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    i += 1
    if i == 50:
        side = 'left_hand'
    print(i, side)

    if ret and i <= 100:
        # gray = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)
        fliped = cv2.flip(frame, 1)
        # img = cv2.transform()

        # Display the resulting frame
        cv2.imshow('frame', fliped)
        # cv2.imwrite()
        save_path = os.path.join(base_path, side, f'img_{i}.jpg')
        print(save_path)
        cv2.imwrite(save_path, fliped)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()