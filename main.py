import cv2
import numpy as np

capture = cv2.VideoCapture(0)

square_size = 500
padding = 50
threshold_ratio = 0.6 # しきい値の割合

while True:
    ret, frame = capture.read()

    # get frame size
    height, width, _ = frame.shape

    # yes or no rectangle
    left_top = (padding, padding)
    left_bottom = (padding + square_size, padding + square_size)
    right_top = (width - padding - square_size, padding)
    right_bottom = (width - padding, padding + square_size)

    # draw initial text
    cv2.putText(frame, 'Yes', (padding+80, padding+300), cv2.FONT_HERSHEY_SIMPLEX, 7, (0, 255, 0), 10)
    cv2.putText(frame, 'No', (width-padding-400, padding+300), cv2.FONT_HERSHEY_SIMPLEX, 7, (0, 255, 0), 10)

    # draw rectangle
    cv2.rectangle(frame, left_top, left_bottom, (0, 255, 0), 2)
    cv2.rectangle(frame, right_top, right_bottom, (0, 255, 0), 2)

    # 肌色の範囲設定(HSV)
    lower_skin = np.array([0, 20, 70])
    upper_skin = np.array([20, 255, 255])

    # 色空間変換
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # マスク作成
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Calculate the threshold for each rectangle
    total_pixels = square_size * square_size
    threshold = total_pixels * threshold_ratio

    # Check skin color pixels in the rectangle
    if cv2.countNonZero(mask[padding:padding + square_size, padding:padding+square_size]) > threshold:
        cv2.rectangle(frame, left_top, left_bottom, (0, 0, 255), -1)
        cv2.putText(frame, 'Yes', (padding+80, padding+300), cv2.FONT_HERSHEY_SIMPLEX, 7, (0, 0, 0), 10)
        print('Yes')
    elif cv2.countNonZero(mask[padding:padding + square_size, width-padding-square_size:width-padding]) > threshold:
        cv2.rectangle(frame, right_top, right_bottom, (0, 255, 0), -1)
        cv2.putText(frame, 'No', (width-padding-400, padding+300), cv2.FONT_HERSHEY_SIMPLEX, 7, (0, 0, 0), 10)
        print('No')
    else:
        print('None')
    
    left_color = (0, 255, 0)
    right_color = (0, 255, 0)

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()