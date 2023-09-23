import cv2
import numpy as np
import json
import time
from PIL import Image, ImageDraw, ImageFont
import textwrap

quiz_data = json.load(open('quiz.json', 'r'))

capture = cv2.VideoCapture(0)

square_size = 500
padding = 100
threshold_ratio = 0.8 # しきい値の割合

font_path = "./NotoSansJP-VariableFont_wght.ttf"

correct_count = 0

# タイマー変数の初期化
timers_start = None

for q in quiz_data['questions']:
    print(q['question'])
    correct_answer = q['answer']

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
        
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_img)
        text = f"Q: {q['question']}"
        wrapper_text = textwrap.TextWrapper(width=19)
        word_list = wrapper_text.wrap(text=text)
        font = ImageFont.truetype(font_path, 100)
        y_text = height - 350
        for line in word_list:
            draw.text((50, y_text), line, font=font, fill=(255, 255, 255, 0))
            y_text += 100
        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

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
        selected_answer = None
        if cv2.countNonZero(mask[padding:padding + square_size, padding:padding+square_size]) > threshold:
            cv2.rectangle(frame, left_top, left_bottom, (0, 0, 255), -1)
            cv2.putText(frame, 'Yes', (padding+80, padding+300), cv2.FONT_HERSHEY_SIMPLEX, 7, (0, 0, 0), 10)
            print('Yes')
            selected_answer = 'Yes'
        elif cv2.countNonZero(mask[padding:padding + square_size, width-padding-square_size:width-padding]) > threshold:
            cv2.rectangle(frame, right_top, right_bottom, (0, 255, 0), -1)
            cv2.putText(frame, 'No', (width-padding-400, padding+300), cv2.FONT_HERSHEY_SIMPLEX, 7, (0, 0, 0), 10)
            print('No')
            selected_answer = 'No'
        else:
            print('None')
        
        left_color = (0, 255, 0)
        right_color = (0, 255, 0)

        if selected_answer:
            if timers_start is None:
                timers_start = time.time()
            elif time.time() - timers_start >= 1:
                print(f'Selected: {selected_answer}')
                if selected_answer == correct_answer:
                    print('Correct!')
                    correct_count += 1
                else:
                    print('Wrong!')
                selected_answer = None
                timers_start = None
                break
        else:
            timers_start = None

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

while True:
    ret, frame = capture.read()
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    text = f"Finish! Your Score: {correct_count} / {len(quiz_data['questions'])}"
    wrapper_text = textwrap.TextWrapper(width=19)
    word_list = wrapper_text.wrap(text=text)
    font = ImageFont.truetype(font_path, 100)
    y_text = height - 350
    for line in word_list:
        draw.text((50, y_text), line, font=font, fill=(255, 255, 255, 0))
        y_text += 100
    frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break


capture.release()
cv2.destroyAllWindows()