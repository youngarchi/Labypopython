import numpy as np
import cv2

cap = cv2.VideoCapture('desktop\mydr.mp4')




while (cap.isOpened()):
    ret, frame = cap.read()
    # перевірка чи зчитався кадр
    if ret == False:
        break
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame', frame)

    # перевірка натискання клавіші виходу
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


cap = cv2.VideoCapture('desktop\mydr.mp4')

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

print("Width = ", width)
print("Height = ", height)
print("FPS = ", fps)

fourcc = int(cv2.VideoWriter_fourcc(*'XVID'))

out = cv2.VideoWriter('desktop\results.avi',fourcc, fps, (width,height))

font = cv2.FONT_HERSHEY_SIMPLEX
text_x = int(width/2)-100
text_y = height - 30

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:

        # повернемо зображення
        frame = cv2.flip(frame, 0)

        # додамо рядок
        cv2.putText(frame, 'Video Processing', (text_x, text_y), font, 4, (255, 255, 255), 2, cv2.LINE_AA)

        # запишемо змінені кадри
        out.write(frame)

        # відобразимо змінені кадри
        cv2.imshow('frame', frame)

        # перевірка виходу
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# закриємо всі вікна та процеси запису
cap.release()
out.release()
cv2.destroyAllWindows()

