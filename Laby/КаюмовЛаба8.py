import cv2
import numpy as np


def nothing(x):
    pass



img = np.zeros((300,512,3), np.uint8)
cv2.namedWindow('image')

cv2.createTrackbar('R','image',0,255,nothing)
cv2.createTrackbar('G','image',0,255,nothing)
cv2.createTrackbar('B','image',0,255,nothing)



switch = '0 : OFF \n1 : ON'
cv2.createTrackbar(switch, 'image',0,1,nothing)




#Нескінченний цикл
while(1):
    cv2.imshow('image',img)
    # Перевірка для виходу з циклу (Esc)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    # Отримуємо поточне положення кожного з трекбарів
    r = cv2.getTrackbarPos('R','image')
    g = cv2.getTrackbarPos('G','image')
    b = cv2.getTrackbarPos('B','image')
    s = cv2.getTrackbarPos(switch,'image')
    # Змінюємо значення кожного кольору відповідно до значення трекбару
    if s == 0:
        img[:] = 0
    else:
        img[:] = [b,g,r]

cv2.destroyAllWindows()






image = cv2.imread("assignments/ooep.png")
out = image



def filtering(vignetteScale, light, y_ch, output):

    #split into channels
    B, G, R = cv2.split(output)

    #define vignette scale
    #vignetteScale = 6

    if (vignetteScale == 0):
        return image

    #calculate the kernel size
    k = np.min([output.shape[1], output.shape[0]])/vignetteScale

    #create kernel to get the Halo effect
    kernelX = cv2.getGaussianKernel(output.shape[1], k)
    kernelY = cv2.getGaussianKernel(output.shape[0], k)
    kernel = kernelY * kernelX.T

    #normalize the kernel
    mask = cv2.normalize(kernel, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    #apply halo effect to all the three channels of the image
    B = B + B*mask
    G = G + G*mask
    R = R + R*mask

    #merge back the channels
    output = cv2.merge([B, G, R])

    output = output /light

    #limit the values between 0 and 255
    output = np.clip(output, 0, 255)

    #convert back to uint8
    output = np.uint8(output)

    #split the channels
    B, G, R = cv2.split(output)


    #Interpolation values
    redValuesOriginal = np.array([0, 42, 105, 148, 185, 255])
    redValues =         np.array([0, 28, 100, 165, 215, 255 ])
    greenValuesOriginal = np.array([0, 40, 85, 125, 165, 212, 255])
    greenValues =         np.array([0, 25, 75, 135, 185, 230, 255 ])
    blueValuesOriginal = np.array([0, 40, 82, 125, 170, 225, 255 ])
    blueValues =         np.array([0, 38, 90, 125, 160, 210, 222])

    #create lookuptable
    allValues = np.arange(0, 256)

    #create lookup table for red channel
    redLookuptable = np.interp(allValues, redValuesOriginal, redValues)
    #apply the mapping for red channel
    R = cv2.LUT(R, redLookuptable)

    #create lookup table for green channel
    greenLookuptable = np.interp(allValues, greenValuesOriginal, greenValues)
    #apply the mapping for red channel
    G = cv2.LUT(G, greenLookuptable)

    #create lookup table for blue channel
    blueLookuptable = np.interp(allValues, blueValuesOriginal, blueValues)
    #apply the mapping for red channel
    B = cv2.LUT(B, blueLookuptable)

    #merge back the channels
    output = cv2.merge([B, G, R])

    #convert back to uint8
    output = np.uint8(output)

    #adjust contrast
    #convert to YCrCb color space
    output = cv2.cvtColor(output, cv2.COLOR_BGR2YCrCb)

    #convert to float32
    output = np.float32(output)

    #split the channels
    Y, Cr, Cb = cv2.split(output)

    #scale the Y channel
    #Y = Y * 1.2
    Y = Y * (1 + y_ch/10)


    #limit the values between 0 and 255
    Y = np.clip(Y, 0, 255)

    #merge back the channels
    output = cv2.merge([Y, Cr, Cb])

    #convert back to uint8
    output = np.uint8(output)

    #convert back to BGR color space
    output = cv2.cvtColor(output, cv2.COLOR_YCrCb2BGR)
    return output




#Trackbar
cv2.namedWindow("xpro", cv2.WINDOW_AUTOSIZE)
cv2.createTrackbar('vignetteScale','xpro',0,50,nothing)
cv2.createTrackbar('light','xpro',2,50,nothing)
cv2.createTrackbar('y_ch','xpro',1,10,nothing)


while (1):
    cv2.imshow("xpro", out)

    # Перевірка для виходу з циклу (Esc)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    vignetteScale = cv2.getTrackbarPos('vignetteScale', 'xpro')
    light = cv2.getTrackbarPos('light', 'xpro')
    y_ch = cv2.getTrackbarPos('y_ch', 'xpro')
    # Перевірка значень
    if (light == 0) or (y_ch == 0):
        light = 1
        y_ch = 1

    out = filtering(vignetteScale, light, y_ch, image)

cv2.destroyAllWindows()
