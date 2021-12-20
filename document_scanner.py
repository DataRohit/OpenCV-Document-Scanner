import cv2 as cv
import numpy as np
import utlis

########################################################################
webCamFeed = True
cap = cv.VideoCapture(1)
cap.set(10,160)
heightImg = 640
widthImg  = 480
########################################################################

utlis.initializeTrackbars()
count=0

while True:

    if webCamFeed:
        success, img = cap.read()
    else:
        print("Webcam Footage Not Detected")
        break

    img = cv.resize(img, (widthImg, heightImg)) # Resizing the Image
    imgBlank = np.zeros((heightImg,widthImg, 3), np.uint8) # Blank Image for Mask, Contours, etc.
    imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # Convert Image to Grayscale
    imgBlur = cv.GaussianBlur(imgGray, (5, 5), 1) # Add Gaussian Blur to Grayscale Image

    thres=utlis.valTrackbars() # Get Track Bar Values for Thresholds
    imgThreshold = cv.Canny(imgBlur,thres[0],thres[1]) # Applying Canny Edge Detection

    kernel = np.ones((5, 5))
    imgDial = cv.dilate(imgThreshold, kernel, iterations=2) # Applying Dilation on the Image
    imgThreshold = cv.erode(imgDial, kernel, iterations=1)  # Applying Erosion on the Image

    # Find all the Contours in the Image
    imgContours = img.copy()
    imgBigContour = img.copy()
    contours, hierarchy = cv.findContours(imgThreshold, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(imgContours, contours, -1, (0, 255, 0), 10) # Draw all Detected Contours

    # Find the Biggest Contour
    biggest, maxArea = utlis.biggestContour(contours)
    if biggest.size != 0:
        biggest=utlis.reorder(biggest)
        cv.drawContours(imgBigContour, biggest, -1, (0, 255, 0), 20) # Draw the Biggest Contour

        imgBigContour = utlis.drawRectangle(imgBigContour,biggest,2)
        pts1 = np.float32(biggest) # Prepare Points for Wrap
        pts2 = np.float32([[0, 0],[widthImg, 0], [0, heightImg],[widthImg, heightImg]])
        matrix = cv.getPerspectiveTransform(pts1, pts2)

        imgWarpColored = cv.warpPerspective(img, matrix, (widthImg, heightImg))

        # Remove 20 Pixels from each side
        imgWarpColored=imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
        imgWarpColored = cv.resize(imgWarpColored,(widthImg,heightImg))

        # Applying Adaptive Threshold
        imgWarpGray = cv.cvtColor(imgWarpColored,cv.COLOR_BGR2GRAY)
        imgAdaptiveThre= cv.adaptiveThreshold(imgWarpGray, 255, 1, 1, 7, 2)
        imgAdaptiveThre = cv.bitwise_not(imgAdaptiveThre)
        imgAdaptiveThre=cv.medianBlur(imgAdaptiveThre,3)

        # Image Array for Display
        imageArray = ([img,imgGray,imgThreshold,imgContours],
                      [imgBigContour,imgWarpColored, imgWarpGray,imgAdaptiveThre])

    else:
        imageArray = ([img,imgGray,imgThreshold,imgContours],
                      [imgBlank, imgBlank, imgBlank, imgBlank])

    # labels for Each Image
    lables = [["Original","Gray","Threshold","Contours"],
              ["Biggest Contour","Warp Prespective","Warp Gray","Adaptive Threshold"]]

    stackedImage = utlis.stackImages(imageArray,0.75,lables)
    cv.imshow("Result",stackedImage)

    # Save the Image when 's' key is pressed
    if cv.waitKey(1) & 0xFF == ord('s'):
        cv.imwrite("./Scanned/myImage"+str(count)+".jpg", imgWarpColored)
        cv.rectangle(stackedImage, ((int(stackedImage.shape[1] / 2) - 230), int(stackedImage.shape[0] / 2) + 50),
                      (1100, 350), (0, 255, 0), cv.FILLED)
        cv.putText(stackedImage, "Scan Saved", (int(stackedImage.shape[1] / 2) - 200, int(stackedImage.shape[0] / 2)),
                    cv.FONT_HERSHEY_DUPLEX, 3, (0, 0, 255), 5, cv.LINE_AA)
        cv.imshow('Result', stackedImage)
        cv.waitKey(300)
        count += 1
        
    elif cv.waitKey(1) & 0xFF == ord('q'):
        break