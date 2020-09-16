import cv2
import numpy as np
import matplotlib.pyplot as plt

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    print(image.shape)
    y1 = image.shape[0]
    y2 = int(y1*(2/5))
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    return np.array([x1, y1, x2, y2])

def averaged_slope_interccept(image, lines):
    left_fit =[]
    right_fit=[]
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope<0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])

def canny(image):
    #사진을 회색으로 바꾼다
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #5x5 정사각형의 범위에 GaussianBlur을 도입한다.
    blur = cv2.GaussianBlur(gray,(5,5),0)
    #모서리 감지 알고리즘
    canny = cv2.Canny(blur, 50, 150)
    return canny

def display_lines(image, lines):
    #사진과 크기가 똑같은 검은 사진을 만든다.
    line_image = np.zeros_like(image)
    #lines배열로 받은 사진속 선을 검정색 사진에 표시한다.
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            cv2.line(line_image, (x1,y1), (x2,y2), (255,0,0), 10) #10은 선의 굵기 이다.
    return line_image


def region_of_interest(image):
    #필요한 부분을 자를때 사용한다.
    height = image.shape[0]
    width = image.shape[1]
    polygons = np.array([
        [(0,500), (width//2,400), (width, 500), (width, 400), (600, 200), (400, 200), (0,400)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def line_detect(image):
    canny_image = canny(image)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=10, maxLineGap=5)
    averaged_lines = averaged_slope_interccept(image, lines)
    line_image = display_lines(image, averaged_lines)
    combo_image =  cv2.addWeighted(image, 0.8, line_image, 1, 1)
    return combo_image

# # image = cv2.imread('test_image.jpg')
# image = cv2.imread('school_lane.jpg')
image = cv2.imread('test.jpg')
lane_image = np.copy(image)
canny = canny(lane_image)
cropped_image = region_of_interest(canny)
#라인 감지
#사진속의 선을 디텍트하는 휴 알고리즘을 돌릴때 세타 값을 1라디안씩 올린다.
lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=10, maxLineGap=5)


# 선의 평균 기울기를 구하여 구한다
averaged_lines = averaged_slope_interccept(lane_image, lines)


# 기존 사진과 크기가 같은 검은 사진에 평균 선 표시
line_image = display_lines(lane_image, averaged_lines)


#기존 사진에 라인 표시
combo_image =  cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)



# plt.imshow(canny)
# plt.show()
cv2.imshow("result",image)
cv2.waitKey(0)
cv2.imshow("result", canny)
cv2.waitKey(0)
cv2.imshow("result", cropped_image)
cv2.waitKey(0)
cv2.imshow("result", line_image)
cv2.waitKey(0)
cv2.imshow("result", combo_image)
cv2.waitKey(0)
result_image = line_detect(image)
cv2.imshow("result", result_image)
cv2.waitKey(0)
