# -*- coding: utf-8 -*-
import math
import cv2


if __name__ == "__main__":
    p = '/Users/zhenwang/code/git/ai/ocr/chineseocr_lite/麦芒6_QQ.png'
    img = cv2.imread(p)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape
    min_border = min(height, width)
    min_line_len = min_border // 3

    temp = img.copy()
    _, threshold = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < min_line_len or h < min_line_len or w >= min_border or h >= min_border:
            continue
        approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        cv2.drawContours(temp, [approx], 0, (0, 0, 255), 5)

    '''
    edges = cv2.Canny(gray, 50, 150, apertureSize=5)
    lines = cv2.HoughLinesP(edges, 1, math.pi/180, min_line_len, 10)
    for line in lines:
        for x1, y1, x2, y2 in line:
            # if abs(x1-x2) < min_line_len or abs(y1-y2) < min_line_len:
            #     continue
            cv2.line(temp, (x1, y1), (x2, y2), (0, 0, 255), 2)
    '''
    cv2.imwrite("debug_im/lines.jpg", temp)


