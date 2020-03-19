# -*- coding: utf-8 -*-
import os
import glob
import re

from PIL import Image
import cv2
import numpy as np
import xlsxwriter

from model import text_predict, crnn_handle

headers = ["AudioContext", "webkitAudioContext", "mediaDevices", "mediaDevices.getUserMedia",
           "navigator.getUserMedia", "navigator.webkitGetUserMedia", "navigator.mozGetUserMedia",
           "是否支持录音", "是否支持canvas渲染"]


def find_contours(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape
    min_border = min(height, width)
    min_line_len = min_border // 3

    temp, all_contours = img.copy(), []
    _, threshold = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    cv2.imwrite('debug_im/binary.jpg', threshold)
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i, contour_ in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour_)

        # -- check contour's width & height
        if w < min_line_len or h < min_line_len or w >= min_border or h >= min_border:
            continue

        # -- check contour shape is a rectangle
        approx = cv2.approxPolyDP(contour_, 0.1 * cv2.arcLength(contour_, True), True)
        if len(approx) != 4:  # rectangle
            continue

        # -- check sub-rectangles
        child_idx = hierarchy[0, i][2]  # last index: [Next, Previous, First_Child, Parent]
        if child_idx != -1:  # have a child contour (e.g. any sub-rectangle)
            sub_contour = contours[child_idx]
            sub_approx = cv2.approxPolyDP(sub_contour, 0.1 * cv2.arcLength(sub_contour, True), True)
            if len(sub_approx) != 4:  # not a rectangle
                continue
            area = abs(cv2.contourArea(approx, True))
            sub_area = abs(cv2.contourArea(sub_approx, True))
            area_ratio = area / (sub_area + 0.01)
            if 0.7 > area_ratio or area_ratio > 1.3:  # region ratio check, approx = 1
                continue
        all_contours.append(np.squeeze(approx))  # shape: (n, 4)
        cv2.drawContours(temp, [approx], 0, (0, 0, 255), 5)
    cv2.imwrite("debug_im/img_all_contours.jpg", temp)
    return all_contours


def edit_distance(s1, s2, i, j):
    if i <= 0 and j <= 0:
        return 0
    elif i <= 0 and j != 0:
        return j
    elif i != 0 and j <= 0:
        return i
    else:  # i >= 1, j >= 1
        d_ij = int(s1[i] != s2[j])
        return min(edit_distance(s1, s2, i-1, j) + 1, edit_distance(s1, s2, i, j-1) + 1, edit_distance(s1, s2, i-1, j-1) + d_ij)


def sim_distance(s1, s2):
    s1_, s2_ = set(s1), set(s2)
    max_len = max(len(s1_), len(s2_))
    return len(s1_ & s2_) / max_len


def detect(imgpath, phone_model, browser):
    d = {
        "phone_model": phone_model,
        "platform": "IOS",
        "browser": browser,
        "AudioContext": None,
        "webkitAudioContext": None,
        "mediaDevices": None,
        "mediaDevices.getUserMedia": None,
        "navigator.getUserMedia": None,
        "navigator.webkitGetUserMedia": None,
        "navigator.mozGetUserMedia": None,
        "是否支持录音": None,
        "是否支持canvas渲染": None,
    }

    img = cv2.imread(imgpath)
    # cv2.imwrite('debug_im/raw.jpg', img)
    all_contour_indices = find_contours(img)
    if not all_contour_indices:
        return None, d  # ignore, no candidate contours

    is_success = True
    for contour_indices in all_contour_indices:
        (x_min, y_min), (x_max, y_max) = contour_indices.min(0), contour_indices.max(0)
        img_roi = img[y_min: y_max, x_min: x_max, :]  # (h, w, c)
        cv2.imwrite('debug_im/crop_by_contour.jpg', img_roi)
        result = text_predict(img_roi)
        res = [{'text': x['text'],
                'name': str(i),
                'box': {'cx': x['cx'],
                        'cy': x['cy'],
                        'w': x['w'],
                        'h': x['h'],
                        'angle': x['degree']

                        }
                } for i, x in enumerate(result)]
        res.sort(key=lambda x: (x['box']['cy'], x['box']['cx']))

        header_ind = 0
        skip_indices = set()
        for i, line in enumerate(res):
            if i in skip_indices:
                continue

            if header_ind >= len(headers):
                break

            s1, h, s3, s4 = line['text'], headers[header_ind], 'true', 'false'
            if s1 == h or sim_distance(s1, h) >= 0.5:  # match key
                pass
            elif s1 == s3 or sim_distance(s3, s1) >= 0.5:  # match value: true
                d[h] = s1
                header_ind += 1
            elif s4 == s1 or sim_distance(s4, s1) >= 0.5:  # match value: false
                d[h] = s1
                header_ind += 1

        if header_ind != len(headers) or list(filter(lambda v: v is None, d.values())):  # fail
            is_success = False
        if is_success:
            break

    # print(d)
    return is_success, d


if __name__ == "__main__":
    workbook = xlsxwriter.Workbook('demo.xlsx')
    worksheet = workbook.add_worksheet('sheet1')

    row = 0
    for c in range(len(headers)):
        worksheet.write(row, c, headers[c])
    row += 1

    ios_paths = glob.glob('/data/dataset/ios/*/*/*')
    cnt_success, cnt_ignore, cnt_total = 0, 0, len(ios_paths)
    for imgpath in ios_paths:
        imgpath = '/data/chineseocr_lite/debug_im/fail01.jpg'
        phone_model = imgpath.split('/')[4]
        browser = '' #  imgpath.split('/')[5]
        is_success, d = detect(imgpath, phone_model, browser)
        if is_success is None:
            cnt_ignore += 1
        elif is_success:
            cnt_success += 1
            for c, header in enumerate(headers):
                worksheet.write(row, c, d[header])
            row += 1
    print(f"success_ratio={cnt_success}/{cnt_total} = {100*cnt_success/cnt_total: .2f}\t"
          f"ignore_ratio={cnt_ignore}/{cnt_total} = {100*cnt_ignore/cnt_total: .2f}")

    android_paths = []
    for root, dirs, files in os.walk(b"/data/dataset/android"):
        for name in files:
            if name.endswith((b'.jpg', b'.jpeg', b'.png')):
                name_ = name[: name.index(b'.')].decode('utf-8')
                name_ = re.sub(r"\s*\(\d+\)$", "", name_)
                phone_model = name_.split('_')[0]
                browser = name_.split('_')[1] if len(name_.split('_')) > 1 else 'origin'
                android_paths.append({
                    'path': os.path.join(root.decode('utf-8'), name.decode('utf-8')),
                    'phone_model': phone_model,
                    'browser': browser,
                })
        for name in dirs:
            for root_, _, files_ in os.walk(os.path.join(root, name)):
                for name in files_:
                    browser = root_.split(b'/')[-1]
                    if name.endswith((b'.jpg', b'.jpeg', b'.png')):
                        name_ = name[: name.index(b'.')].decode('utf8')
                        phone_model = re.sub(r"\s*\(\d+\)$", "", name_)
                        android_paths.append({
                            'path': os.path.join(root_.decode('utf8'), name.decode('utf8')),
                            'phone_model': phone_model,
                            'browser': browser.decode('utf8'),
                        })

    cnt_success, cnt_ignore, cnt_total = 0, 0, len(android_paths)
    for i, pd in enumerate(android_paths):
        imgpath = pd['path']
        phone_model = pd['phone_model']
        browser = pd['browser']

        if i > 0 and i % 50 == 0:
            print(f"success_ratio={cnt_success}/{i} = {100*cnt_success/i: .2f}\t"
                  f"ignore_ratio={cnt_ignore}/{i} = {100*cnt_ignore/i: .2f}")
        try:
            is_success, d = detect(imgpath, phone_model, browser)
            if is_success is None:
                cnt_ignore += 1
            elif is_success:
                cnt_success += 1
                for c, header in enumerate(headers):
                    worksheet.write(row, c, d[header])
                row += 1
        except Exception as ex:
            print(f"Image Path: {i}, path = {imgpath}")
            print(ex)
    print(f"success_ratio={cnt_success}/{cnt_total} = {100*cnt_success/cnt_total: .2f}\t"
          f"ignore_ratio={cnt_ignore}/{cnt_total} = {100*cnt_ignore/cnt_total: .2f}")
    workbook.close()
