### main 함수에는 bool_only_for_stopped 여부에 따라 거의 같은 코드가 두개로 나뉘어 있습니다.
### 한 쪽을 수정하면 나머지도 수정해 주어야 합니다.

import numpy as np
import cv2 as cv
from collections import OrderedDict
import pandas as pd
import copy
import time

PATH_TO_IMAGES = "/data/dykim/initial/jpg_files/"
INPUT_CSV = "/data/dykim/initial/tracking_csv/tracking_test1.csv"
STARTING_JPG_FILENAME_INDEX = 2102  # 돌리기 시작할 이미지 파일 index. csv 파일의 시작 행 데이터와 맞춰 주어야 함

### Visualization 할거면 True, 안할거면 False
bool_visualization = False

### stop 상태에 대해서만 OF 수행할거면 True, 아니면 False
bool_only_for_stopped = True

### OF difference 계산해서 csv에 열 데이터로 추가할거면 True, 아니면 False
bool_calculate_OF_difference = True


lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

color = np.random.randint(0, 255, (500, 3))


def optical_flow_run(old_gray, frame, objects, flows):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ### 없어진 object 에 대해 flows 삭제
    copied_flows = copy.deepcopy(flows)
    for objectID in copied_flows.keys():
        if objectID not in objects:
            del flows[objectID]

    ### 새로 등록된 object 에 대해 flows 에 OF 값 기입
    for objectID, rect in objects.items():
        if objectID not in flows.keys():
            xmin, ymin, xmax, ymax = rect
            point_list = []
            for _y in range(ymin, ymax, 10):
                for _x in range(xmin, xmax, 10):
                    point_list.append((_x, _y))
            points = np.array(point_list)
            points = np.float32(points[:, np.newaxis, :])
            p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, points, None, **lk_params)
            flows[objectID] = [p1, st]
        else:
            points = flows[objectID][0]
            p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, points, None, **lk_params)
            flows[objectID] = [p1, st]
    return flows


def run_visualization(frame, oldflows, flows):
    img = frame
    for objectID, dataset in flows.items():
        if objectID not in oldflows.keys():
            continue
        mask = np.zeros_like(frame)
        p1 = dataset[0]
        st = dataset[1]
        old_points = oldflows[objectID][0]
        # 좋은 포인트를 고른다
        good_new = p1[st == 1]
        good_old = old_points[st == 1]

        # 추적하는 것 그리기
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv.line(mask, (a, b), (c, d), color[i].tolist(), 2)
            tempframe = cv.circle(mask, (a, b), 3, color[i].tolist(), -1)
        img = cv.add(tempframe, img)

    if type(img) != int:
        cv.imshow('frame', img)
    k = cv.waitKey(1) & 0xff
    if k == 27:
        raise KeyError


def calculate_OF_difference(list_of_objects, oldflows, flows):
    OF_dif = OrderedDict()
    for objectID in list_of_objects:
        if objectID in oldflows.keys() and objectID in flows.keys():
            point_new = flows[objectID][0]
            st = flows[objectID][1]
            point_old = oldflows[objectID][0]
            good_new = point_new[st == 1]
            good_old = point_old[st == 1]
            diff = 0
            for new, old in zip(good_new, good_old):
                a, b = new.ravel()
                c, d = old.ravel()
                diff += abs(a - c) + abs(b - d)
            diff /= len(good_new)
            OF_dif[objectID] = diff

    return OF_dif


def main():
    data = pd.read_csv(INPUT_CSV)
    df = data.copy()
    if bool_calculate_OF_difference:
        df['OF_difference'] = -1
    current_frame = 1
    objects = OrderedDict()
    flows = OrderedDict()
    old_gray = cv.imread(PATH_TO_IMAGES + str(STARTING_JPG_FILENAME_INDEX).zfill(8) + ".jpg", cv.IMREAD_COLOR)
    old_gray = cv.cvtColor(old_gray, cv.COLOR_BGR2GRAY)

    if bool_only_for_stopped:
        for i in range(len(df)):  ###################################CCCCC
            if df.iloc[i, 1] == current_frame:
                if str(df.iloc[i, 10]) == 'stop':
                    objectID = df.iloc[i, 9]
                    if objectID not in objects.keys():
                        # objects[objectID] = (int(df.iloc[i, j]) for j in range(3, 7))
                        objects[objectID] = (int(df.iloc[i, 3]), int(df.iloc[i, 4]), int(df.iloc[i, 5]), int(df.iloc[i, 6]))
            else:
                image = cv.imread(
                    PATH_TO_IMAGES + str(STARTING_JPG_FILENAME_INDEX + current_frame - 1).zfill(8) + ".jpg",
                    cv.IMREAD_COLOR)
                oldflows = copy.deepcopy(flows)
                print("INFO : Running with image {}.jpg".format(str(STARTING_JPG_FILENAME_INDEX + current_frame - 1)))
                timecalc = time.time()
                flows = optical_flow_run(old_gray, image, objects, flows)
                print("INFO : Calculated with time {} ms".format(time.time() - timecalc))
                # Visualization 섹션
                if bool_visualization:
                    run_visualization(image, oldflows, flows)
                # OF_difference_calculation 섹션
                if bool_calculate_OF_difference:
                    OF_dif = calculate_OF_difference(list(objects.keys()), oldflows, flows)
                    for j in range(i - 1, 0, -1):
                        if df.iloc[j, 1] != current_frame:
                            break
                        if df.iloc[j, 9] in OF_dif.keys():
                            df.iloc[j, 11] = OF_dif[df.iloc[j, 9]]
                current_frame = df.iloc[i, 1]
                objects = OrderedDict()
                objectID = df.iloc[i, 9]
                if str(df.iloc[i, 10]) == 'stop':
                # objects[objectID] = (int(df.iloc[i, j]) for j in range(3, 7))
                    objects[objectID] = (int(df.iloc[i, 3]), int(df.iloc[i, 4]), int(df.iloc[i, 5]), int(df.iloc[i, 6]))
                old_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    else:
        for i in range(len(df)):  ###################################CCCCC
            if df.iloc[i, 1] == current_frame:
                objectID = df.iloc[i, 9]
                if objectID not in objects.keys():
                    # objects[objectID] = (int(df.iloc[i, j]) for j in range(3, 7))
                    objects[objectID] = (int(df.iloc[i, 3]), int(df.iloc[i, 4]), int(df.iloc[i, 5]), int(df.iloc[i, 6]))
            else:
                image = cv.imread(
                    PATH_TO_IMAGES + str(STARTING_JPG_FILENAME_INDEX + current_frame - 1).zfill(8) + ".jpg",
                    cv.IMREAD_COLOR)
                oldflows = copy.deepcopy(flows)
                print("INFO : Running with image {}.jpg".format(str(STARTING_JPG_FILENAME_INDEX + current_frame - 1)))
                flows = optical_flow_run(old_gray, image, objects, flows)
                # Visualization 섹션
                if bool_visualization:
                    run_visualization(image, oldflows, flows)
                # OF_difference_calculation 섹션
                if bool_calculate_OF_difference:
                    OF_dif = calculate_OF_difference(list(objects.keys()), oldflows, flows)
                    for j in range(i - 1, 0, -1):
                        if df.iloc[j, 1] != current_frame:
                            break
                        if df.iloc[j, 9] in OF_dif.keys():
                            df.iloc[j, 11] = OF_dif[df.iloc[j, 9]]
                current_frame = df.iloc[i, 1]
                objects = OrderedDict()
                objectID = df.iloc[i, 9]
                # objects[objectID] = (int(df.iloc[i, j]) for j in range(3, 7))
                objects[objectID] = (int(df.iloc[i, 3]), int(df.iloc[i, 4]), int(df.iloc[i, 5]), int(df.iloc[i, 6]))
                old_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    if bool_visualization:
        cv.destroyAllWindows()
        cap.release()
    if bool_calculate_OF_difference:
        df.to_csv(r"/data/dykim/initial/tracking_csv/tracking_test11.csv", index=False)  ###DESTINATION 파일 수정
    return flows


if __name__ == '__main__':
    print("INFO : Running Optical Flow Calculation. . .")
    flows = main()
    print('finished')
