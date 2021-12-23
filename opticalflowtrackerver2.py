### Optical Flow를 사용하여 객체의 움직임 여부를 결정하고, 멈춰 있을 시에 operating 중인지 (flow 값이 충분히 변하는지) 판별합니다.
### 이 CLASS 는 기본적으로 매 프레임 별 update 함수를 통해 호출하여 진행됩니다.
### ex) OpticalFlowTracker.update(오브젝트의 바운딩 박스 좌표 리스트, 이전 프레임 grayscale 이미지, 새 프레임 RGB 이미지)
### objects = [[현재 x_centroid, y_centroid], [x_min, y_min, x_max, y_max], deque(프레임 간 centroid 변위), [bigbox optical flow의 x_centroid, y_centroid],
###           [bigbox optical flow의 x_min, y_min, x_max, y_max], deque(프레임 간 bigbox centroid 변위), 상태 판별자, 상태, [detection 시 x길이, y길이], [detection 시 xmin, ymin, xmax, ymax]]

from collections import OrderedDict, deque
import cv2 as cv
import numpy as np

class OpticalFlowTracker():
    def __init__(self, moveflag_cycle=5, moveflag_threshold=5):
        self.nextObjectID = 1
        self.objects = OrderedDict()
        self.flows = OrderedDict()
        self.bigflows = OrderedDict()
        self.fastDT = cv.FastFeatureDetector_create(30)
        self.big_lk_params = dict(winSize=(15, 15), maxLevel=2,
                                  criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
        self.lk_params = dict(winSize=(25,25), maxLevel=2,
                              criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
        self.moveflag_cycle = moveflag_cycle
        self.moveflag_threshold = moveflag_threshold

    def update(self, rects, old_gray, image):
        for rect in rects:
            xmin, ymin, xmax, ymax = rect
            self.objects[self.nextObjectID] = [[(xmin + xmax) // 2, (ymin + ymax) // 2], [xmin, ymin, xmax, ymax], deque(),
                                               [-1, -1], [-1, -1, -1, -1], deque(), [0,0], 0, [xmax-xmin, ymax-ymin], [xmin, ymin, xmax, ymax]]
            self.nextObjectID += 1
        self.run_opticalflow(old_gray, image)
        self.run_bigopticalflow(old_gray, image)
        self.track_with_flow()
        self.track_with_bigflow()
        self.check_status()

    def initiate_opticalflow(self, objectID, image):
        if objectID in self.flows.keys():
            del self.flows[objectID]
        xmin, ymin, xmax, ymax = self.objects[objectID][-1]
        xlen, ylen = self.objects[objectID][-2]
        x_cutting_length = round(xlen / 2.5)
        y_cutting_length = round(ylen / 3)
        detectedpoints = self.fastDT.detect(image[ymin + y_cutting_length:ymax - y_cutting_length,
                                            xmin + x_cutting_length:xmax - x_cutting_length], None)

        points = cv.KeyPoint_convert(detectedpoints).reshape(-1, 1, 2)
        points[:, 0, 0] += (xmin + x_cutting_length)
        points[:, 0, 1] += (ymin + y_cutting_length)
        st = np.full_like(points[:, :, 0], 1)
        self.flows[objectID] = [points, st, np.zeros_like(st, dtype=float)]

    def run_opticalflow(self, old_gray, image):
        # 없어진 object 에 대해 flows 삭제
        copied_flows_keys = set(self.flows.keys())
        for objectID in copied_flows_keys:
            if objectID not in self.objects.keys():
                del self.flows[objectID]

        # 새로 등록된 object 에 대해 flows 에 OF 값 기입
        for objectID, data in self.objects.items():
            if objectID not in self.flows.keys():
                self.initiate_opticalflow(objectID, image)
            else:
                st = self.flows[objectID][1]
                points = self.flows[objectID][0][st == 1]
                points = np.reshape(points, (-1, 1, 2))
                p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, image, points, None, **self.lk_params)
                diff = self.flows[objectID][2]
                for i in range(len(p1)):
                    if st[i] == 1:
                        diff[i] += (abs(p1[i][0][1] - points[i][0][1]) + abs(p1[i][0][0] - points[i][0][0]))

                self.flows[objectID] = [p1, st, diff]

    def initiate_bigopticalflow(self, objectID):
        if objectID in self.bigflows.keys():
            del self.bigflows[objectID]
        xcen, ycen = self.objects[objectID][0]
        halfxlen, halfylen = map(lambda x: x/2, self.objects[objectID][-2])
        xmin, xmax = int(xcen - halfxlen), int(xcen + halfxlen)
        ymin, ymax = int(ycen - halfylen), int(ycen + halfylen)
        point_list = []
        for _y in range(ymin, ymax, 10):
            for _x in range(xmin, xmax, 10):
                point_list.append((_x, _y))
        points = np.array(point_list)
        points = np.float32(points[:, np.newaxis, :])
        st = np.full_like(points[:,:,0], 1)
        self.bigflows[objectID] = [points, st, np.zeros_like(st, dtype=float)]

    def run_bigopticalflow(self, old_gray, image):
        # 없어진 object 에 대해 flows 삭제
        copied_flows_keys = set(self.bigflows.keys())
        for objectID in copied_flows_keys:
            if objectID not in self.objects.keys():
                del self.bigflows[objectID]

        # 새로 등록된 object 에 대해 flows 에 OF 값 기입
        for objectID, data in self.objects.items():
            if objectID not in self.bigflows.keys():
                self.initiate_bigopticalflow(objectID)
            else:
                st = self.bigflows[objectID][1]
                points = self.bigflows[objectID][0][st == 1]
                points = np.reshape(points, (-1, 1, 2))
                p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, image, points, None, **self.big_lk_params)
                diff = self.bigflows[objectID][2]
                for i in range(len(p1)):
                    if st[i] == 1:
                        diff[i] += (abs(p1[i][0][1] - points[i][0][1]) + abs(p1[i][0][0] - points[i][0][0]))
                self.bigflows[objectID] = [p1, st, diff]

    def track_with_flow(self):
        for objectID, flow in self.flows.items():
            st = flow[1]
            if len(flow[0][st==1]) == 0:
                del self.flows[objectID]
                del self.objects[objectID]
                continue
            xmin = min(flow[0][st==1][:, 0])
            ymin = min(flow[0][st==1][:, 1])
            xmax = max(flow[0][st==1][:, 0])
            ymax = max(flow[0][st==1][:, 1])
            xcen, ycen = (xmin + xmax)//2, (ymin + ymax)//2
            status_checker = [0, 0]
            cent_dif = self.objects[objectID][2]
            if len(cent_dif) >= self.moveflag_cycle:
                cent_dif.popleft()
                cent_dif.append(abs(self.objects[objectID][0][0] - xcen) + abs(self.objects[objectID][0][1] - ycen))
                if sum(cent_dif) > self.moveflag_threshold:
                    if objectID == 1 and (self.objects[objectID][-1][0] > xcen or xcen > self.objects[objectID][-1][2]):
                        status_checker[0] = 2
                    elif objectID == 2:
                        status_checker[0] = 2
                else:
                    status_checker[0] = 1
            else:
                cent_dif.append(abs(self.objects[objectID][0][0] - xcen) + abs(self.objects[objectID][0][1] - ycen))

            if objectID == 3:
                print('flowdif', sum(cent_dif))

            self.objects[objectID][0] = [(xmin + xmax)//2, (ymin + ymax)//2]
            self.objects[objectID][1] = [int(xmin), int(ymin), int(xmax), int(ymax)]
            self.objects[objectID][2] = cent_dif
            self.objects[objectID][6] = status_checker

    def track_with_bigflow(self):
        for objectID, flow in self.bigflows.items():
            st = flow[1]
            if len(flow[0][st==1]) == 0:
                del self.flows[objectID]
                del self.objects[objectID]
                continue
            xmin = min(flow[0][st == 1][:, 0])
            ymin = min(flow[0][st == 1][:, 1])
            xmax = max(flow[0][st == 1][:, 0])
            ymax = max(flow[0][st == 1][:, 1])
            xcen = sum(flow[0][st == 1][:, 0]) // len(flow[0][st == 1])
            ycen = sum(flow[0][st == 1][:, 1]) // len(flow[0][st == 1])
            status_checker = self.objects[objectID][6]
            bigcent_dif = self.objects[objectID][5]
            if len(bigcent_dif) >= self.moveflag_cycle:
                bigcent_dif.popleft()
                bigcent_dif.append(abs(self.objects[objectID][3][0] - xcen) + abs(self.objects[objectID][3][1] - ycen))
                if sum(bigcent_dif) > self.moveflag_threshold:
                    status_checker[1] = 2
                else:
                    status_checker[1] = 1
            else:
                bigcent_dif.append(abs(self.objects[objectID][3][0] - xcen) + abs(self.objects[objectID][3][1] - ycen))

            if objectID == 3:
                print('bigflowdif', sum(bigcent_dif))

            self.objects[objectID][3] = [(xmin + xmax)//2, (ymin + ymax)//2]
            self.objects[objectID][4] = [int(xmin), int(ymin), int(xmax), int(ymax)]
            self.objects[objectID][5] = bigcent_dif
            self.objects[objectID][6] = status_checker

    def check_status(self):
        for objectID, data in self.objects.items():
            status_checker = data[6]
            if status_checker[0] == 0 or status_checker[1] == 0:
                status = 0
            elif status_checker[0] == 2 and status_checker[1] == 2:
                status = 2
                halfxlen = self.objects[objectID][-2][0] / 2
                halfylen = self.objects[objectID][-2][1] / 2
                self.objects[objectID][-1] = [self.objects[objectID][0][0] - halfxlen, self.objects[objectID][0][1] - halfylen,
                                              self.objects[objectID][0][0] + halfxlen, self.objects[objectID][0][1] + halfylen]
                self.initiate_bigopticalflow(objectID)
            else:
                status = 1
            data[7] = status
