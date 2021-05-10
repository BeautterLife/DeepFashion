import json
import os
import numpy as np
import cv2
import math

flag = True
save_path = 'C:/Users/pgh98/Desktop/train/class_div5/'
categories = ['short sleeve top', 'long sleeve top', 'short sleeve outwear', 'long sleeve outwear', 'vest', 'sling',
              'shorts', 'trousers', 'skirt', 'short sleeve dress', 'long sleeve dress', 'vest dress', 'sling dress']
class_count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
standard_height = 300
standard_width = 300
black_color = (0, 0, 0)
divSet = {0, 1, 2, 3, 4}

cnt=0
class Node():
    def __init__(self, g, num, cur, nex):
        self.group = g.copy()
        self.group.add(num)
        self.number = num
        self.left = None
        self.right = None
        self.img = ~(cv2.add(cur.copy(), nex[num]))


    def setChild(self, nex):
        if len(self.group) >= 5:
            return

        self.left = Node(self.group, min(divSet - self.group), ~self.img, nex)
        self.left.setChild(nex)

        self.right = Node(self.group, max(divSet - self.group), ~self.img, nex)
        self.right.setChild(nex)


def preOrder(node):
    if node.left is None:
        cv2.imshow(str(node.group), node.img)
        cv2.waitKey(1000)
        cv2.destroyWindow(str(node.group))
        return
    cv2.imshow(str(node.group), node.img)
    cv2.waitKey(1000)
    cv2.destroyWindow(str(node.group))
    preOrder(node.left)
    preOrder(node.right)


root = Node(set(),0)
root.setChild()
preOrder(root)




def find_item_area(seg):
    min_x = min_y = 3000
    max_x = max_y = 0
    for i in range(len(seg)):
        if seg[i][0] > max_x: max_x = seg[i][0]
        if seg[i][0] < min_x: min_x = seg[i][0]
        if seg[i][1] > max_y: max_y = seg[i][1]
        if seg[i][1] < min_y: min_y = seg[i][1]
    if min_x < 0: min_x = 0
    if min_y < 0: min_y = 0
    return min_x, min_y, max_x, max_y


# 리턴 이미지는 사이즈 320x320
# grayscale로 변환하여 0,255 값만 가지는 단일 채널 이미지
def resizeByRatioAnd2Gray(img):
    final_img = np.zeros((320, 320, 3), np.uint8) + 255
    height = img.shape[0]
    width = img.shape[1]
    ratio = standard_height / height if height > width else standard_width / width
    contour_img = cv2.resize(img, None, fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)

    contour_height = contour_img.shape[0]
    contour_width = contour_img.shape[1]
    final_img[(320 - contour_height) // 2:(320 - contour_height) // 2 + contour_height,
    (320 - contour_width) // 2:(320 - contour_width) // 2 + contour_width] = contour_img

    f, final_img = cv2.threshold(final_img, 200, 255, cv2.THRESH_BINARY)
    final_img = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('a',final_img)
    # cv2.waitKey(1000)

    return final_img

cnt=0
for root, dirs, files in os.walk('C:/Users/pgh98/Desktop/train/annos'):
    for fname in files:
        full_fname = os.path.join(root, fname)
        with open(full_fname, 'r') as f:
            photo_infos = json.load(f)
        cnt+=1
        print(cnt)
        # segmenation좌표를 담을 넘파이 배열
        item1_segmentation = [[1, 2]]
        item1_segmentation = np.array(item1_segmentation).reshape(-1, 2)

        category = photo_infos['item1']['category_name']

        for i in photo_infos['item1']['segmentation']:
            item1_segmentation2 = np.array([i].copy(), np.int32)
            for ele2 in item1_segmentation2.reshape(-1, 2):
                if ele2 not in item1_segmentation.reshape((-1, 2)):
                    item1_segmentation = np.append(item1_segmentation, item1_segmentation2)

        item1_segmentation = np.delete(item1_segmentation, 0)
        item1_segmentation = np.delete(item1_segmentation, 0)
        item1_segmentation = item1_segmentation.reshape(-1, 2)

        min_x, min_y, max_x, max_y = find_item_area(item1_segmentation)

        for i in range(13):
            if category == categories[i]:
                class_num = i
                break

        class_count[class_num]+=1

        strokeLen = 0
        row = len(item1_segmentation)
        strokeArr = []
        # print(len(item1_segmentation),' ',row)
        for i in range(len(item1_segmentation)):
            item1_segmentation[-i][0] -= (min_x - 1)
            item1_segmentation[-i][1] -= (min_y - 1)

            strokeLen += math.sqrt(math.pow((item1_segmentation[-i][0] - item1_segmentation[-i - 1][0]), 2) + math.pow(
                (item1_segmentation[-i][1] - item1_segmentation[-i - 1][1]), 2))
            strokeArr.append(strokeLen)
        canvas = np.zeros((max_y - min_y + 10, max_x - min_x + 10, 3), np.uint8) + 255

        img1 = canvas.copy()
        partition = strokeLen / 5

        part_five = []
        strokeLength = 0
        img = canvas.copy()


        divArr = [0]
        for i in range(len(item1_segmentation)):
            img = cv2.line(img, (item1_segmentation[-i][0], item1_segmentation[-i][1]),
                           (item1_segmentation[-i - 1][0], item1_segmentation[-i - 1][1]), black_color, 2)
            if strokeArr[i] / partition >= len(part_five) + 1 and len(part_five) < 4:
                divArr.append(i)
                part_five.append(resizeByRatioAnd2Gray(img))

                img = canvas.copy()

        part_five.append(resizeByRatioAnd2Gray(img))
        part_five.reverse()
        part_five_gray = [i for i in part_five]
        for i in range(4):
            part_five_gray[i + 1] = ~(cv2.add(~part_five_gray[i], ~part_five[i + 1]))
        for i in range(5):
            cv2.imwrite(save_path+category+'/'+str(class_count[class_num]).zfill(6)+"-"+str(i+1)+'.jpg',part_five_gray[i])


        if 'item2' in photo_infos.keys():

            item2_segmentation = [[1, 2]]
            item2_segmentation = np.array(item2_segmentation).reshape(-1, 2)

            category = photo_infos['item2']['category_name']
            for i in photo_infos['item2']['segmentation']:
                item2_segmentation2 = np.array([i].copy(), np.int32)
                for ele2 in item2_segmentation2.reshape(-1, 2):
                    if ele2 not in item2_segmentation.reshape((-1, 2)):
                        item2_segmentation = np.append(item2_segmentation, item2_segmentation2)


            item2_segmentation = np.delete(item2_segmentation, 0)
            item2_segmentation = np.delete(item2_segmentation, 0)
            item2_segmentation = item2_segmentation.reshape(-1, 2)

            min_x, min_y, max_x, max_y = find_item_area(item2_segmentation)

            for i in range(13):
                if category == categories[i]:
                    class_num = i
                    break

            class_count[class_num] += 1

            strokeLen = 0
            row = len(item2_segmentation)
            strokeArr = []
            # print(len(item1_segmentation),' ',row)
            for i in range(len(item2_segmentation)):
                item2_segmentation[-i][0] -= (min_x - 1)
                item2_segmentation[-i][1] -= (min_y - 1)

                strokeLen += math.sqrt(
                    math.pow((item2_segmentation[-i][0] - item2_segmentation[-i - 1][0]), 2) + math.pow(
                        (item2_segmentation[-i][1] - item2_segmentation[-i - 1][1]), 2))
                strokeArr.append(strokeLen)
            canvas = np.zeros((max_y - min_y + 10, max_x - min_x + 10, 3), np.uint8) + 255
            img1 = canvas.copy()
            partition = strokeLen / 5

            # 5등분 그리는 부분.

            part_five = []
            strokeLength = 0
            img = canvas.copy()


            for i in range(len(item2_segmentation)):
                img = cv2.line(img, (item2_segmentation[-i][0], item2_segmentation[-i][1]),
                               (item2_segmentation[-i - 1][0], item2_segmentation[-i - 1][1]), black_color, 2)

                if strokeArr[i] / partition >= len(part_five) + 1 and len(part_five) < 4:
                    part_five.append(resizeByRatioAnd2Gray(img))
                    img = canvas.copy()

            part_five.append(resizeByRatioAnd2Gray(img))
            part_five.reverse()
            part_five_gray = [i for i in part_five]
            for i in range(4):
                part_five_gray[i + 1] = ~(cv2.add(~part_five_gray[i], ~part_five[i + 1]))
            for i in range(5):
                #cv2.imshow('d',part_five_gray[i])
                #cv2.waitKey(1000)
                cv2.imwrite(save_path + category + '/' + str(class_count[class_num]).zfill(6) + "-" + str(
                    i + 1) + '.jpg', part_five_gray[i])
