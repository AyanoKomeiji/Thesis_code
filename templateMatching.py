import numpy as np
import cv2



class Matched_Rect:
    def __init__(self, max_loc, max_val):
        self.max_loc = max_loc
        self.max_val = max_val
        self.index = 0
        self.Flag = True


def set_index(Match_list, index1=np.zeros(2), index2=np.zeros(2), index3=np.zeros(2)):
    num = 0
    for match in Match_list:
        if num == 0:
            match.index = 1
            index1 = np.array(match.max_loc)
        else:
            if np.linalg.norm(np.array(match.max_loc) - index1) < 30:
                match.index = 1
            elif np.all(index2 == 0):
                match.index = 2
                index2 = np.array(match.max_loc)
            elif np.linalg.norm(np.array(match.max_loc) - index2) < 30:
                match.index = 2
            elif np.all(index3 == 0):
                match.index = 3
                index3 = np.array(match.max_loc)
            elif np.linalg.norm(np.array(match.max_loc) - index3) < 30:
                match.index = 3
        num += 1

def norm_vec(match,pre):
    dif = np.linalg.norm(np.array(match.max_loc) - np.array(pre.max_loc))
    return dif

def track_index(funds, match_list):
    difs1 = []
    difs2 = []
    difs3 = []
    dif_list=[]
    for match in match_list:
        if match != False:
            match.index=0
            dif1 = norm_vec(match, funds[0])
            dif2 = norm_vec(match, funds[1])
            dif3 = norm_vec(match, funds[2])
            difs1.append(dif1)
            difs2.append(dif2)
            difs3.append(dif3)
            dif_list.append([dif1,dif2,dif3])
    dif_list = np.array(dif_list)
    while np.all(dif_list == 100000) == False:
        num, index = np.where(dif_list == dif_list.min())
        for a,b in zip(num,index):
            match_list[a].index = funds[b].index
            dif_list[a, :] = 100000
            dif_list[:, b] = 100000

def occlusion_check(funds, best_match):
    for a in best_match:
        for b in best_match:
            if a!=False and b!=False and a!=b:
                if norm_vec(a,b)<50:
                    for fund in funds:
                        if fund.index==a.index and abs(fund.max_loc[1]-a.max_loc[1])>10:
                            new_index=a.index
                            a.index=b.index
                            b.index=new_index


def read(fn, num):
    im = cv2.imread(fn + str(num) + ".jpg", 0)
    # dst=np.array(im,dtype="float32")
    return im


def read_temp(temp_fn, num):
    temp_list = []
    for a in range(num):
        im = cv2.imread(temp_fn + str(a + 1) + ".png", 0)
        temp_list.append(im)
    return temp_list


def fft(img_re, top_left, bottom_right, num):
    img_fft = img_re[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    cv2.imwrite("data/fft/stick_1123_2/img_" + str(num) + ".jpg", img_fft)


def drow_rect(match, w, h, img_out):
    for Match in match:
        if Match != False:
            x = Match.max_loc[0]
            y = Match.max_loc[1]
            bottom_right = (x + w, y + h)
            if Match.index == 1:
                # index1 is Red
                cv2.rectangle(img_out, (x, y), bottom_right, (0, 0, 255), 2)
            elif Match.index == 2:
                # index2 is Green
                cv2.rectangle(img_out, (x, y), bottom_right, (0, 255, 0), 2)
            elif Match.index == 3:
                # index3 is Blue
                cv2.rectangle(img_out, (x, y), bottom_right, (255, 0, 0), 2)
            else:
                cv2.rectangle(img_out, (x, y), bottom_right, (255, 255, 255), 2)
            cv2.circle(img_out, (int(x + w / 2), int(y + h / 2)), 2, (0, 0, 0), thickness=-1)


def temp_Matches(fn, ofn, num, temp, funds=[], start=False):
    img = read(fn, num)
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    img_out = cv2.cvtColor(img, cv2.COLOR_GRAY2RGBA)
    # img = cv2.rectangle(img, (0, 920), (1080, 1920), (0, 0, 0), thickness=-1)
    img_re = img.copy()
    match_list = []
    out_list = []
    for temp_im in temp:
        res1 = cv2.matchTemplate(img_re, temp_im, cv2.TM_CCOEFF_NORMED)
        w, h = temp_im.shape[:2]
        ys, xs = np.where(res1 >= 0.8)
        for x, y in zip(xs, ys):
            val = res1[y][x]
            match_list.append(Matched_Rect([x, y], val))
    if start == True:
        set_index(match_list)
        best_Match = choice_Match(match_list)
        funds = best_Match
    else:
        set_index(match_list)
        best_Match = choice_Match(match_list)
        track_index(funds, best_Match)
        occlusion_check(funds,best_Match)
        new_funds = []
        false_index=[False,False,False]
        for best in best_Match:
            if best!=False:
                if best.index==1:
                    false_index[0]=True
                elif best.index==2:
                    false_index[1]=True
                elif best.index==3:
                    false_index[2]=True
        for a in range(3):
            if false_index[a]==True:
                for best in best_Match:
                    if best!=False:
                        if best.index==a+1:
                            new_funds.append(best)
                            break
            else:
                for fund in funds:
                    if fund.index==a+1:
                        new_funds.append(fund)
                        break
        funds = new_funds
    out_list.append(best_Match)
    drow_rect(best_Match, w, h, img_out)
    # center = [top_left[0] + w / 2 - img_re.shape[1] / 2, -(top_left[1] + h / 2 - img_re.shape[0] / 2)]
    for match in best_Match:
        if match != False:
            center = [match.max_loc[0] + w / 2 - img_re.shape[1] / 2,
                      match.max_loc[1] + h / 2 - img_re.shape[0] / 2, match.index, num]
            centers.append(center)
    # cv2.imwrite(ofn + "/img_" + str(num) + ".jpg", img_out)
    return funds, centers


def choice_Match(match_list):
    val1, val2, val3 = 0, 0, 0
    best1, best2, best3 = False, False, False
    for m in match_list:
        if m.index == 1:
            if val1 < m.max_val:
                best1 = m
                val1 = m.max_val
        if m.index == 2:
            if val2 < m.max_val:
                best2 = m
                val2 = m.max_val
        if m.index == 3:
            if val3 < m.max_val:
                best3 = m
                val3 = m.max_val
    best_Match = [best1, best2, best3]
    return best_Match


fn = "data/images/lower_1203/img_"
temp_fn = "data/template/template_lower/low_"
ofn = "data/template/lower_1203/"
# os.mkdir(ofn)
centers = []
temp = read_temp(temp_fn, 3)

start=2680-303
stop=21000-303
time=10

for num in range(start, stop, time):
    if num == start:
        funds, centers = temp_Matches(fn, ofn, num, temp, [], True)
    else:
        funds, centers = temp_Matches(fn, ofn, num, temp, funds, False)

np.savetxt("data/reconstruct/tops_1203_lower.csv", centers, delimiter=',', fmt='%.1f')

