# %%
import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
import colorsys
import math
import time
import pandas as pd
from pynput.keyboard import Controller

k = Controller()

# %%
Hindi_Dict = {
    "LHTap": {
        "IndexTop": "\u0915",
        "IndexMiddle": "\u0917",
        "IndexBottom": "\u0919",
        "IndexNail": "LTq",
        "MiddleTop": "\u091A",
        "MiddleMiddle": "\u091C",
        "MiddleBottom": "\u091E",
        "MiddleNail": "LTw",
        "RingTop": "\u091F",
        "RingMiddle": "\u0921",
        "RingBottom": "\u0923",
        "RingNail": "LTe",
        "LittleTop": "\u0924",
        "LittleMiddle": "\u0926",
        "LittleBottom": "\u0928",
        "LittleNail": "LTt"
    },
    "RHTap": {
        "IndexTop": "RT1",
        "IndexMiddle": "RT2",
        "IndexBottom": "RT3",
        "IndexNail": "RTq",
        "MiddleTop": "\u0937",
        "MiddleMiddle": "\u0939",
        "MiddleBottom": "RT6",
        "MiddleNail": "RTw",
        "RingTop": "\u092F",
        "RingMiddle": "\u0932",
        "RingBottom": "\u0936",
        "RingNail": "RTe",
        "LittleTop": "\u092A",
        "LittleMiddle": "\u092C",
        "LittleBottom": "\u092E",
        "LittleNail": "RTt"
    },
    "LHLongPress": {
        "IndexTop": "\u0916",
        "IndexMiddle": "\u0918",
        "IndexBottom": "LL3",
        "IndexNail": "LLq",
        "MiddleTop": "\u091B",
        "MiddleMiddle": "\u091D",
        "MiddleBottom": "LL6",
        "MiddleNail": "LLw",
        "RingTop": "\u0920",
        "RingMiddle": "\u0922",
        "RingBottom": "LL9",
        "RingNail": "LLe",
        "LittleTop": "\u0925",
        "LittleMiddle": "\u0927",
        "LittleBottom": "LL#",
        "LittleNail": "LLt"
    },
    "RHLongPress": {
        "IndexTop": "RL1",
        "IndexMiddle": "RL2",
        "IndexBottom": "RL3",
        "IndexNail": "RLq",
        "MiddleTop": "\u0938",
        "MiddleMiddle": "?",
        "MiddleBottom": "RL6",
        "MiddleNail": "RLw",
        "RingTop": "\u0930",
        "RingMiddle": "\u0935",
        "RingBottom": "RL9",
        "RingNail": "RLe",
        "LittleTop": "\u092B",
        "LittleMiddle": "\u092D",
        "LittleBottom": "RL#",
        "LittleNail": "RLt"
    },
    "LHFold": {
        "IndexTop": "\u094C",
        "IndexMiddle": "\u0902",
        "IndexBottom": "\u0903",
        "IndexNail": "LFq",
        "MiddleTop": "\u0947",
        "MiddleMiddle": "\u0948",
        "MiddleBottom": "\u094B",
        "MiddleNail": "LFw",
        "RingTop": "\u0940",
        "RingMiddle": "\u0941",
        "RingBottom": "\u0942",
        "RingNail": "LFe",
        "LittleTop": "LF*",
        "LittleMiddle": "\u093E",
        "LittleBottom": "\u093F",
        "LittleNail": "LFt"
    },
    "RHFold": {
        "IndexTop": "\u0914",
        "IndexMiddle": "RF2",
        "IndexBottom": "RF3",
        "IndexNail": "RFq",
        "MiddleTop": "\u090F",
        "MiddleMiddle": "\u0910",
        "MiddleBottom": "\u0913",
        "MiddleNail": "RFw",
        "RingTop": "\u0908",
        "RingMiddle": "\u0909",
        "RingBottom": "\u090A",
        "RingNail": "RFe",
        "LittleTop": "\u0905",
        "LittleMiddle": "\u0906",
        "LittleBottom": "\u0907",
        "LittleNail": "RFt"
    }
}

# %%
Keys_Dict = {
    "LHTap": {
        "IndexTop": "LT1",
        "IndexMiddle": "LT2",
        "IndexBottom": "LT3",
        "IndexNail": "LTq",
        "MiddleTop": "LT4",
        "MiddleMiddle": "LT5",
        "MiddleBottom": "LT6",
        "MiddleNail": "LTw",
        "RingTop": "LT7",
        "RingMiddle": "LT8",
        "RingBottom": "LT9",
        "RingNail": "LTe",
        "LittleTop": "LT*",
        "LittleMiddle": "LT0",
        "LittleBottom": "LT#",
        "LittleNail": "LTt"
    },
    "RHTap": {
        "IndexTop": "RT1",
        "IndexMiddle": "RT2",
        "IndexBottom": "RT3",
        "IndexNail": "RTq",
        "MiddleTop": "RT4",
        "MiddleMiddle": "RT5",
        "MiddleBottom": "RT6",
        "MiddleNail": "RTw",
        "RingTop": "RT7",
        "RingMiddle": "RT8",
        "RingBottom": "RT9",
        "RingNail": "RTe",
        "LittleTop": "RT*",
        "LittleMiddle": "RT0",
        "LittleBottom": "RT#",
        "LittleNail": "RTt"
    },
    "LHLongPress": {
        "IndexTop": "LL1",
        "IndexMiddle": "LL2",
        "IndexBottom": "LL3",
        "IndexNail": "LLq",
        "MiddleTop": "LL4",
        "MiddleMiddle": "LL5",
        "MiddleBottom": "LL6",
        "MiddleNail": "LLw",
        "RingTop": "LL7",
        "RingMiddle": "LL8",
        "RingBottom": "LL9",
        "RingNail": "LLe",
        "LittleTop": "LL*",
        "LittleMiddle": "LL0",
        "LittleBottom": "LL#",
        "LittleNail": "LLt"
    },
    "RHLongPress": {
        "IndexTop": "RL1",
        "IndexMiddle": "RL2",
        "IndexBottom": "RL3",
        "IndexNail": "RLq",
        "MiddleTop": "RL4",
        "MiddleMiddle": "RL5",
        "MiddleBottom": "RL6",
        "MiddleNail": "RLw",
        "RingTop": "RL7",
        "RingMiddle": "RL8",
        "RingBottom": "RL9",
        "RingNail": "RLe",
        "LittleTop": "RL*",
        "LittleMiddle": "RL0",
        "LittleBottom": "RL#",
        "LittleNail": "RLt"
    },
    "LHFold": {
        "IndexTop": "LF1",
        "IndexMiddle": "LF2",
        "IndexBottom": "LF3",
        "IndexNail": "LFq",
        "MiddleTop": "LF4",
        "MiddleMiddle": "LF5",
        "MiddleBottom": "LF6",
        "MiddleNail": "LFw",
        "RingTop": "LF7",
        "RingMiddle": "LF8",
        "RingBottom": "LF9",
        "RingNail": "LFe",
        "LittleTop": "LF*",
        "LittleMiddle": "LF0",
        "LittleBottom": "LF#",
        "LittleNail": "LFt"
    },
    "RHFold": {
        "IndexTop": "RF1",
        "IndexMiddle": "RF2",
        "IndexBottom": "RF3",
        "IndexNail": "RFq",
        "MiddleTop": "RF4",
        "MiddleMiddle": "RF5",
        "MiddleBottom": "RF6",
        "MiddleNail": "RFw",
        "RingTop": "RF7",
        "RingMiddle": "RF8",
        "RingBottom": "RF9",
        "RingNail": "RFe",
        "LittleTop": "RF*",
        "LittleMiddle": "RF0",
        "LittleBottom": "RF#",
        "LittleNail": "RFt"
    }
}

# %%
Keys = pd.DataFrame(Keys_Dict,
                    index=["IndexTop", "IndexMiddle", "IndexBottom", "IndexNail",
                           "MiddleTop", "MiddleMiddle", "MiddleBottom", "MiddleNail",
                           "RingTop", "RingMiddle", "RingBottom", "RingNail",
                           "LittleTop", "LittleMiddle", "LittleBottom", "LittleNail"])

# %%
Hindi = pd.DataFrame(Hindi_Dict,
                     index=["IndexTop", "IndexMiddle", "IndexBottom", "IndexNail",
                            "MiddleTop", "MiddleMiddle", "MiddleBottom", "MiddleNail",
                            "RingTop", "RingMiddle", "RingBottom", "RingNail",
                            "LittleTop", "LittleMiddle", "LittleBottom", "LittleNail"])

# %%
PredtoKey = ["1", "2", "3", "q", "4", "5", "6", "w", "7", "8", "9", "e", "*", "0", "#", "t", "f", "-1"]


# %%
def KeyPrediction(cord):
    try:
        f_arr = fold(cord)
        if 1 in f_arr:
            if f_arr == [1, 1, 1, 1]:
                return -2, 4
            else:
                for i, num in enumerate(f_arr):
                    if num == 1:
                        return (4 * i + 3), 4
        dist = np.ndarray((4, 4), dtype=float)
        min = [75, -4]
        key = [0, 0, 1, 2, 4, 4, 5, 6, 8, 8, 9, 10, 12, 12, 13, 14]
        lst = [8, 7, 6, 5, 12, 11, 10, 9, 16, 15, 14, 13, 20, 19, 18, 17]
        for i, id in enumerate(lst):
            dist[i % 4, int(i / 4)] = math.sqrt(
                (cord[id][0] - cord[4][0]) ** 2 + (cord[id][1] - cord[4][1]) ** 2 + (cord[id][2] - cord[4][2]) ** 2 / 3)
            if min[0] > dist[i % 4, int(i / 4)]:
                min[0] = dist[i % 4, int(i / 4)]
                min[1] = i
        if min[1] == -4:
            return -1, 4
        return key[min[1]], lst[min[1]]
    except IndexError:
        return -1, 4


# %%
rest_state = [["-1", "-1"], ["f", "-1"], ["-1", "f"], ["f", "f"]]


def Keypress(ptime, ctime, pkey, ckey, keypad, keypad_hindi):
    timediff = ctime - ptime
    if pkey in rest_state and ckey in rest_state:
        return ckey, 0, keypad, keypad_hindi
    elif pkey in rest_state and ckey not in rest_state:
        return ckey, ctime, keypad, keypad_hindi
    elif pkey == ckey and timediff < 1:
        return pkey, ptime, keypad, keypad_hindi
    elif pkey == ckey:
        if ckey[1] == "-1":
            keypad = keypad + Keys["LHLongPress"][PredtoKey.index(ckey[0])]
            keypad_hindi = keypad_hindi + Hindi["LHLongPress"][PredtoKey.index(ckey[0])]
            try:
                k.press(Hindi["LHLongPress"][PredtoKey.index(ckey[0])])
            except ValueError:
                pass
            return ["r", "-1"], 0, keypad, keypad_hindi
        elif ckey[0] == "-1":
            keypad = keypad + Keys["RHLongPress"][PredtoKey.index(ckey[1])]
            keypad_hindi = keypad_hindi + Hindi["RHLongPress"][PredtoKey.index(ckey[1])]
            try:
                k.press(Hindi["RHLongPress"][PredtoKey.index(ckey[1])])
            except ValueError:
                pass
            return ["-1", "r"], 0, keypad, keypad_hindi
        return ["-1", "-1"], 0, keypad, keypad_hindi
    elif pkey != ckey and ("r" not in pkey) and pkey != ["-1", "-1"] and timediff > 0.3:
        if pkey[1] == "-1":
            keypad = keypad + Keys["LHTap"][PredtoKey.index(pkey[0])]
            keypad_hindi = keypad_hindi + Hindi["LHTap"][PredtoKey.index(pkey[0])]
            try:
                k.press(Hindi["LHTap"][PredtoKey.index(pkey[0])])
            except ValueError:
                pass
            return ["r", "-1"], 0, keypad, keypad_hindi
        elif pkey[1] == "f":
            keypad = keypad + Keys["LHFold"][PredtoKey.index(pkey[0])]
            keypad_hindi = keypad_hindi + Hindi["LHFold"][PredtoKey.index(pkey[0])]
            try:
                k.press(Hindi["LHFold"][PredtoKey.index(pkey[0])])
            except ValueError:
                pass
            return ["r", "-1"], 0, keypad, keypad_hindi
        elif pkey[0] == "-1":
            keypad = keypad + Keys["RHTap"][PredtoKey.index(pkey[1])]
            keypad_hindi = keypad_hindi + Hindi["RHTap"][PredtoKey.index(pkey[1])]
            try:
                k.press(Hindi["RHTap"][PredtoKey.index(pkey[1])])
            except ValueError:
                pass
            return ["-1", "r"], 0, keypad, keypad_hindi
        elif pkey[0] == "f":
            keypad = keypad + Keys["RHFold"][PredtoKey.index(pkey[1])]
            keypad_hindi = keypad_hindi + Hindi["RHFold"][PredtoKey.index(pkey[1])]
            try:
                k.press(Hindi["RHFold"][PredtoKey.index(pkey[1])])
            except ValueError:
                pass
            return ["-1", "r"], 0, keypad, keypad_hindi
        return ["-1", "-1"], 0, keypad, keypad_hindi
    elif "r" in pkey and ckey in rest_state:
        return ckey, 0, keypad, keypad_hindi
    elif pkey != ckey and "r" not in pkey:
        return ["-1", "-1"], 0, keypad, keypad_hindi
    else:
        return pkey, 0, keypad, keypad_hindi


def fold(cord):
    X = np.append(np.array(cord[5::4])[:, 0].reshape(4, 1), np.ones((4, 1)), axis=1)
    Y = np.array(cord[5::4])[:, 1]
    [m, c] = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    f = [0, 0, 0, 0]
    for i, lno in enumerate([8, 12, 16, 20]):
        if cord[lno][1] - m * cord[lno][0] - c > 0:
            f[i] = 1
    return f


cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      min_detection_confidence=0.8)
ptime = 0
pkey = ["r", "r"]
keypad = ""
dist = 0
handkey = {'Left': 1, 'Right': 0}
keypad = ""
nkeypad = ""
keypad_hindi = ""

while cv2.waitKey(1) != 113:
    success, img = cap.read()
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    # flipRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # imgRGB = cv2.flip(flipRGB, 1)
    # results = hands.process(imgRGB)
    h, w, c = img.shape
    tempi = [np.zeros((500, 500, 3), dtype=np.uint8), np.zeros((500, 500, 3), dtype=np.uint8)]
    ctime = time.time()
    Pred = ["-1", "-1"]

    if results.multi_hand_landmarks:
        for no, hand in enumerate(results.multi_hand_landmarks):
            hand.landmark[4].x = hand.landmark[3].x + (hand.landmark[4].x - hand.landmark[3].x) * 1.1
            cord = []
            y_max = 0
            x_max = 0
            x_min = w
            y_min = h
            for i in hand.landmark:
                x, y = int(i.x * w), int(i.y * h)
                if x > x_max:
                    x_max = x
                if x < x_min:
                    x_min = x
                if y > y_max:
                    y_max = y
                if y < y_min:
                    y_min = y
            hno = handkey[results.multi_handedness[no].classification[0].label]
            for id, lm in enumerate(hand.landmark):
                cord.append([int((250 - (x_max - x_min) / 2) + lm.x * w - x_min),
                             int((250 - (y_max - y_min) / 2) + lm.y * h - y_min), lm.z * w])
                cv2.circle(tempi[hno], (cord[id][0], cord[id][1]), 3,
                           tuple(255 * i for i in colorsys.hsv_to_rgb(abs(lm.z * 2), 0.8, 0.8)), cv2.FILLED)
            KeyPred, pt = KeyPrediction(cord)
            Pred[hno] = PredtoKey[KeyPred]
        ckey = Pred
        pkey, ptime, keypad, keypad_hindi = Keypress(ptime, ctime, pkey, ckey, keypad, keypad_hindi)
    final = np.concatenate(tempi, axis=1)
    cv2.putText(final, str(Pred[0]) + str(Pred[1]), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.putText(final, str(keypad[-20:]), (10, 110), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow("Image", final)
cv2.destroyAllWindows()

