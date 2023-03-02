import numpy as np

# (TP, TN, FP1, FP2, FN)

metrics = np.array(
    [[186, 0, 107, 0, 46],
    [99, 0, 41, 0, 34],
    [195, 56, 145, 34, 53],
    [159, 78, 217, 24, 86],
    [165, 46, 179, 16, 101],
    [165, 46, 179, 16, 101],
    [482, 42, 270, 24, 172],
    [86, 33, 71, 2, 39],
    [359, 74, 269, 13, 125],
    [282, 61, 225, 42, 104],
    [94, 0, 92, 0, 45],
    [464, 76, 273, 10, 110],
    [329, 55, 190, 5, 114],
    [408, 120, 240, 14, 79],
    [106, 26, 118, 19, 16],
    [277, 15, 237, 18, 113],
    [370, 61, 344, 64, 94],
    [147, 9, 170, 4, 66],
    [165, 40, 73, 17, 62]]
)
total1 = np.array([0, 0, 0, 0, 0])  # Include FP1 and FP2
total2 = np.array([0, 0, 0, 0])  # Mix FP1 and FP2

for metric in metrics:
    total1 += metric

total2[0] = total1[0]
total2[1] = total1[1]
total2[2] = total1[2] + total1[3]
total2[3] = total1[4]

recall = total2[0] / (total2[0] + total2[3])  # TP/(TP+FN)
specificity = total2[1] / (total2[1] + total2[2])  # TN/(TN+FP)
precision = total2[0] / (total2[0] + total2[2])  # TP/(TP+FP)
accuracy = (total2[0] + total2[1]) / (total2[0] + total2[1] + total2[2] + total2[3])  # (TN+TP)/(TN+TP+FN+FP)
tpr = total2[0] / (total2[0] + total2[3])  # TP/(TP+FN)
tnr = total2[2] / (total2[2] + total2[1])  # FP/(FP+TN)
f1 = (precision*recall) / (precision+recall)

print(total1)
print(total2)
print("recall/sensitivity: ", recall*100)
print("specificity: ", specificity*100)
print("precision: ", precision*100)
print("accuracy: ", accuracy*100)
print("f1: ", f1*100)
