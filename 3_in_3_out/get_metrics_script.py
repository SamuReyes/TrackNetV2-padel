import numpy as np

# (TP, TN, FP1, FP2, FN)

metrics = np.array(
    [
        [330, 6, 51, 2, 49],
[174, 11, 37, 5, 46],
[164, 14, 12, 1, 25],
[431, 104, 195, 3, 56],
[175, 38, 98, 3, 31],
[171, 57, 110, 9, 52]
    ]
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
