from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np


def Evaluation(model, xte, yte):
    test_scores = model.evaluate(xte, yte, verbose=0)
    print(f'Test loss: {test_scores[0]}')
    print(f'Test accuracy: {test_scores[1]}')

    y_pred = model.predict(xte, batch_size=16)
    y_pred_classes = np.argmax(y_pred, axis=1)

    y_test = np.argmax(yte, axis=1)

    f1 = f1_score(y_test, y_pred_classes,  average='macro')
    recall = recall_score(y_test, y_pred_classes,  average='macro')
    precision = precision_score(y_test, y_pred_classes, average='macro')

    # In kết quả
    print("F1-score:", f1)
    print("Recall:", recall)
    print("Precision:", precision)

    cm = confusion_matrix(y_test, y_pred_classes, normalize='true')
    print(y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

    # Set the number of ticks for both axes
    n_classes = len(labels)
    disp.plot()

    plt.show()
