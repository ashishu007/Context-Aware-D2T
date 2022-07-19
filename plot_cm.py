import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

CLF = 'rf'
FTR = 'emb'
DOWN = True
THEME = 'standing'

test_y = np.array(pd.read_csv(f'./data/{THEME}/test_text.csv', names=['text', 'label'])['label'].to_list())
pred_y = np.load(f'./preds/streak/{CLF}-{FTR}-down_{"yes" if DOWN else "no"}.npy')
cm = confusion_matrix(test_y, pred_y)
print(cm)
df = pd.DataFrame(cm, index=['yes', 'no'], columns=['yes', 'no'])
plt.figure(figsize=(10,7))
plt.rcParams['font.size'] = 18
sns.heatmap(df, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
tick_marks = [0.5, 1.5] #np.arange(len(set(test_y)))
plt.xticks(tick_marks, ['No', 'Yes'], rotation=0)
plt.yticks(tick_marks, ['No', 'Yes']) #np.unique(test_y))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()
