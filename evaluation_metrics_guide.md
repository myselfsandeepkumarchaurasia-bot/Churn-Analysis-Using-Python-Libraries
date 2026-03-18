# Evaluation Techniques for Regression and Classification

This guide explains evaluation metrics from beginner to advanced level, with intuition, formulas, numerical examples, Python code, comparison tables, real-life examples, interview questions, common mistakes, quick revision notes, a cheat sheet, and practice questions.

---

## 1) Regression Evaluation Metrics

Regression is used when the target is a continuous value, such as house price, salary, sales, demand, or temperature.

We'll reuse this sample data in several examples:

- `y_true = [100, 120, 130, 150]`
- `y_pred = [110, 115, 128, 140]`

Errors:

- `100 - 110 = -10`
- `120 - 115 = 5`
- `130 - 128 = 2`
- `150 - 140 = 10`

### 1.1 MSE — Mean Squared Error

#### Simple Hinglish intuition
MSE batata hai ki model average mein kitna squared error kar raha hai. Pehle actual aur predicted ka difference nikaalte hain, phir square karte hain, aur phir average lete hain. Square karne ki wajah se bade errors ko zyada punishment milti hai.

#### Formula

\[
MSE = \frac{1}{n} \sum_{i=1}^{n}(y_i - \hat{y}_i)^2
\]

#### Terms
- `n`: number of observations
- `y_i`: actual value
- `\hat{y}_i`: predicted value
- `(y_i - \hat{y}_i)`: prediction error

#### Step-by-step example
Squared errors:
- `(-10)^2 = 100`
- `(5)^2 = 25`
- `(2)^2 = 4`
- `(10)^2 = 100`

Sum = `100 + 25 + 4 + 100 = 229`

\[
MSE = \frac{229}{4} = 57.25
\]

#### sklearn code
```python
from sklearn.metrics import mean_squared_error

y_true = [100, 120, 130, 150]
y_pred = [110, 115, 128, 140]

mse = mean_squared_error(y_true, y_pred)
print(mse)
```

#### When to use
- When large errors should be penalized strongly
- When discussing optimization/loss functions

#### When not to use
- When business users need easy interpretation in original units
- When outliers dominate the data

---

### 1.2 RMSE — Root Mean Squared Error

#### Simple Hinglish intuition
RMSE, MSE ka square root hota hai. Isliye result target variable ki same unit mein aata hai. Agar house price predict kar rahe ho, to RMSE bhi price units mein hoga.

#### Formula

\[
RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
\]

#### Step-by-step example
We already found MSE = `57.25`

\[
RMSE = \sqrt{57.25} \approx 7.57
\]

#### sklearn code
```python
from sklearn.metrics import mean_squared_error
import numpy as np

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
print(rmse)
```

#### When to use
- When you want error in original units
- When large mistakes matter

#### When not to use
- When outliers are too extreme
- When you need percentage-based reporting

---

### 1.3 MAE — Mean Absolute Error

#### Simple Hinglish intuition
MAE batata hai ki model average mein kitna absolute error kar raha hai. Isme square nahi hota, sirf absolute difference liya jata hai, isliye outliers ka effect MSE/RMSE se kam hota hai.

#### Formula

\[
MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|
\]

#### Step-by-step example
Absolute errors:
- `|-10| = 10`
- `|5| = 5`
- `|2| = 2`
- `|10| = 10`

Sum = `27`

\[
MAE = \frac{27}{4} = 6.75
\]

#### sklearn code
```python
from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_true, y_pred)
print(mae)
```

#### When to use
- When interpretability matters
- When you want average error directly
- When outliers should not dominate too much

#### When not to use
- When large errors must be punished heavily

---

### 1.4 R² Score — Coefficient of Determination

#### Simple Hinglish intuition
R² batata hai ki model ne target variable ki variability ka kitna hissa explain kiya. `1` means perfect fit, `0` means baseline mean predictor jaisa performance, and negative means baseline se bhi worse.

#### Formula

\[
R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
\]

#### Terms
- `\bar{y}`: mean of actual values
- numerator: unexplained variation (RSS)
- denominator: total variation (TSS)

#### Step-by-step example
Mean of actuals:

\[
\bar{y} = \frac{100 + 120 + 130 + 150}{4} = 125
\]

RSS = `229`

TSS:
- `(100 - 125)^2 = 625`
- `(120 - 125)^2 = 25`
- `(130 - 125)^2 = 25`
- `(150 - 125)^2 = 625`

TSS = `1300`

\[
R^2 = 1 - \frac{229}{1300} \approx 0.824
\]

#### sklearn code
```python
from sklearn.metrics import r2_score

r2 = r2_score(y_true, y_pred)
print(r2)
```

#### When to use
- To explain variance captured by the model
- To compare regression models broadly

#### When not to use
- As the only metric
- When you care more about actual business error size

---

### 1.5 Adjusted R²

#### Simple Hinglish intuition
Regular R² extra features add karne par aksar badh jata hai, chahe feature useful ho ya na ho. Adjusted R² useless features ko penalize karta hai.

#### Formula

\[
Adjusted\ R^2 = 1 - \left( \frac{(1-R^2)(n-1)}{n-p-1} \right)
\]

#### Terms
- `R²`: regular coefficient of determination
- `n`: number of observations
- `p`: number of predictors/features

#### Step-by-step example
Assume:
- `R² = 0.824`
- `n = 10`
- `p = 2`

\[
Adjusted\ R^2 = 1 - \left( \frac{(1 - 0.824)(10 - 1)}{10 - 2 - 1} \right)
\]

\[
= 1 - \left( \frac{0.176 \times 9}{7} \right)
= 1 - 0.2263
= 0.7737
\]

#### sklearn/manual code
```python
from sklearn.metrics import r2_score

r2 = r2_score(y_true, y_pred)
n = 4
p = 2
adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
print(adjusted_r2)
```

#### When to use
- Multiple regression
- Feature selection comparison

#### When not to use
- When there is only one feature
- When practical error metrics matter more

---

### 1.6 MAPE — Mean Absolute Percentage Error

#### Simple Hinglish intuition
MAPE batata hai ki model average mein kitne percent ka error kar raha hai. Business dashboards mein ye kaafi popular hota hai.

#### Formula

\[
MAPE = \frac{100}{n}\sum_{i=1}^{n}\left|\frac{y_i - \hat{y}_i}{y_i}\right|
\]

#### Step-by-step example
- `|100 - 110| / 100 = 0.10 = 10%`
- `|120 - 115| / 120 = 0.0417 = 4.17%`
- `|130 - 128| / 130 = 0.0154 = 1.54%`
- `|150 - 140| / 150 = 0.0667 = 6.67%`

Average percentage error:

\[
MAPE = \frac{10 + 4.17 + 1.54 + 6.67}{4} \approx 5.60\%
\]

#### sklearn code
```python
from sklearn.metrics import mean_absolute_percentage_error

mape = mean_absolute_percentage_error(y_true, y_pred) * 100
print(mape)
```

#### When to use
- Forecasting and business reporting
- When percentage interpretation is useful

#### When not to use
- When actual values can be zero or near zero

---

## 2) Regression Real-Life Examples

### House Price Prediction
- MAE tells average price error directly
- RMSE punishes expensive large mistakes more
- R² shows how much price variation is explained
- MAPE gives percentage-style error reporting

### Salary Prediction
If a few salary predictions are wildly wrong, RMSE exposes that strongly. If you want a directly understandable average miss, MAE is often better.

---

## 3) Classification Evaluation Metrics

Classification is used when the output is a category, like spam/not spam, disease/no disease, fraud/not fraud.

We'll use this confusion matrix example:

- TP = 40
- TN = 45
- FP = 5
- FN = 10
- Total = 100

### 3.1 Confusion Matrix

#### Simple Hinglish intuition
Confusion matrix ek table hota hai jo dikhata hai model ne kaunse predictions sahi kiye aur kaunse galat.

#### Terms
- **TP (True Positive):** actual positive, predicted positive
- **TN (True Negative):** actual negative, predicted negative
- **FP (False Positive):** actual negative, predicted positive
- **FN (False Negative):** actual positive, predicted negative

#### Real-life intuition
- Spam detection: FP means a real email went to spam
- Disease detection: FN means a sick patient was missed

#### sklearn code
```python
from sklearn.metrics import confusion_matrix

y_true = [1, 1, 1, 1, 0, 0, 0, 0, 1, 0]
y_pred = [1, 1, 0, 1, 0, 0, 1, 0, 1, 0]

cm = confusion_matrix(y_true, y_pred)
print(cm)
```

#### Visualization code
```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Pred 0', 'Pred 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
```

---

### 3.2 Accuracy

#### Simple Hinglish intuition
Accuracy batata hai total predictions mein se kitni correct thi.

#### Formula

\[
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
\]

#### Step-by-step example

\[
Accuracy = \frac{40 + 45}{40 + 45 + 5 + 10} = \frac{85}{100} = 0.85 = 85\%
\]

#### sklearn code
```python
from sklearn.metrics import accuracy_score

acc = accuracy_score(y_true, y_pred)
print(acc)
```

#### When to use
- Balanced classes
- Simple overview metric

#### When not to use
- Imbalanced datasets
- Medical/fraud use cases where one error type is dangerous

---

### 3.3 Precision

#### Simple Hinglish intuition
Precision ka answer: model ne jitni baar positive bola, unmein se kitni baar woh actually positive tha?

#### Formula

\[
Precision = \frac{TP}{TP + FP}
\]

#### Step-by-step example

\[
Precision = \frac{40}{40 + 5} = \frac{40}{45} \approx 0.8889 = 88.89\%
\]

#### Real-life example
Spam detection mein high precision ka matlab jab model spam bole, to mostly woh sach mein spam ho.

#### sklearn code
```python
from sklearn.metrics import precision_score

precision = precision_score(y_true, y_pred)
print(precision)
```

#### When to use
- False positives are costly
- Spam filtering, legal alerts, risk flags

#### When not to use
- When missing positives is more dangerous than false alarms

---

### 3.4 Recall

#### Simple Hinglish intuition
Recall ka answer: actual positive cases mein se model ne kitne pakde?

#### Formula

\[
Recall = \frac{TP}{TP + FN}
\]

#### Step-by-step example

\[
Recall = \frac{40}{40 + 10} = \frac{40}{50} = 0.80 = 80\%
\]

#### Real-life example
Disease screening mein recall high hona chahiye, kyunki sick cases miss karna dangerous hota hai.

#### sklearn code
```python
from sklearn.metrics import recall_score

recall = recall_score(y_true, y_pred)
print(recall)
```

#### When to use
- When false negatives are costly
- Medical screening, fraud detection, security monitoring

#### When not to use
- When too many false alarms are unacceptable

---

### 3.5 F1-Score

#### Simple Hinglish intuition
F1-score precision aur recall ka balanced summary hai. Ye tab useful hai jab dono important hon.

#### Formula

\[
F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}
\]

Alternative:

\[
F1 = \frac{2TP}{2TP + FP + FN}
\]

#### Step-by-step example
Using precision = `0.8889` and recall = `0.80`:

\[
F1 \approx 2 \cdot \frac{0.8889 \cdot 0.80}{0.8889 + 0.80} \approx 0.842 = 84.2\%
\]

#### sklearn code
```python
from sklearn.metrics import f1_score

f1 = f1_score(y_true, y_pred)
print(f1)
```

#### When to use
- When both precision and recall matter
- When classes are imbalanced and you need one summary metric

#### When not to use
- When TN matters a lot and should not be ignored

---

### 3.6 ROC Curve and AUC

#### Simple Hinglish intuition
Many classifiers probability dete hain, sirf final class nahi. Threshold change karne se TP/FP trade-off change hota hai. ROC curve different thresholds par TPR vs FPR dikhati hai.

#### Formula pieces

\[
TPR = \frac{TP}{TP + FN}
\]

\[
FPR = \frac{FP}{FP + TN}
\]

ROC plots `FPR` on x-axis and `TPR` on y-axis.

AUC means area under the ROC curve.
- `1.0` = perfect
- `0.5` = random guessing

#### Small threshold example
Suppose predicted probabilities are:
- positives: `0.90`, `0.80`, `0.60`
- negatives: `0.70`, `0.40`, `0.30`

At threshold `0.5`:
- TP = 3
- FP = 1
- FN = 0
- TN = 2

So:

\[
TPR = 3/(3+0) = 1
\]

\[
FPR = 1/(1+2) = 0.333
\]

Different thresholds give different points.

#### sklearn code
```python
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

y_true = [1, 1, 0, 1, 0, 0]
y_scores = [0.90, 0.80, 0.70, 0.60, 0.40, 0.30]

fpr, tpr, thresholds = roc_curve(y_true, y_scores)
auc = roc_auc_score(y_true, y_scores)

print(auc)
plt.plot(fpr, tpr, marker='o', label=f'AUC = {auc:.2f}')
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

#### When to use
- Comparing classifiers by ranking ability
- Evaluating across many thresholds

#### When not to use
- When the dataset is highly imbalanced and PR metrics are more informative

---

### 3.7 Log Loss

#### Simple Hinglish intuition
Log loss evaluate karta hai ki model ne sirf sahi class predict ki ya nahi, balki kitni confidence se predict ki. Wrong and confident prediction ko heavy punishment milti hai.

#### Formula

\[
LogLoss = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i\log(p_i) + (1-y_i)\log(1-p_i)\right]
\]

#### Step-by-step example
Suppose:

| Actual | Predicted Prob(1) |
|---|---:|
| 1 | 0.9 |
| 0 | 0.2 |
| 1 | 0.8 |
| 0 | 0.1 |

Per-row losses are approximately:
- `-log(0.9) = 0.105`
- `-log(0.8) = 0.223`
- `-log(0.8) = 0.223`
- `-log(0.9) = 0.105`

Average:

\[
LogLoss = (0.105 + 0.223 + 0.223 + 0.105)/4 = 0.164
\]

#### sklearn code
```python
from sklearn.metrics import log_loss

y_true = [1, 0, 1, 0]
y_prob = [0.9, 0.2, 0.8, 0.1]

loss = log_loss(y_true, y_prob)
print(loss)
```

#### When to use
- When probability quality matters
- When calibration/confidence matters

#### When not to use
- When business users only care about final labels

---

## 4) Classification Real-Life Examples

### Spam Detection
- Precision important: genuine mail should not go to spam
- Recall also useful: spam should not slip into inbox
- F1 useful if you want balance

### Disease Prediction
- Recall critical: sick patients should not be missed
- Confusion matrix is essential to inspect FN specifically
- Log loss useful if probabilities drive medical review

---

## 5) Comparison Tables

### 5.1 Regression Metrics Comparison

| Metric | Better Direction | Outlier Sensitivity | Easy Interpretation | Best For |
|---|---|---|---|---|
| MSE | Lower | High | Medium | Penalizing large errors |
| RMSE | Lower | High | High | Error in original units |
| MAE | Lower | Lower | Very High | Simple average error |
| R² | Higher | Indirect | High | Variance explained |
| Adjusted R² | Higher | Indirect | Medium | Multi-feature model comparison |
| MAPE | Lower | Problem near zero actuals | High | Percentage error |

### 5.2 Classification Metrics Comparison

| Metric | Focus | Best Use | Main Limitation |
|---|---|---|---|
| Accuracy | Overall correctness | Balanced classes | Misleading on imbalance |
| Precision | Positive prediction quality | FP costly | Ignores missed positives |
| Recall | Capture of real positives | FN costly | Can increase FP |
| F1-score | Balance of precision and recall | Both matter | Ignores TN |
| Confusion Matrix | Full error breakdown | Deep diagnosis | Not a single score |
| ROC-AUC | Ranking across thresholds | Compare classifiers | Less informative under severe imbalance |
| Log Loss | Probability confidence | Probabilistic models | Harder to explain |

### 5.3 Which Metric When?

#### Regression
- Large mistakes matter most → RMSE / MSE
- Direct average error wanted → MAE
- Percentage reporting needed → MAPE
- Explained variance needed → R²
- Comparing models with different predictor counts → Adjusted R²

#### Classification
- Balanced data → Accuracy
- FP costly → Precision
- FN costly → Recall
- Both FP and FN matter → F1-score
- Threshold-free ranking → ROC-AUC
- Probability quality matters → Log Loss

---

## 6) Important Interview Questions

### Concept-based
1. What is the difference between MAE and RMSE?
2. Can R² be negative? Why?
3. Why do we need Adjusted R²?
4. Why is accuracy misleading on imbalanced data?
5. Explain precision vs recall in one sentence each.
6. Why does F1-score use harmonic mean?
7. Why can log loss be more informative than accuracy?

### Scenario-based
1. For cancer screening, which metric matters most and why?
2. For spam detection, when would you prioritize precision over recall?
3. In house price prediction, when would you report RMSE instead of MAE?
4. In demand forecasting dashboards, why is MAPE popular?
5. How would you compare two regression models with different numbers of features?

### Trick questions with short answers
1. Does high accuracy always mean a good model? → No.
2. Is higher recall always better? → Not always, because FP may rise.
3. Can precision be 1 while recall is low? → Yes.
4. Can recall be 1 while precision is low? → Yes.
5. Does low log loss guarantee highest accuracy? → No.

---

## 7) Common Mistakes and Misconceptions

### Regression mistakes
- Looking only at R² and ignoring MAE/RMSE
- Using MAPE when actual values are zero
- Treating MAE and RMSE as identical
- Ignoring Adjusted R² in multi-feature comparison

### Classification mistakes
- Using accuracy on imbalanced datasets
- Confusing precision and recall
- Assuming F1 is always the best metric
- Ignoring threshold tuning
- Using ROC-AUC blindly for rare-event problems

---

## 8) Quick Revision Notes

### Regression
- MSE: squared average error
- RMSE: same-unit version of MSE
- MAE: average absolute error
- R²: explained variance
- Adjusted R²: penalized R² for extra features
- MAPE: percentage error, avoid zeros

### Classification
- Accuracy: overall correct predictions
- Precision: when model says positive, how often is it right?
- Recall: among real positives, how many did it catch?
- F1: balance of precision and recall
- Confusion Matrix: TP, TN, FP, FN breakdown
- ROC-AUC: ranking quality across thresholds
- Log Loss: confidence/probability quality

---

## 9) Cheat Sheet

### Regression Cheat Sheet
- `MSE = average((y - y_hat)^2)`
- `RMSE = sqrt(MSE)`
- `MAE = average(|y - y_hat|)`
- `R² = 1 - RSS/TSS`
- `Adjusted R² = 1 - ((1-R²)(n-1)/(n-p-1))`
- `MAPE = average(|(y - y_hat)/y|) * 100`

### Classification Cheat Sheet
- `Accuracy = (TP + TN) / Total`
- `Precision = TP / (TP + FP)`
- `Recall = TP / (TP + FN)`
- `F1 = 2PR / (P + R)`
- `TPR = TP / (TP + FN)`
- `FPR = FP / (FP + TN)`
- `LogLoss = -avg(y log p + (1-y) log(1-p))`

---

## 10) Practice Questions

### Regression practice
1. For `y_true = [50, 60, 70]` and `y_pred = [55, 58, 68]`, calculate MAE, MSE, and RMSE.
2. Can a model have high R² and still poor MAE? Explain with intuition.
3. Why is MAPE unsuitable when actual values include zero?
4. In salary prediction, when would you prefer RMSE over MAE?

### Classification practice
1. If TP = 30, TN = 50, FP = 10, FN = 10, calculate Accuracy, Precision, Recall, and F1-score.
2. In fraud detection, which is worse: FP or FN? Why?
3. Why can 99% accuracy still mean a bad model?
4. How is ROC-AUC different from accuracy?

---

## 11) Compact sklearn Reference

### Regression metrics
```python
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
)
import numpy as np

y_true = [100, 120, 130, 150]
y_pred = [110, 115, 128, 140]

mse = mean_squared_error(y_true, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)
mape = mean_absolute_percentage_error(y_true, y_pred) * 100

print('MSE :', mse)
print('RMSE:', rmse)
print('MAE :', mae)
print('R2  :', r2)
print('MAPE:', mape)
```

### Classification metrics
```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    log_loss,
)

y_true = [1, 1, 1, 1, 0, 0, 0, 0, 1, 0]
y_pred = [1, 1, 0, 1, 0, 0, 1, 0, 1, 0]
y_prob = [0.95, 0.90, 0.40, 0.80, 0.10, 0.20, 0.60, 0.30, 0.85, 0.15]

print('Accuracy :', accuracy_score(y_true, y_pred))
print('Precision:', precision_score(y_true, y_pred))
print('Recall   :', recall_score(y_true, y_pred))
print('F1 Score :', f1_score(y_true, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_true, y_pred))
print('ROC AUC  :', roc_auc_score(y_true, y_prob))
print('Log Loss :', log_loss(y_true, y_prob))
```

---

## 12) Final 10-Line Summary

1. Regression predicts numbers; classification predicts classes.
2. MSE and RMSE punish large errors more strongly.
3. MAE gives average error directly and is easier to explain.
4. R² measures how much variance is explained.
5. Adjusted R² is better than plain R² for comparing models with different feature counts.
6. MAPE gives percentage error but fails near zero actual values.
7. Accuracy is useful mainly when classes are balanced.
8. Precision is critical when false positives are costly.
9. Recall is critical when false negatives are dangerous.
10. F1 balances precision and recall; ROC-AUC measures ranking; Log Loss measures probability quality.
