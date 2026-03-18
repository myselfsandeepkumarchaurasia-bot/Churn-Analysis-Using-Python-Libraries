# Evaluation Metrics Interview Q&A

## Concept-Based Q&A

### 1. What is the difference between MAE and RMSE?
**Answer:** MAE uses absolute errors, so each error contributes linearly. RMSE squares errors before averaging, so larger errors receive more punishment. Use MAE for simple average error and RMSE when large misses matter more.

### 2. Can R² be negative?
**Answer:** Yes. R² becomes negative when the model performs worse than simply predicting the mean of the target for every observation.

### 3. Why do we need Adjusted R²?
**Answer:** Because regular R² often increases when more features are added, even if those features are useless. Adjusted R² adds a penalty for unnecessary predictors.

### 4. Why can accuracy be misleading?
**Answer:** In imbalanced data, a model can predict the majority class almost always and still get high accuracy while being practically useless.

### 5. Precision vs Recall in one line?
**Answer:** Precision asks, “When the model predicts positive, how often is it correct?” Recall asks, “Out of all actual positives, how many did the model catch?”

### 6. Why does F1 use harmonic mean?
**Answer:** Because harmonic mean drops sharply when either precision or recall is low, so it prevents one high value from hiding one weak value.

### 7. Why is log loss powerful?
**Answer:** Because it evaluates not only whether the class is correct, but also whether the predicted probability is sensible and well calibrated.

## Scenario-Based Q&A

### 8. Which metric matters most in cancer screening?
**Answer:** Recall. Missing a truly sick patient is usually more dangerous than sending a healthy patient for additional testing.

### 9. Which metric matters most in spam detection?
**Answer:** Often precision, because genuine messages should not be wrongly filtered. In practice F1 is also useful if both spam leakage and false filtering matter.

### 10. When should you report RMSE in house price prediction?
**Answer:** When large prediction mistakes should be penalized more strongly and the stakeholder wants error expressed in actual currency units.

### 11. Why is MAPE popular in forecasting dashboards?
**Answer:** Because percentage errors are easy for business teams to interpret and compare across products or regions.

### 12. Which metric would you use to compare multiple regression models with different numbers of predictors?
**Answer:** Adjusted R², because it accounts for the number of predictors while evaluating explained variance.

## Trick Q&A

### 13. Does high accuracy always mean a good model?
**Answer:** No. A highly imbalanced problem can show high accuracy even when the model fails to detect the minority class.

### 14. Is higher recall always better?
**Answer:** Not always. Recall can be increased by predicting more positives, but that may also sharply increase false positives.

### 15. Can precision be perfect while recall is poor?
**Answer:** Yes. If the model predicts positive only a few times and all of them are correct, precision can be 1 while many actual positives are missed.

### 16. Can recall be perfect while precision is poor?
**Answer:** Yes. If the model predicts nearly everything as positive, it can catch all actual positives but create many false positives.

### 17. Does low log loss guarantee the highest accuracy?
**Answer:** No. Log loss and accuracy evaluate different things. Accuracy uses hard labels; log loss uses probability quality.

## Short Revision Q&A

### 18. Which regression metric is easiest to explain to business users?
**Answer:** MAE, because it directly says the average prediction is off by a certain number of units.

### 19. Which regression metric is most sensitive to outliers?
**Answer:** MSE, and therefore RMSE as well.

### 20. Which classification metric should you inspect first before all others?
**Answer:** The confusion matrix, because it reveals the exact pattern of errors.
