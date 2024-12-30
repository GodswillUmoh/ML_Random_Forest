# ML_Random_Forest

## Explanation
> A Random Forest is an ensemble learning technique primarily used for classification and regression tasks. It builds multiple decision trees and combines their outputs to produce a more accurate and robust prediction.
> It is best with high dimensional dataset. That is dataset with multiple features
---

## What is Ensemble Learning?
> Ensemble Learning is a machine learning technique where multiple models (referred to as "weak learners" or "base models") are combined to produce a more robust and accurate predictive model. The idea is that by combining diverse models, the ensemble can outperform any individual model.
> It is when you take multiple algorithm or same algorithm multiple time to build a robust model for better predictions

## How Random forest works
> Many predictions are made from each tree say about different 500 predictions are made, nut the final outcome takes the average of all 500 predictions to produce the final one outcome result.

## Advantages of Random Forest
> + Robustness to Overfitting: Reduces overfitting by averaging predictions from multiple trees.
> + Handles Non-linear Relationships: Decision trees in the forest can capture complex patterns in the data.
> + Feature Importance: Provides a measure of feature importance, helping with feature selection.
> + Handles Missing Data: Can handle datasets with missing values effectively.

## Python code for Random Forest Model
---
### Importing the libraries
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

### Importing the dataset
```python
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values

```

### Training the Random Forest Regression model on the whole dataset
```python
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state=0)
regressor.fit(X, y)
```

### Predicting a new result
```python
regressor.predict([[6.5]])
```

### Visualising the Random Forest Regression results (higher resolution)
```python
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
```

+ [To View Code in the Terminal, Click Here](https://colab.research.google.com/drive/1PSzboLTX2uNR-zFVYVl7iJtKqe0xadhS#scrollTo=ArK19GxT4g8m)


