"""
LAB4 - INTRO TO AI/ML
Feature Selection Methods
"""

# -------------------------------
# Filter Methods
# -------------------------------

# 1. Variance Threshold
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.feature_selection import VarianceThreshold

# Load Iris dataset
iris = load_iris(as_frame=True)
X, y = iris.data, iris.target

# Apply Variance Threshold
selector = VarianceThreshold(threshold=0.01)
X_new = selector.fit_transform(X)
print("Selected Features:", X.columns[selector.get_support()])


# 2. Correlation-based Selection
import numpy as np

# Compute correlation matrix
corr_matrix = X.corr().abs()

# Upper triangle
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Drop highly correlated features (threshold=0.85)
to_drop = [column for column in upper.columns if any(upper[column] > 0.85)]
print("Dropped Features (High Correlation):", to_drop)


# 3. Chi-Square Test
# (works only with non-negative data, hence we use Iris)
from sklearn.feature_selection import chi2, SelectKBest

# Apply Chi-Square test
chi2_selector = SelectKBest(chi2, k=2)  # select top 2 features
X_kbest = chi2_selector.fit_transform(X, y)
print("Top features by Chi-Square:", X.columns[chi2_selector.get_support()])


# 4. Mutual Information
from sklearn.feature_selection import mutual_info_classif

mi = mutual_info_classif(X, y, random_state=42)
mi_scores = pd.Series(mi, index=X.columns)
print("Mutual Information Scores:")
print(mi_scores.sort_values(ascending=False))


# -------------------------------
# Wrapper Methods
# -------------------------------

# 5. Recursive Feature Elimination (RFE)
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Logistic regression as base model
model = LogisticRegression(max_iter=200)
rfe = RFE(model, n_features_to_select=2)
rfe.fit(X, y)
print("Selected Features by RFE:", X.columns[rfe.support_])


# 6. Stepwise Selection (Forward Selection Example)
# (Manual implementation using statsmodels AIC – only works for regression)
import statsmodels.api as sm
from sklearn.datasets import load_diabetes

# Load dataset
data = load_diabetes(as_frame=True)
X2, y2 = data.data, data.target

def forward_selection(X, y):
    remaining_features = list(X.columns)
    selected_features = []
    current_score, best_new_score = float('inf'), float('inf')
    
    while remaining_features:
        scores_with_candidates = []
        for candidate in remaining_features:
            model = sm.OLS(y, sm.add_constant(X[selected_features + [candidate]])).fit()
            score = model.aic
            scores_with_candidates.append((score, candidate))
        
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates[0]
        if current_score > best_new_score:
            remaining_features.remove(best_candidate)
            selected_features.append(best_candidate)
            current_score = best_new_score
        else:
            break
    return selected_features

selected = forward_selection(X2, y2)
print("Selected Features by Stepwise Selection:", selected)


# -------------------------------
# Embedded Methods
# -------------------------------

# 7. Lasso (L1 Regularization)
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.01)
lasso.fit(X2, y2)
print("Lasso Coefficients:")
print(pd.Series(lasso.coef_, index=X2.columns))


# 8. Ridge (L2 Regularization)
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0)
ridge.fit(X2, y2)
print("Ridge Coefficients:")
print(pd.Series(ridge.coef_, index=X2.columns))


# 9. ElasticNet (L1 + L2)
from sklearn.linear_model import ElasticNet

enet = ElasticNet(alpha=0.01, l1_ratio=0.5)
enet.fit(X2, y2)
print("ElasticNet Coefficients:")
print(pd.Series(enet.coef_, index=X2.columns))


# 10. Decision Trees
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier(random_state=42)
tree.fit(X, y)
print("Decision Tree Feature Importance:")
print(pd.Series(tree.feature_importances_, index=X.columns))


# 11. Random Forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=42)
rf.fit(X, y)
print("Random Forest Feature Importance:")
print(pd.Series(rf.feature_importances_, index=X.columns))


# 12. XGBoost
from xgboost import XGBClassifier

xgb = XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")
xgb.fit(X, y)
print("XGBoost Feature Importance:")
print(pd.Series(xgb.feature_importances_, index=X.columns))
