# =========================
# Imports
# =========================
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import RobustScaler


# =========================
# Load data
# =========================
websites = pd.read_csv("../website.csv")


# =========================
# Explore data
# =========================
print(websites.head())
print(websites.tail())
print("Shape:", websites.shape)
print("Columns:\n", websites.columns)
print(websites.info())
print(websites.dtypes)

print("\nFeature columns:")
print(websites.drop("Type", axis=1).columns)

print("\nTarget column: Type")
print("0 = benign, 1 = malicious")

print("\nCategorical columns:")
print(websites.select_dtypes(include="object").columns)


# =========================
# Correlation analysis
# =========================
corr = websites.select_dtypes(include=["int64", "float64"]).corr()

plt.figure(figsize=(12, 8))
sns.heatmap(corr, cmap="coolwarm", annot=False)
plt.title("Correlation Matrix")
plt.show()

high_corr = (
    corr.abs()
    .where(corr.abs() > 0.8)
    .stack()
    .reset_index()
)
high_corr.columns = ["Column 1", "Column 2", "Correlation"]
high_corr = high_corr[high_corr["Column 1"] != high_corr["Column 2"]]
high_corr = high_corr[high_corr["Column 1"] < high_corr["Column 2"]]

print("\nHighly correlated pairs:")
print(high_corr)


# =========================
# XGBoost feature importance
# =========================
xgb = XGBClassifier(random_state=1)
X_num = websites.select_dtypes(include=["int64", "float64"]).drop("Type", axis=1)
y_num = websites["Type"]

xgb.fit(X_num, y_num)

sort_idx = xgb.feature_importances_.argsort()
plt.figure(figsize=(10, 6))
plt.barh(X_num.columns[sort_idx], xgb.feature_importances_[sort_idx])
plt.title("XGBoost Feature Importances")
plt.show()


# =========================
# Remove highly collinear columns
# =========================
corr_abs = websites.select_dtypes(include=["int64", "float64"]).corr().abs()
upper = corr_abs.where(np.triu(np.ones(corr_abs.shape), k=1).astype(bool))

to_drop = [col for col in upper.columns if any(upper[col] > 0.8)]

# Drop first 4 highly collinear columns as requested
websites = websites.drop(columns=to_drop[:4])
print("\nDropped columns:", to_drop[:4])


# =========================
# Missing values
# =========================
print("\nMissing values before cleaning:")
print(websites.isna().sum()[websites.isna().sum() > 0])

# Drop columns with more than 50% missing values
websites = websites.loc[:, websites.isna().mean() < 0.5]

# Drop remaining rows with missing values
websites = websites.dropna()

print("\nMissing values after cleaning:")
print(websites.isna().sum())


# =========================
# Clean WHOIS_COUNTRY
# =========================
print("\nWHOIS_COUNTRY value counts before cleaning:")
print(websites["WHOIS_COUNTRY"].value_counts())

good_country = {
    "None": "None",
    "US": "US",
    "SC": "SC",
    "GB": "UK",
    "UK": "UK",
    "RU": "RU",
    "AU": "AU",
    "CA": "CA",
    "PA": "PA",
    "se": "SE",
    "IN": "IN",
    "LU": "LU",
    "TH": "TH",
    "[u'GB'; u'UK']": "UK",
    "FR": "FR",
    "NL": "NL",
    "UG": "UG",
    "JP": "JP",
    "CN": "CN",
    "SE": "SE",
    "SI": "SI",
    "IL": "IL",
    "ru": "RU",
    "KY": "KY",
    "AT": "AT",
    "CZ": "CZ",
    "PH": "PH",
    "BE": "BE",
    "NO": "NO",
    "TR": "TR",
    "LV": "LV",
    "DE": "DE",
    "ES": "ES",
    "BR": "BR",
    "us": "US",
    "KR": "KR",
    "HK": "HK",
    "UA": "UA",
    "CH": "CH",
    "United Kingdom": "UK",
    "BS": "BS",
    "PK": "PK",
    "IT": "IT",
    "Cyprus": "CY",
    "BY": "BY",
    "AE": "AE",
    "IE": "IE",
    "UY": "UY",
    "KG": "KG",
}

websites["WHOIS_COUNTRY"] = websites["WHOIS_COUNTRY"].apply(lambda x: good_country[x])

print("\nWHOIS_COUNTRY unique values after standardization:")
print(websites["WHOIS_COUNTRY"].unique())


def print_bar_plot(x, y, figsize=(15, 6), rotation=90):
    plt.figure(figsize=figsize)
    plt.bar(x, y)
    plt.xticks(rotation=rotation)
    plt.tight_layout()
    plt.show()


print_bar_plot(
    websites["WHOIS_COUNTRY"].value_counts().index,
    websites["WHOIS_COUNTRY"].value_counts().values
)

# Keep top 10 countries, label others as OTHER
top10 = websites["WHOIS_COUNTRY"].value_counts().nlargest(10).index
websites["WHOIS_COUNTRY"] = websites["WHOIS_COUNTRY"].apply(
    lambda x: x if x in top10 else "OTHER"
)

print("\nWHOIS_COUNTRY after top-10 grouping:")
print(websites["WHOIS_COUNTRY"].value_counts())


# =========================
# Drop unwanted categorical/date columns
# =========================
websites = websites.drop(columns=["WHOIS_STATEPRO", "WHOIS_REGDATE", "WHOIS_UPDATED_DATE"])

print("\nData types after dropping WHOIS columns:")
print(websites.dtypes)


# =========================
# Drop URL
# =========================
websites = websites.drop(columns=["URL"])


# =========================
# Inspect remaining categorical columns
# =========================
print("\nCHARSET value counts:")
print(websites["CHARSET"].value_counts())

print("\nSERVER value counts before cleaning:")
print(websites["SERVER"].value_counts())


# =========================
# Clean SERVER
# =========================
websites["SERVER"] = websites["SERVER"].astype(str).apply(
    lambda x: "Microsoft" if "Microsoft" in x
    else "Apache" if "Apache" in x
    else "nginx" if "nginx" in x
    else "Other"
)

print("\nSERVER value counts after cleaning:")
print(websites["SERVER"].value_counts())


# =========================
# Convert categorical variables to dummy variables
# =========================
website_dummy = pd.get_dummies(websites, drop_first=True)

print("\nFinal data types after get_dummies:")
print(website_dummy.dtypes)


# =========================
# Train-test split
# =========================
X = website_dummy.drop(columns=["Type"])
y = website_dummy["Type"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# =========================
# Logistic Regression
# =========================
logreg = LogisticRegression(max_iter=5000)
logreg.fit(X_train, y_train)

y_pred_logreg = logreg.predict(X_test)

print("\nLogistic Regression")
print(confusion_matrix(y_test, y_pred_logreg))
print("Accuracy:", accuracy_score(y_test, y_pred_logreg))


# =========================
# Decision Tree (max_depth=3)
# =========================
tree_3 = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_3.fit(X_train, y_train)

y_pred_tree_3 = tree_3.predict(X_test)

print("\nDecision Tree (max_depth=3)")
print(confusion_matrix(y_test, y_pred_tree_3))
print("Accuracy:", accuracy_score(y_test, y_pred_tree_3))


# =========================
# Decision Tree (max_depth=5)
# =========================
tree_5 = DecisionTreeClassifier(max_depth=5, random_state=42)
tree_5.fit(X_train, y_train)

y_pred_tree_5 = tree_5.predict(X_test)

print("\nDecision Tree (max_depth=5)")
print(confusion_matrix(y_test, y_pred_tree_5))
print("Accuracy:", accuracy_score(y_test, y_pred_tree_5))


# =========================
# Bonus: Feature Scaling + Logistic Regression
# =========================
scaler = RobustScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg_scaled = LogisticRegression(max_iter=5000)
logreg_scaled.fit(X_train_scaled, y_train)

y_pred_logreg_scaled = logreg_scaled.predict(X_test_scaled)

print("\nLogistic Regression with RobustScaler")
print(confusion_matrix(y_test, y_pred_logreg_scaled))
print("Accuracy:", accuracy_score(y_test, y_pred_logreg_scaled))