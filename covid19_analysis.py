# COVID-19 -- simple EDA + basic ML (with visualizations)
# Keep the three CSV files inside the project data/ folder, then run this script.
# pip install pandas numpy matplotlib seaborn scikit-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, mean_absolute_error

sns.set_theme(style="whitegrid")   # nicer default look for all charts


# 1. LOAD DATA

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

day = pd.read_csv(DATA_DIR / "day_wise.csv", parse_dates=["Date"])
country = pd.read_csv(DATA_DIR / "country_wise_latest.csv")
full = pd.read_csv(DATA_DIR / "full_grouped.csv", parse_dates=["Date"])

print(day.head())
print(day.info())
print(day.describe())
print("Missing values:\n", country.isnull().sum())


# 2. EDA (exploratory data analysis) -- 13 charts


# --- Chart 1: global cases over time
plt.figure(figsize=(9, 5))
for col in ["Confirmed", "Active", "Recovered", "Deaths"]:
    plt.plot(day["Date"], day[col], label=col)
plt.title("1. Global COVID-19 cases over time")
plt.legend()
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# --- Chart 2: daily new cases with a 7-day average line
plt.figure(figsize=(9, 4))
plt.plot(day["Date"], day["New cases"], color="lightblue", label="Daily new cases")
plt.plot(day["Date"], day["New cases"].rolling(7).mean(), color="blue", label="7-day average")
plt.title("2. Daily new cases (world)")
plt.legend()
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# --- Chart 3: death rate and recovery rate over time
plt.figure(figsize=(9, 4))
plt.plot(day["Date"], day["Deaths / 100 Cases"], color="red", label="Deaths per 100 cases")
plt.plot(day["Date"], day["Recovered / 100 Cases"], color="green", label="Recovered per 100 cases")
plt.title("3. Death rate and recovery rate over time")
plt.legend()
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# --- Chart 4 and 5: top 10 countries by confirmed cases and by deaths
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
top_conf = country.sort_values("Confirmed", ascending=False).head(10)
top_death = country.sort_values("Deaths", ascending=False).head(10)
sns.barplot(data=top_conf, x="Confirmed", y="Country/Region", color="steelblue", ax=axes[0])
axes[0].set_title("4. Top 10 countries by confirmed cases")
sns.barplot(data=top_death, x="Deaths", y="Country/Region", color="indianred", ax=axes[1])
axes[1].set_title("5. Top 10 countries by deaths")
plt.tight_layout()
plt.show()

# --- Chart 6: how the top 5 countries grew over time
top5 = country.sort_values("Confirmed", ascending=False).head(5)["Country/Region"]
plt.figure(figsize=(9, 5))
for name in top5:
    one = full[full["Country/Region"] == name]
    plt.plot(one["Date"], one["Confirmed"], label=name)
plt.title("6. Confirmed cases over time (top 5 countries)")
plt.legend()
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# --- Chart 7 and 8: WHO regions (pie chart and bar chart)
region = country.groupby("WHO Region")[["Confirmed", "Deaths"]].sum()
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].pie(region["Confirmed"], labels=region.index, autopct="%1.0f%%", startangle=90)
axes[0].set_title("7. Share of confirmed cases by WHO region")
region.plot(kind="bar", ax=axes[1])
axes[1].set_title("8. Cases and deaths by WHO region")
plt.tight_layout()
plt.show()

# --- Chart 9: how death rates are spread across countries
# (only countries with at least 1000 cases, so tiny numbers don't distort it)
big = country[country["Confirmed"] >= 1000]
plt.figure(figsize=(8, 4))
sns.histplot(big["Deaths / 100 Cases"], bins=25, color="indianred")
plt.title("9. Distribution of death rate (countries with 1000+ cases)")
plt.tight_layout()
plt.show()

# --- Chart 10: death rate by region (boxplot)
plt.figure(figsize=(10, 5))
sns.boxplot(data=big, x="WHO Region", y="Deaths / 100 Cases")
plt.title("10. Death rate by WHO region")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# --- Chart 11: confirmed vs deaths for every country (log scale)
plt.figure(figsize=(7, 6))
sns.scatterplot(data=country, x="Confirmed", y="Deaths", hue="WHO Region")
plt.xscale("log")
plt.yscale("log")
plt.title("11. Confirmed vs deaths (log scale)")
plt.tight_layout()
plt.show()

# --- Chart 12: fastest growing countries in the last week
fast = country[country["Confirmed"] >= 5000].sort_values("1 week % increase", ascending=False).head(10)
plt.figure(figsize=(8, 5))
sns.barplot(data=fast, x="1 week % increase", y="Country/Region", color="seagreen")
plt.title("12. Fastest growth in last 7 days (countries with 5000+ cases)")
plt.tight_layout()
plt.show()

# --- Chart 13: correlation between columns
cols = ["Confirmed", "Deaths", "Recovered", "Active", "New cases",
        "Deaths / 100 Cases", "Recovered / 100 Cases"]
plt.figure(figsize=(8, 6))
sns.heatmap(country[cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("13. Correlation heatmap")
plt.tight_layout()
plt.show()


# 3. ML MODEL 1: predict Deaths of a country (regression)
#    Linear Regression vs Random Forest
# Country numbers vary hugely (USA = millions, small islands = a few),
# so we take log10 of everything. The +1 avoids log(0).
X = np.log10(country[["Confirmed", "Confirmed last week", "New cases"]] + 1)
y = np.log10(country["Deaths"] + 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lin = LinearRegression().fit(X_train, y_train)
rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)

print("\n--- Model 1: predict Deaths per country (log scale) ---")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (name, model) in zip(axes, [("Linear Regression", lin), ("Random Forest", rf)]):
    pred = model.predict(X_test)
    print(f"{name:18s} R2 = {r2_score(y_test, pred):.3f}   MAE = {mean_absolute_error(y_test, pred):.3f}")

    # --- Chart 14 and 15: actual vs predicted (points near the line = good model)
    ax.scatter(y_test, pred, alpha=0.6)
    ax.plot([0, 6], [0, 6], color="red", linestyle="--")   # perfect prediction line
    ax.set_xlabel("Actual deaths (log10)")
    ax.set_ylabel("Predicted deaths (log10)")
    ax.set_title(f"{name}  (R2 = {r2_score(y_test, pred):.2f})")
plt.suptitle("14-15. Model 1: actual vs predicted")
plt.tight_layout()
plt.show()

# --- Chart 16: which input matters most for the Random Forest?
plt.figure(figsize=(7, 3))
pd.Series(rf.feature_importances_, index=X.columns).sort_values().plot(kind="barh", color="teal")
plt.title("16. Random Forest feature importance")
plt.tight_layout()
plt.show()


# 4. ML MODEL 2: forecast global confirmed cases (time series)
#    Use day number as input, train on the past, test on the last 28 days

day["day_no"] = np.arange(len(day))
X = day[["day_no"]]
y = day["Confirmed"]

split = len(day) - 28                      # last 28 days are the test set
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# straight line
line = LinearRegression().fit(X_train, y_train)
pred_line = line.predict(X_test)

# curve (polynomial, degree 2)
poly = PolynomialFeatures(degree=2)
curve = LinearRegression().fit(poly.fit_transform(X_train), y_train)
pred_curve = curve.predict(poly.transform(X_test))

print("\n--- Model 2: forecast confirmed cases (last 28 days) ---")
print(f"Straight line  R2 = {r2_score(y_test, pred_line):.3f}   MAE = {mean_absolute_error(y_test, pred_line):.0f}")
print(f"Polynomial (2) R2 = {r2_score(y_test, pred_curve):.3f}   MAE = {mean_absolute_error(y_test, pred_curve):.0f}")

# --- Chart 17: forecast vs actual (the grey area is the test period)
plt.figure(figsize=(9, 5))
plt.plot(day["Date"], y, label="Actual", color="black")
plt.plot(day["Date"][split:], pred_line, label="Linear", linestyle="--")
plt.plot(day["Date"][split:], pred_curve, label="Polynomial", linestyle="--")
plt.axvspan(day["Date"].iloc[split], day["Date"].iloc[-1], color="grey", alpha=0.15, label="Test period")
plt.title("17. Model 2: forecasting global confirmed cases")
plt.legend()
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# --- Chart 18: error of each model on the test days
plt.figure(figsize=(9, 4))
plt.plot(day["Date"][split:], pred_line - y_test, label="Linear error")
plt.plot(day["Date"][split:], pred_curve - y_test, label="Polynomial error")
plt.axhline(0, color="black", linewidth=1)
plt.title("18. Prediction error (predicted - actual)")
plt.legend()
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


# 5. ML MODEL 3: group similar countries (KMeans clustering)

cluster_data = pd.DataFrame({
    "log_confirmed": np.log10(country["Confirmed"] + 1),
    "death_rate": country["Deaths / 100 Cases"],
    "recovery_rate": country["Recovered / 100 Cases"],
    "weekly_growth": country["1 week % increase"].clip(upper=60),
})
scaled = StandardScaler().fit_transform(cluster_data)

# --- Chart 19: elbow plot (helps choose the number of clusters)
inertia = [KMeans(n_clusters=k, n_init=10, random_state=42).fit(scaled).inertia_ for k in range(1, 9)]
plt.figure(figsize=(6, 4))
plt.plot(range(1, 9), inertia, marker="o")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia (lower = tighter clusters)")
plt.title("19. Elbow plot")
plt.tight_layout()
plt.show()

kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
country["cluster"] = kmeans.fit_predict(scaled)

print("\n--- Model 3: KMeans clusters ---")
print(country.groupby("cluster")[["Confirmed", "Deaths / 100 Cases",
                                  "Recovered / 100 Cases", "1 week % increase"]].median().round(1))
for c in range(4):
    names = country[country["cluster"] == c].sort_values("Confirmed", ascending=False)["Country/Region"].head(8)
    print(f"Cluster {c}:", ", ".join(names))

# --- Chart 20 and 21: clusters on two different views + how many countries per cluster
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
sns.scatterplot(data=country, x="Recovered / 100 Cases", y="Deaths / 100 Cases",
                hue="cluster", palette="tab10", ax=axes[0])
axes[0].set_title("20. Clusters: recovery rate vs death rate")
sns.scatterplot(data=country, x="Confirmed", y="1 week % increase",
                hue="cluster", palette="tab10", ax=axes[1])
axes[1].set_xscale("log")
axes[1].set_title("21. Clusters: cases vs weekly growth")
sns.countplot(data=country, x="cluster", hue="cluster", palette="tab10", legend=False, ax=axes[2])
axes[2].set_title("22. Countries per cluster")
plt.tight_layout()
plt.show()