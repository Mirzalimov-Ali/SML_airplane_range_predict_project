import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
from joblib import dump
from skopt import BayesSearchCV
from src.logger import get_logger
from rich.table import Table
from rich.console import Console
from skopt.space import Categorical

logger = get_logger('use_training', 'training.log')

os.chdir(r'C:\SML_Projects\SML_airplane_price_project')

df = pd.read_csv('data/preprocessed/preprocessed_dataset.csv')
# df = df.sample(frac=0.4, random_state=42).reset_index(drop=True)

X = df.drop(columns="Range_(km)", errors='ignore')
y = df['Range_(km)']

kf = KFold(n_splits=3, shuffle=True, random_state=42)

# ===================== Bayesian Optimization ==========================
search_space = {
    'fit_intercept': Categorical([True, False]),
    'positive': Categorical([True, False])
}

bayes_search = BayesSearchCV(
    estimator=LinearRegression(),
    search_spaces=search_space,
    n_iter=20,
    scoring='r2',
    cv=kf,
    n_jobs=-1,
    random_state=42,
    verbose=0
)

bayes_search.fit(X, y)

model = bayes_search.best_estimator_

logger.info(f"Bayesian Best Params: {bayes_search.best_params_}")
print("\nBest Params:", bayes_search.best_params_)

# ===================== Train/Test Split ==========================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

r2 = r2_score(y_test, y_pred_test)
mae = mean_absolute_error(y_test, y_pred_test)
scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
kf_mean = scores.mean()
kf_std = scores.std()

# ===================== Save Model ==========================
os.makedirs('pipeline', exist_ok=True)
dump(model, 'pipeline/final_pipeline.joblib', compress=3)
logger.info("Final model pipeline saved.")

# ===================== Save Results Table ==========================
results = []
console = Console()

results.append([
    "LinearRegression",
    r2,
    mae,
    kf_mean,
    kf_std,
])

results_sorted = sorted(results, key=lambda x: x[-1], reverse=True)
best_model = max(results_sorted, key=lambda x: x[-1])
worst_model = min(results_sorted, key=lambda x: x[-1])

table = Table(title="LinearRegression  Results", show_lines=True)
table.add_column("Algorithm")
table.add_column("R2 score")
table.add_column("Mean Absolute Error")
table.add_column("K-Fold mean")
table.add_column("K-Fold std")

for row in results_sorted:
    algo, r2, mae, kmean, kstd = row

    if row == best_model:
        table.add_row(f"[bold green]{algo}[/bold green]",
                      f"[bold green]{r2}[/bold green]",
                      f"[bold green]{mae:.2f}[/bold green]",
                      f"[bold green]{kmean:.2f}[/bold green]",
                      f"[bold green]{kstd:.2f}[/bold green]")
    elif row == worst_model:
        table.add_row(f"[bold red]{algo}[/bold red]",
                      f"[bold red]{r2}[/bold red]",
                      f"[bold red]{mae:.2f}[/bold red]",
                      f"[bold red]{kmean:.2f}[/bold red]",
                      f"[bold red]{kstd:.2f}[/bold red]")
    else:
        table.add_row(algo, r2, f"{mae:.2f}", f"{kmean:.2f}", f"{kstd:.2f}")

console.print(table)

temp_console = Console(record=True)
temp_console.print(table)
text = temp_console.export_text()
with open("results/final_results.txt", "w", encoding="utf-8") as f:
    f.write(text)
logger.info("Comparison table saved at results/final_results.txt")

print("Results saved to results/final_results.txt")