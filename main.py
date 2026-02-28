# ── 1. IMPORT LIBRARIES ──────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score,
    precision_score, recall_score, accuracy_score
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

print("=" * 60)
print("     STUDENT LOAN DEFAULT PREDICTOR")
print("=" * 60)


# ── 2. CREATE REALISTIC DATASET ───────────────────────────────
# We are creating our OWN dataset based on real-world

np.random.seed(42)
n = 5000  # 5000 student loan records

print("\n📦 Creating realistic loan dataset...")

# Basic student information
age            = np.random.randint(18, 35, n)
income         = np.random.randint(15000, 120000, n)       # Annual income in ₹
loan_amount    = np.random.randint(50000, 1500000, n)      # Loan in ₹
loan_term      = np.random.choice([12, 24, 36, 48, 60], n) # Months
credit_score   = np.random.randint(300, 850, n)

# Categorical features
education      = np.random.choice(['Graduate', 'Postgraduate', 'PhD'], n,
                                   p=[0.6, 0.3, 0.1])
employment     = np.random.choice(['Full-time', 'Part-time', 'Unemployed'], n,
                                   p=[0.5, 0.3, 0.2])
marital_status = np.random.choice(['Single', 'Married'], n, p=[0.6, 0.4])
dependents     = np.random.randint(0, 5, n)
cosigner       = np.random.choice([0, 1], n, p=[0.4, 0.6])  # 1 = has cosigner

# ── Create Target Variable (Default = 1, Repay = 0) ──────────
# Based on REAL risk logic used by banks:
# Low credit score → more likely to default
# High loan vs income → more likely to default
# Unemployed → more likely to default
# No cosigner → more likely to default

default_probability = (
    (credit_score < 500).astype(int) * 0.35 +       # Low credit score is biggest risk
    (loan_amount / income > 5).astype(int) * 0.25 +  # Loan much higher than income
    (employment == 'Unemployed').astype(int) * 0.20 + # No job = big risk
    (cosigner == 0).astype(int) * 0.10 +              # No guarantor
    (dependents > 2).astype(int) * 0.10               # Too many dependents
)

# Normalize to 0-1 range and add some randomness
# Threshold set to 0.65 so only ~25% students default
# This matches REAL WORLD bank data where 75% repay, 25% default
default_probability = np.clip(default_probability + np.random.normal(0, 0.1, n), 0, 1)
default             = (default_probability > 0.65).astype(int)

# Build the DataFrame
df = pd.DataFrame({
    'Age'           : age,
    'Income'        : income,
    'LoanAmount'    : loan_amount,
    'LoanTerm'      : loan_term,
    'CreditScore'   : credit_score,
    'Education'     : education,
    'Employment'    : employment,
    'MaritalStatus' : marital_status,
    'Dependents'    : dependents,
    'CoSigner'      : cosigner,
    'Default'       : default
})

print(f"   ✅ Dataset created: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\n📊 Default Rate:")
default_counts = df['Default'].value_counts()
print(f"   Will Repay  (0): {default_counts[0]} students ({default_counts[0]/n*100:.1f}%)")
print(f"   Will Default(1): {default_counts[1]} students ({default_counts[1]/n*100:.1f}%)")
print(f"\n   ⚠️  Data is IMBALANCED — this is why we need SMOTE!")


# ── 3. EDA — EXPLORATORY DATA ANALYSIS ───────────────────────
print("\n📊 Running Exploratory Data Analysis...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Student Loan Default — Exploratory Data Analysis',
             fontsize=16, fontweight='bold', y=1.02)

# Plot 1: Default vs Repay count
ax1 = axes[0, 0]
colors = ['#4CAF50', '#F44336']
df['Default'].value_counts().plot(kind='bar', ax=ax1, color=colors, edgecolor='black', rot=0)
ax1.set_title('Default vs Repay Count', fontweight='bold')
ax1.set_xticklabels(['Will Repay ✅', 'Will Default ❌'])
ax1.set_ylabel('Number of Students')
for bar in ax1.patches:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
             f'{int(bar.get_height())}', ha='center', fontweight='bold')

# Plot 2: Credit Score Distribution by Default
ax2 = axes[0, 1]
df[df['Default'] == 0]['CreditScore'].hist(ax=ax2, alpha=0.6, color='#4CAF50',
                                            label='Will Repay', bins=30)
df[df['Default'] == 1]['CreditScore'].hist(ax=ax2, alpha=0.6, color='#F44336',
                                            label='Will Default', bins=30)
ax2.set_title('Credit Score vs Default', fontweight='bold')
ax2.set_xlabel('Credit Score')
ax2.set_ylabel('Frequency')
ax2.legend()

# Plot 3: Default Rate by Employment Type
ax3 = axes[0, 2]
emp_default = df.groupby('Employment')['Default'].mean() * 100
emp_default.sort_values(ascending=False).plot(kind='bar', ax=ax3,
    color=['#F44336', '#FF9800', '#4CAF50'], edgecolor='black', rot=0)
ax3.set_title('Default Rate by Employment Type', fontweight='bold')
ax3.set_ylabel('Default Rate (%)')
for bar in ax3.patches:
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{bar.get_height():.1f}%', ha='center', fontweight='bold')

# Plot 4: Loan Amount Distribution
ax4 = axes[1, 0]
df[df['Default'] == 0]['LoanAmount'].hist(ax=ax4, alpha=0.6, color='#4CAF50',
                                           label='Will Repay', bins=30)
df[df['Default'] == 1]['LoanAmount'].hist(ax=ax4, alpha=0.6, color='#F44336',
                                           label='Will Default', bins=30)
ax4.set_title('Loan Amount vs Default', fontweight='bold')
ax4.set_xlabel('Loan Amount (₹)')
ax4.set_ylabel('Frequency')
ax4.legend()

# Plot 5: Default Rate by Education
ax5 = axes[1, 1]
edu_default = df.groupby('Education')['Default'].mean() * 100
edu_default.sort_values(ascending=False).plot(kind='bar', ax=ax5,
    color=['#2196F3', '#9C27B0', '#FF9800'], edgecolor='black', rot=0)
ax5.set_title('Default Rate by Education Level', fontweight='bold')
ax5.set_ylabel('Default Rate (%)')
for bar in ax5.patches:
    ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f'{bar.get_height():.1f}%', ha='center', fontweight='bold')

# Plot 6: Income Distribution
ax6 = axes[1, 2]
df[df['Default'] == 0]['Income'].hist(ax=ax6, alpha=0.6, color='#4CAF50',
                                       label='Will Repay', bins=30)
df[df['Default'] == 1]['Income'].hist(ax=ax6, alpha=0.6, color='#F44336',
                                       label='Will Default', bins=30)
ax6.set_title('Income vs Default', fontweight='bold')
ax6.set_xlabel('Annual Income (₹)')
ax6.set_ylabel('Frequency')
ax6.legend()

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ EDA plots saved as 'eda_plots.png'")


# ── 4. FEATURE ENGINEERING ───────────────────────────────────
print("\n🛠️  Creating new financial risk features...")

# Loan to Income Ratio — KEY risk indicator used by real banks!
# High ratio = student owes much more than they earn = risky
df['LoanToIncomeRatio'] = df['LoanAmount'] / df['Income']

# Monthly EMI (simple calculation)
df['MonthlyEMI'] = df['LoanAmount'] / df['LoanTerm']

# EMI Burden — what % of monthly income goes to EMI
df['EMIBurden'] = df['MonthlyEMI'] / (df['Income'] / 12)

print("   → LoanToIncomeRatio : Loan ÷ Annual Income")
print("   → MonthlyEMI        : Loan ÷ Loan Term")
print("   → EMIBurden         : Monthly EMI ÷ Monthly Income")
print("   ✅ These are REAL metrics banks use for risk assessment!")


# ── 5. ENCODE CATEGORICAL COLUMNS ────────────────────────────
# ML models only understand numbers, not words like "Graduate"
# So we convert text → numbers using LabelEncoder

print("\n🔢 Converting text columns to numbers...")
le = LabelEncoder()
for col in ['Education', 'Employment', 'MaritalStatus']:
    df[col] = le.fit_transform(df[col])
    print(f"   → {col} encoded")


# ── 6. PREPARE DATA ───────────────────────────────────────────
X = df.drop('Default', axis=1)   # All columns except target
y = df['Default']                  # Target column

# Train-Test Split: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# stratify=y ensures both train and test have same % of defaulters

print(f"\n✂️  Data Split:")
print(f"   Training : {X_train.shape[0]} students")
print(f"   Testing  : {X_test.shape[0]} students")


# ── 7. HANDLE IMBALANCED DATA WITH SMOTE ─────────────────────
print(f"\n⚖️  Handling Imbalanced Data with SMOTE...")
print(f"   Before SMOTE → Repay: {sum(y_train==0)} | Default: {sum(y_train==1)}")

smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"   After  SMOTE → Repay: {sum(y_train_balanced==0)} | Default: {sum(y_train_balanced==1)}")
print(f"   ✅ Now both classes are equal! Model will learn fairly.")


# ── 8. FEATURE SCALING ───────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled  = scaler.transform(X_test)


# ── 9. TRAIN 3 MODELS ────────────────────────────────────────
print("\n🤖 Training 3 Models independently on same data...")
print("   (All 3 work independently — like 3 different bank experts!)\n")

models = {
    "Logistic Regression" : LogisticRegression(random_state=42, max_iter=1000),
    "Random Forest"       : RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost"             : XGBClassifier(random_state=42, eval_metric='logloss',
                                           verbosity=0)
}

results  = []
trained_models = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train_balanced)
    y_pred      = model.predict(X_test_scaled)
    y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]

    acc       = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    auc       = roc_auc_score(y_test, y_pred_prob)

    results.append({
        'Model'    : name,
        'Accuracy' : round(acc, 4),
        'Precision': round(precision, 4),
        'Recall'   : round(recall, 4),
        'F1 Score' : round(f1, 4),
        'ROC-AUC'  : round(auc, 4)
    })
    trained_models[name] = (model, y_pred, y_pred_prob)
    print(f"   ✔ {name:<25} | F1: {f1:.4f} | Recall: {recall:.4f} | AUC: {auc:.4f}")


# ── 10. RESULTS COMPARISON ───────────────────────────────────
results_df = pd.DataFrame(results).sort_values('ROC-AUC', ascending=False)

print("\n" + "=" * 65)
print("   MODEL PERFORMANCE COMPARISON (sorted by ROC-AUC)")
print("=" * 65)
print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]['Model']
best_auc        = results_df.iloc[0]['ROC-AUC']
best_recall     = results_df.iloc[0]['Recall']
print(f"\n🏆 Best Model : {best_model_name}")
print(f"   ROC-AUC    : {best_auc} (higher = better at separating defaulters)")
print(f"   Recall     : {best_recall} (out of all actual defaulters, caught this many)")


# ── 11. MODEL COMPARISON CHART ───────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(20, 6))
fig.suptitle('Model Comparison — Student Loan Default Predictor',
             fontsize=14, fontweight='bold')

metrics = ['Accuracy', 'F1 Score', 'Recall', 'ROC-AUC']
colors  = ['#2196F3', '#4CAF50', '#F44336']

for ax, metric in zip(axes, metrics):
    bars = ax.bar(results_df['Model'], results_df[metric], color=colors, edgecolor='black')
    ax.set_title(metric, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Score')
    ax.tick_params(axis='x', rotation=15)
    for bar, val in zip(bars, results_df[metric]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontweight='bold', fontsize=10)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print("\n✅ Model comparison chart saved as 'model_comparison.png'")


# ── 12. CONFUSION MATRIX (Best Model) ────────────────────────
best_model_obj, best_y_pred, best_y_prob = trained_models[best_model_name]
cm = confusion_matrix(y_test, best_y_pred)

plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Predicted: Repay', 'Predicted: Default'],
            yticklabels=['Actual: Repay', 'Actual: Default'],
            linewidths=2, linecolor='white', annot_kws={'size': 14})
plt.title(f'Confusion Matrix — {best_model_name}', fontweight='bold', fontsize=13)
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

tn, fp, fn, tp = cm.ravel()
print(f"\n📊 Confusion Matrix — {best_model_name}:")
print(f"   ✅ Correctly identified repayers (TN)   : {tn}")
print(f"   ✅ Correctly caught defaulters (TP)     : {tp}")
print(f"   ❌ Missed defaulters — dangerous! (FN)  : {fn}")
print(f"   ⚠️  Wrongly flagged repayers (FP)        : {fp}")
print("   ✅ Confusion matrix saved as 'confusion_matrix.png'")


# ── 13. ROC CURVE ─────────────────────────────────────────────
plt.figure(figsize=(8, 6))
colors_roc = ['#2196F3', '#4CAF50', '#F44336']

for (name, (model, y_pred, y_prob)), color in zip(trained_models.items(), colors_roc):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_val      = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {auc_val:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Guessing (AUC = 0.5)')
plt.fill_between([0, 1], [0, 1], alpha=0.05, color='gray')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate (Recall)', fontsize=12)
plt.title('ROC Curve — All 3 Models Compared', fontweight='bold', fontsize=13)
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ ROC curve saved as 'roc_curve.png'")


# ── 14. FEATURE IMPORTANCE ────────────────────────────────────
# Using Random Forest for feature importance
rf_model = trained_models["Random Forest"][0]
feat_imp  = pd.Series(rf_model.feature_importances_,
                       index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(12, 5))
colors_fi = ['#F44336' if i < 3 else '#2196F3' for i in range(len(feat_imp))]
feat_imp.plot(kind='bar', color=colors_fi, edgecolor='black')
plt.title('Feature Importance — Top Risk Factors for Loan Default',
          fontweight='bold', fontsize=13)
plt.ylabel('Importance Score')
plt.xticks(rotation=45, ha='right')
plt.axhline(y=feat_imp.mean(), color='orange', linestyle='--',
            label=f'Average importance = {feat_imp.mean():.3f}')
plt.legend()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n🌟 Top 3 Risk Factors for Loan Default:")
for i, (feat, score) in enumerate(feat_imp.head(3).items(), 1):
    print(f"   {i}. {feat:<25} → Importance: {score:.4f}")
print("✅ Feature importance saved as 'feature_importance.png'")


# ── 15. PREDICT ON A NEW STUDENT ─────────────────────────────
print("\n" + "=" * 60)
print("   SAMPLE PREDICTION — NEW STUDENT APPLICATION")
print("=" * 60)

# Create a sample student profile
sample_student = pd.DataFrame([{
    'Age'              : 23,
    'Income'           : 25000,       # ₹25,000/year — low income
    'LoanAmount'       : 800000,      # ₹8,00,000 loan — very high
    'LoanTerm'         : 36,
    'CreditScore'      : 420,         # Low credit score — red flag!
    'Education'        : 0,           # Graduate
    'Employment'       : 2,           # Unemployed — red flag!
    'MaritalStatus'    : 0,           # Single
    'Dependents'       : 3,           # 3 dependents — burden
    'CoSigner'         : 0,           # No cosigner — red flag!
    'LoanToIncomeRatio': 800000/25000,
    'MonthlyEMI'       : 800000/36,
    'EMIBurden'        : (800000/36) / (25000/12)
}])

sample_scaled  = scaler.transform(sample_student)
prediction     = best_model_obj.predict(sample_scaled)[0]
probability    = best_model_obj.predict_proba(sample_scaled)[0][1]

print(f"\n   Student Profile:")
print(f"   → Age: 23 | Income: ₹25,000 | Loan: ₹8,00,000")
print(f"   → Credit Score: 420 (LOW) | Employment: Unemployed")
print(f"   → Dependents: 3 | CoSigner: No")
print(f"\n   Prediction  : {'❌ WILL DEFAULT' if prediction == 1 else '✅ WILL REPAY'}")
print(f"   Probability : {probability*100:.1f}% chance of defaulting")
print(f"\n   Bank Decision: {'🚫 REJECT LOAN APPLICATION' if prediction == 1 else '✅ APPROVE LOAN APPLICATION'}")

# print("\n" + "=" * 60)
# print("   🎉 PROJECT COMPLETE!")
# print("=" * 60)
# print("\n   Files saved:")
# print("   → eda_plots.png")
# print("   → model_comparison.png")
# print("   → confusion_matrix.png")
# print("   → roc_curve.png")
# print("   → feature_importance.png")
