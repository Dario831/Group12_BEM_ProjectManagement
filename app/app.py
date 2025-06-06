# app.py

from flask import Flask, render_template, request
import torch
import joblib
import pandas as pd
from pathlib import Path
from catboost import CatBoostClassifier, Pool  # <-- import Pool here

# ----------------------------------------------------------------
# 1) IMPORT (or define) your three FeedForward classes
# ----------------------------------------------------------------

# ––– (a) Performance classifier (from models/FeedForward.py) –––
from models.FeedForward import FeedForward as ClassifierNet

# ––– (b) Salary regressor (MonthSalary training) –––
class FeedForwardRegressor(torch.nn.Module):
    def __init__(self, d_in, hidden):
        super().__init__()
        layers, prev = [], d_in
        for h in hidden:
            layers += [torch.nn.Linear(prev, h), torch.nn.ReLU()]
            prev = h
        layers.append(torch.nn.Linear(prev, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)

# ––– (c) WorkHours regressor (WorkHoursRegressor training) –––
class WorkHoursNet(torch.nn.Module):
    def __init__(self, d_in, hidden):
        super().__init__()
        layers, prev = [], d_in
        for h in hidden:
            layers += [
                torch.nn.Linear(prev, h),
                torch.nn.BatchNorm1d(h),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.25)
            ]
            prev = h
        layers.append(torch.nn.Linear(prev, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


# ----------------------------------------------------------------
# 2) FLASK SETUP
# ----------------------------------------------------------------

app = Flask(__name__)

# ───────────────────────────────────────────────────────────────────────────────
# CONFIGURATION FOR PERFORMANCE CLASSIFIER
# ───────────────────────────────────────────────────────────────────────────────
CLASSIFIER_HIDDEN       = [512, 256, 128, 64]
CLASSIFIER_NUM_CLASSES  = 5
CLASSIFIER_PREPROC_PATH = "models/preprocessor_performance.joblib"
CLASSIFIER_MODEL_PATH   = "models/performance.pt"

# ───────────────────────────────────────────────────────────────────────────────
# CONFIGURATION FOR SALARY REGRESSOR
# ───────────────────────────────────────────────────────────────────────────────
SALARY_HIDDEN       = [512, 256, 128, 64]
SALARY_PREPROC_PATH = "models/preprocessor_monthly_salary.joblib"
SALARY_MODEL_PATH   = "models/monthly_salary.pt"

# ───────────────────────────────────────────────────────────────────────────────
# CONFIGURATION FOR WORKHOURS REGRESSOR
# ───────────────────────────────────────────────────────────────────────────────
WH_HIDDEN       = [512, 256, 128, 64]
WH_PREPROC_PATH = "models/preprocessor_workhours.joblib"
WH_MODEL_PATH   = "models/workhours.pt"

# ───────────────────────────────────────────────────────────────────────────────
# CONFIGURATION FOR SATISFACTION CatBoost
# ───────────────────────────────────────────────────────────────────────────────
SATISFACTION_JOBLIB_PATH = "models/preeprocessor_satisfaction.joblib"
SATISFACTION_PT_PATH     = "models/satisfaction.pt"


# ───────────────────────────────────────────────────────────────────────────────
# 3) COMPUTE “DeptPerfMean” DICTIONARY FOR SATISFACTION FEATURE
# ───────────────────────────────────────────────────────────────────────────────
# (This block reads dataset.xlsx once to build dept_perf_mean; keep it if you still need it.
#  Otherwise, you can replace with loading a pre-saved dept_perf_mean.joblib.)
full_df = pd.read_excel("../final_project/dataset.xlsx")

cols_to_drop = [
    "Employee_ID", "Gender", "Projects_Handled", "Overtime_Hours",
    "Sick_Days", "Remote_Work_Frequency", "Training_Hours",
    "Promotions", "ID_progressive", "Resigned"
]
for c in cols_to_drop:
    if c in full_df.columns:
        full_df = full_df.drop(columns=c)

dept_perf_mean = (
    full_df.groupby("Department")["Performance_Score"]
    .mean()
    .to_dict()
)


# ───────────────────────────────────────────────────────────────────────────────
# 4) LOAD PERFORMANCE PREPROCESSOR & MODEL
# ───────────────────────────────────────────────────────────────────────────────
clf_preproc = joblib.load(CLASSIFIER_PREPROC_PATH)

clf_ohe     = clf_preproc.named_transformers_["cat"]
d_cat_clf   = sum(len(cats) for cats in clf_ohe.categories_)
clf_scaler  = clf_preproc.named_transformers_["num"]
d_num_clf   = clf_scaler.n_features_in_
d_in_clf    = d_cat_clf + d_num_clf
print(f">>> Performance preprocessor outputs {d_in_clf} features.")

classifier_net = ClassifierNet(
    d_in=d_in_clf,
    hidden=CLASSIFIER_HIDDEN,
    n_cls=CLASSIFIER_NUM_CLASSES
)
clf_state = torch.load(CLASSIFIER_MODEL_PATH, map_location=torch.device("cpu"))
classifier_net.load_state_dict(clf_state)
classifier_net.eval()


# ───────────────────────────────────────────────────────────────────────────────
# 5) LOAD SALARY PREPROCESSOR & MODEL
# ───────────────────────────────────────────────────────────────────────────────
sal_preproc = joblib.load(SALARY_PREPROC_PATH)

sal_ohe    = sal_preproc.named_transformers_["cat"]
d_cat_sal  = sum(len(cats) for cats in sal_ohe.categories_)
sal_scaler = sal_preproc.named_transformers_["num"]
d_num_sal  = sal_scaler.n_features_in_
d_in_sal   = d_cat_sal + d_num_sal
print(f">>> Salary preprocessor outputs {d_in_sal} features.")

salary_net = FeedForwardRegressor(
    d_in=d_in_sal,
    hidden=SALARY_HIDDEN
)
sal_state = torch.load(SALARY_MODEL_PATH, map_location=torch.device("cpu"))
salary_net.load_state_dict(sal_state)
salary_net.eval()


# ───────────────────────────────────────────────────────────────────────────────
# 6) LOAD WORKHOURS PREPROCESSOR & MODEL
# ───────────────────────────────────────────────────────────────────────────────
wh_preproc = joblib.load(WH_PREPROC_PATH)

wh_ohe    = wh_preproc.named_transformers_["cat"]
d_cat_wh  = sum(len(cats) for cats in wh_ohe.categories_)
wh_scaler = wh_preproc.named_transformers_["num"]
d_num_wh  = wh_scaler.n_features_in_
d_in_wh   = d_cat_wh + d_num_wh
print(f">>> WorkHours preprocessor outputs {d_in_wh} features.")

workhours_net = WorkHoursNet(
    d_in=d_in_wh,
    hidden=WH_HIDDEN
)
wh_state = torch.load(WH_MODEL_PATH, map_location=torch.device("cpu"))
workhours_net.load_state_dict(wh_state)
workhours_net.eval()


# ───────────────────────────────────────────────────────────────────────────────
# 7) LOAD SATISFACTION CatBoost MODEL
# ───────────────────────────────────────────────────────────────────────────────
# Load via joblib (this preserves cat_feature info)
satis_model: CatBoostClassifier = joblib.load(SATISFACTION_JOBLIB_PATH)
# (Alternatively, torch.load(SATISFACTION_PT_PATH) also works if you used torch.save.)


# ───────────────────────────────────────────────────────────────────────────────
# 8) FORM → MODEL‐INPUT HELPERS
# ───────────────────────────────────────────────────────────────────────────────

def form_to_input_classifier(form):
    data = {
        "Department":       [ form["department"] ],
        "Job_Title":        [ form["job_title"]  ],
        "Education_Level":  [ form["education_level"] ],
        "Age":              [ float(form["age"]) ],
        "Years_At_Company": [ float(form["experience"]) ],
        "Team_Size":        [ float(form["team_size"]) ],
        # Dummy salary during classification
        "Monthly_Salary":   [ 2000.0 ]
        # If your classifier preprocessor still expects “Performance_Score”, uncomment:
        # "Performance_Score": [0.0]
    }
    df = pd.DataFrame(data)
    Xp = clf_preproc.transform(df)
    return torch.tensor(Xp, dtype=torch.float32)


def form_to_input_salary(form):
    data = {
        "Department":       [ form["department"] ],
        "Job_Title":        [ form["job_title"]  ],
        "Education_Level":  [ form["education_level"] ],
        "Age":              [ float(form["age"]) ],
        "Years_At_Company": [ float(form["experience"]) ],
        "Team_Size":        [ float(form["team_size"]) ],
        "Performance_Score": 3
    }
    df = pd.DataFrame(data)
    Xp = sal_preproc.transform(df)
    return torch.tensor(Xp, dtype=torch.float32)


def form_to_input_workhours(form, pred_salary: float, pred_perf: int):
    data = {
        "Department":       [ form["department"] ],
        "Job_Title":        [ form["job_title"]  ],
        "Education_Level":  [ form["education_level"] ],
        "Age":              [ float(form["age"]) ],
        "Years_At_Company": [ float(form["experience"]) ],
        "Team_Size":        [ float(form["team_size"]) ],
        "Monthly_Salary":   [ float(pred_salary)        ],
        "Performance_Score":[ float(pred_perf)          ]
    }
    df = pd.DataFrame(data)
    Xp = wh_preproc.transform(df)
    return torch.tensor(Xp, dtype=torch.float32)


def form_to_input_satisfaction(
    form,
    pred_salary: float,
    pred_perf:   int,
    pred_workhours: float
):
    dept = form["department"]
    dept_mean = dept_perf_mean.get(dept, 0.0)

    data = {
        "Department":         [ dept ],
        "Job_Title":          [ form["job_title"] ],
        "Education_Level":    [ form["education_level"] ],
        "Age":                [ float(form["age"]) ],
        "Years_At_Company":   [ float(form["experience"]) ],
        "Team_Size":          [ float(form["team_size"]) ],
        "Monthly_Salary":     [ float(pred_salary)    ],
        "Performance_Score":  [ float(pred_perf)      ],
        "Work_Hours_Per_Week":[ float(pred_workhours)],
        "DeptPerfMean":       [ float(dept_mean)      ]
    }
    df = pd.DataFrame(data)
    return df  # we’ll wrap this in Pool(...) below


# ───────────────────────────────────────────────────────────────────────────────
# 9) FLASK ROUTES
# ───────────────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET", "POST"])
def form_view():
    if request.method == "POST":
        form_data = request.form
        age = int(form_data["age"])
        experience = int(form_data["experience"])
        if experience > age:
            return render_template("form.html", error="Years of experience cannot be greater than age.")
        # ——— (a) Performance classifier ———
        x_clf = form_to_input_classifier(form_data)
        with torch.no_grad():
            logits     = classifier_net(x_clf)
            pred_class = logits.argmax(dim=1).item() + 1

        # ——— (b) Salary regressor ———
        x_sal = form_to_input_salary(form_data)
        with torch.no_grad():
            sal_pred_tensor  = salary_net(x_sal)
            predicted_salary = sal_pred_tensor.item()

        # ——— (c) WorkHours regressor ———
        x_wh = form_to_input_workhours(
            form_data,
            pred_salary = predicted_salary,
            pred_perf   = pred_class
        )
        with torch.no_grad():
            wh_pred_tensor      = workhours_net(x_wh)
            predicted_workhours = wh_pred_tensor.item()

        # ——— (d) Satisfaction CatBoost ———
        satis_df = form_to_input_satisfaction(
            form_data,
            pred_salary      = predicted_salary,
            pred_perf        = pred_class,
            pred_workhours   = predicted_workhours
        )

        # Tell CatBoost which columns are categorical:
        # In training you used cat_cols = ["Department","Job_Title","Education_Level"]
        cat_cols   = ["Department", "Job_Title", "Education_Level"]
        cat_indices = [satis_df.columns.get_loc(c) for c in cat_cols]

        # Build a Pool so that CatBoost knows exactly which columns are categorical:
        satis_pool = Pool(data=satis_df, cat_features=cat_indices)

        # Now predict:
        pred_satis = satis_model.predict(satis_pool)[0]  # "slightly"/"moderately"/"very"

        # Render all four predictions
        return render_template(
            "result.html",
            data                   = form_data,
            performance_class      = pred_class,
            predicted_salary       = predicted_salary,
            predicted_workhours    = predicted_workhours,
            predicted_satisfaction = pred_satis
        )

    return render_template("form.html")


if __name__ == "__main__":
    app.run(debug=True)
