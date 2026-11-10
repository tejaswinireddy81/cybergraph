"""[fp_investigation] Read-only diagnostic of the autoencoder branch and the fusion layer. Inference on existing models only."""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.chdir(ROOT); sys.path.insert(0, ROOT)
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from train_xgboost_ae import FEATURE_COLS
from modules.trusted_model_loader import safe_load_joblib

df = pd.read_csv("NF-UNSW-NB15-v3.csv", usecols=FEATURE_COLS + ["Attack", "FLOW_START_MILLISECONDS"])
df = df.sort_values("FLOW_START_MILLISECONDS").reset_index(drop=True)
y = np.where(df["Attack"].astype(str).str.strip().str.lower() == "benign", 0, 1)
X = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
n = int(len(X) * 0.75)
Xte, yte = X.iloc[n:], y[n:]

sc = safe_load_joblib("models/scaler.pkl")
ae = safe_load_joblib("models/ae_model.pkl")
fu = safe_load_joblib("models/fusion_model.pkl")
art = json.load(open("models/xgboost_ae_artifacts.json"))
mse_max = art["ae_mse_max"]; thr = art["ae_threshold"]
m = xgb.XGBClassifier(); m.load_model("models/xgboost_model.json")

Xs = sc.transform(Xte).astype(np.float32)
mse = np.mean((ae.predict(Xs) - Xs) ** 2, axis=1)
xp = m.predict_proba(Xte)[:, 1]
norm = np.minimum(mse / (mse_max + 1e-9), 1.0)

q = lambda a: np.round(np.quantile(a, [.01, .25, .5, .75, .95, .99, .999]), 5).tolist()
print("== AE reconstruction MSE on chronological test split (existing model)")
print("benign quantiles [1,25,50,75,95,99,99.9%]:", q(mse[yte == 0]))
print("attack quantiles                        :", q(mse[yte == 1]))
print(f"AE alone: ROC-AUC={roc_auc_score(yte, mse):.4f}  PR-AUC={average_precision_score(yte, mse):.4f}")
print(f"saved ae_threshold={thr:.4f} (99th pct of benign val)  ae_mse_max={mse_max:.2f}")
print(f"fraction of test flows with MSE>threshold: benign {np.mean(mse[yte==0] > thr):.4f}  attack {np.mean(mse[yte==1] > thr):.4f}")
print("\n== ae_norm = min(mse/ae_mse_max, 1) (what the fusion layer actually receives)")
print("ae_norm quantiles benign:", q(norm[yte == 0]))
print("ae_norm quantiles attack:", q(norm[yte == 1]))
print(f"share of ae_norm < 0.01: {np.mean(norm < 0.01):.3f} | share == 1.0 (clipped): {np.mean(norm >= 1.0):.4f}")
print(f"corr(xgb_prob, ae_norm) = {np.corrcoef(xp, norm)[0,1]:.3f}")

print("\n== Fusion layer")
print("saved LR coef [xgb, ae_norm]:", np.round(fu.coef_[0], 3), "intercept", np.round(fu.intercept_, 3))
fp = fu.predict_proba(np.column_stack([xp, norm]))[:, 1]
print(f"fusion  ROC-AUC={roc_auc_score(yte, fp):.4f} PR-AUC={average_precision_score(yte, fp):.4f}")
print(f"xgb only ROC-AUC={roc_auc_score(yte, xp):.4f} PR-AUC={average_precision_score(yte, xp):.4f}")
# why is the weight ~0? ask what a LR would learn with alternative AE encodings on the SAME data (throwaway)
cut = int(len(xp) * .7)
def fit_eval(feats, name):
    lr = LogisticRegression(class_weight="balanced", max_iter=2000).fit(feats[:cut], yte[:cut])
    p = lr.predict_proba(feats[cut:])[:, 1]
    print(f"  {name:34s} coef={np.round(lr.coef_[0],2)}  holdout ROC-AUC={roc_auc_score(yte[cut:], p):.4f}")
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
print("throwaway re-fits (same data, NOT saved):")
fit_eval(np.column_stack([xp]), "xgb only")
fit_eval(np.column_stack([xp, norm]), "xgb + ae_norm (as shipped)")
fit_eval(np.column_stack([logit(xp), np.log1p(mse)]), "logit(xgb) + log1p(mse)")
fit_eval(np.column_stack([logit(xp), (np.log1p(mse) - np.log1p(mse).mean()) / np.log1p(mse).std()]), "logit(xgb) + standardised log mse")

# live traffic view (needs outputs/fp_live_flows.json from scripts/fp_investigation/02)
lp = os.path.join(ROOT, "outputs", "fp_live_flows.json")
if os.path.exists(lp):
    live = json.load(open(lp))
    LV = np.array([r["vec"] for r in live], dtype=np.float32)
    kind = np.array(["IGMP" if int(r["PROTOCOL"]) == 2 else "UDP-DNS" if int(r["PROTOCOL"]) == 17 and int(r["L4_DST_PORT"]) == 53
                     else "UDP-other" if int(r["PROTOCOL"]) == 17 else "TCP" for r in live])
    ls = sc.transform(pd.DataFrame(LV, columns=FEATURE_COLS)).astype(np.float32)
    lm = np.mean((ae.predict(ls) - ls) ** 2, axis=1)
    print("\n== AE on LIVE flows (MSE vs saved threshold %.3f; dataset benign p99=%.3f)" % (thr, np.quantile(mse[yte == 0], .99)))
    for k in ("TCP", "UDP-DNS", "UDP-other", "IGMP"):
        s = lm[kind == k]
        if len(s): print(f"  {k:10s} n={len(s):3d} median MSE={np.median(s):9.3f}  share>threshold={np.mean(s > thr):.2f}")
