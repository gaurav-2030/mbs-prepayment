"""Build the 2-page PDF write-up from outputs/results.json:  python report/build_report.py"""
import json
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

TEAM = "Team: Bhoomika Kshatriya (PES1UG24AM068)  |  Gaurav Girish (PES1UG24AM103)  |  Section: B  |  Problem #: 55"
R = json.load(open("outputs/results.json"))
info, res = R["info"], R["results"]
syn = info["data"] == "synthetic"
best = max(res, key=lambda r: r["pr_auc"])
bf1 = max(res, key=lambda r: r["f1"])

ss = getSampleStyleSheet()
B = ParagraphStyle("b", parent=ss["Normal"], fontSize=8.6, leading=10.8, spaceAfter=3)
H = ParagraphStyle("h", parent=ss["Heading2"], fontSize=10.5, spaceBefore=5, spaceAfter=2, textColor=colors.HexColor("#1a365d"))
T = ParagraphStyle("t", parent=ss["Title"], fontSize=14, spaceAfter=2)
S = ParagraphStyle("s", parent=B, alignment=1, textColor=colors.grey, fontSize=8)
bul = lambda t: Paragraph("&bull; " + t, ParagraphStyle("bl", parent=B, leftIndent=10, firstLineIndent=-7, spaceAfter=1.5))

doc = SimpleDocTemplate("outputs/PES1UG24AM068_PES1UG24AM103.pdf", pagesize=A4, leftMargin=1.6*cm, rightMargin=1.6*cm,
                        topMargin=1.3*cm, bottomMargin=1.2*cm, title="MBS Prepayment Prediction - Mini-Project Write-up")
st = [Paragraph("Predicting Mortgage-Backed Securities Prepayment Using Machine Learning", T),
      Paragraph("UE24CS352A Machine Learning - Mini-Project &nbsp;|&nbsp; " + TEAM, S)]

st += [Paragraph("1. Problem Statement", H),
       Paragraph("A residential mortgage-backed security (MBS) pays investors the interest and principal collected from a pool of home loans. "
                 "<b>Prepayment</b> - a borrower repaying part or all of a loan earlier than scheduled - stops future interest and cuts the "
                 "investor's return, so predicting it matters. Industry usually models prepayment at <i>pool</i> level with hand-specified "
                 "statistical models. We instead frame it as <b>loan-level binary classification</b>: given a loan's origination attributes, "
                 "its current state and the macro-economic environment in a given month, will it be prepaid <i>that month</i> (1) or not (0)? "
                 "We implement and compare logistic regression, SVMs, Gaussian Discriminant Analysis and a feed-forward neural network, "
                 "following the reference study by Guo, Boccio and Cameron.", B)]

drow = (f"<b>Run reported here:</b> {info['rows']:,} loan-month rows from {info['loans']:,} loans, {info['features']} features after one-hot encoding, "
        f"prepayment rate {info['prepay_rate']*100:.2f}% (imbalance {info['imbalance']}). Split 60/20/20 by <i>loan</i> "
        f"(train {info['train']:,} / val {info['val']:,} / test {info['test']:,}; {info['test_positives']} positive test rows).")
if syn:
    drow = ("<b>NOTE - synthetic data:</b> the numbers below come from the bundled synthetic loan-month generator (same schema as Freddie Mac, "
            "prepayment hazard driven by rate incentive, balance, age, credit score, LTV, DTI). They validate the pipeline and are not Freddie Mac results. " + drow)
st += [Paragraph("2. Dataset", H),
       Paragraph("<b>Source:</b> Freddie Mac Single-Family Loan-Level Dataset (~45.5M fixed-rate 30-year mortgages originated 1999-2020). "
                 "<i>Origination</i> files hold static fields (credit score, DTI, LTV/CLTV, original UPB and rate, occupancy, purpose, channel); "
                 "<i>monthly performance</i> files hold current UPB, loan age, months to maturity, current rate and the zero-balance code. "
                 "The two are merged on loan id; macro features (mortgage rate, house-price change, unemployment) are joined by month; "
                 "<b>label = 1</b> when the zero-balance code is 01 (voluntary payoff); because code 01 also covers maturity, only 30-year (360-month) loans are kept. Missing sentinels (999/9999) become NaN and are median-imputed; "
                 "categoricals are one-hot encoded; numeric features are standardised inside each model pipeline.", B),
       Paragraph(drow, B)]

st += [Paragraph("3. Approach and Implementation", H),
       bul("<b>Logistic regression</b> - baseline, L1 (feature selection) and L2 (multicollinearity) regularised."),
       bul("<b>SVM</b> - polynomial, RBF and sigmoid kernels; trained on a subset because the kernel matrix grows quadratically."),
       bul("<b>GDA</b> - linear (shared covariance, i.e. LDA) and quadratic (class-specific covariance, QDA)."),
       bul("<b>Neural network</b> - feed-forward MLP (128-256-128, ReLU, Adam, early stopping), adapted from the paper's 3-layer net; positives oversampled 10x in place of the paper's positive loss weight of 10."),
       bul(f"<b>Evaluation under imbalance</b> - accuracy is misleading at {info['imbalance']} (predicting 'never prepay' scores {(1 - info['prepay_rate']) * 100:.1f}%), so we also report precision, recall, F1, "
           "ROC-AUC and PR-AUC. Each model's decision threshold is tuned on the validation split (max F1) and applied once to the test split."),
       bul("<b>Code layout</b> - <font face='Courier'>src/data.py</font> (loader, synthetic generator, loan-grouped split), "
           "<font face='Courier'>src/models.py</font> (model zoo), <font face='Courier'>src/train.py</font> (train, evaluate, plots), "
           "<font face='Courier'>src/demo.py</font> (live scoring), <font face='Courier'>src/macro.py</font> (macro features from FRED), "
           "<font face='Courier'>src/tune.py</font> (hyper-parameter sweep), <font face='Courier'>tests/</font> (pytest suite for the data pipeline), "
           "<font face='Courier'>notebooks/eda.ipynb</font> (exploratory analysis), <font face='Courier'>report/</font> (this PDF and the deck).")]

cols = ["Model", "Acc.", "Prec.", "Recall", "F1", "ROC-AUC", "PR-AUC"]
rows = [cols] + [[r["model"], f"{r['accuracy']:.4f}", f"{r['precision']:.3f}", f"{r['recall_prepay']:.3f}",
                  f"{r['f1']:.3f}", f"{r['roc_auc']:.3f}", f"{r['pr_auc']:.3f}"] for r in res]
tb = Table(rows, colWidths=[4.2*cm] + [2.2*cm]*6, repeatRows=1)
tb.setStyle(TableStyle([("FONTSIZE", (0, 0), (-1, -1), 7.8), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a365d")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("GRID", (0, 0), (-1, -1), .25, colors.grey),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#edf2f7")]),
                        ("ALIGN", (1, 0), (-1, -1), "CENTER"), ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]))
st += [Paragraph("4. Results (test split)", H), tb, Spacer(1, 4)]
st += [Table([[Image("outputs/roc_pr.png", width=9.2*cm, height=3.7*cm), Image("outputs/feature_importance.png", width=7.4*cm, height=4.9*cm)]],
             colWidths=[9.4*cm, 7.8*cm])]

top = open("outputs/feature_importance.csv").read().splitlines()[1].split(",")[0]
tt = [l.split(",") for l in open("outputs/tuning_test.csv").read().splitlines()[1:]]
qda = [r for r in tt if r[0] == "GDA (Quadratic)"]  # [chosen on validation, current]
st += [Paragraph("5. Conclusions", H),
       bul(f"Best ranking quality (PR-AUC): <b>{best['model']}</b> ({best['pr_auc']:.3f}); best F1: <b>{bf1['model']}</b> ({bf1['f1']:.3f}). "
           "Regularisation barely changed logistic performance, but L1 shrinks the feature set."),
       bul(f"The most influential feature in the L1 model is <b>{top.replace('_', ' ')}</b>, consistent with finance theory (borrowers refinance when the market rate falls below their loan rate)."),
       bul("Under heavy imbalance, accuracy alone hides poor prepayment recall; PR-AUC / F1 and a tuned threshold give an honest comparison."),
       bul(f"<b>Tuning</b> (chosen on validation, scored once on test): QDA needs strong covariance shrinkage because some features are exact duplicates "
           f"(reg_param {qda[1][3]} to {qda[0][3]} lifts its test PR-AUC from {float(qda[1][5]):.3f} to {float(qda[0][5]):.3f}); logistic regression is insensitive to C, "
           "so the model ranking does not change."),
       bul("<b>Leakage control:</b> Freddie Mac reports current UPB = 0 in the payoff month, which reveals the label (the reference paper's top feature was current UPB). "
           "Our loader therefore uses the balance at the <i>start</i> of the month (previous month's UPB)."),
       bul("<b>Limitations / future work:</b> kernel SVMs see only a subset of the data; next steps are time-based validation, an ensemble of the models, "
           "and comparison with pool-level models."),
       Paragraph("<i>Reference: C. Guo, J. Boccio, C. Cameron, 'Predicting Mortgage Backed Securities Prepayment Using Machine Learning Methods', "
                 "CS229 project report, Stanford University, Spring 2021 (course project, not peer-reviewed), "
                 "https://cs229.stanford.edu/proj2021spr/report2/81995903.pdf. Code: https://github.com/gaurav-2030/mbs-prepayment.</i>", S)]
doc.build(st)
print("wrote outputs/PES1UG24AM068_PES1UG24AM103.pdf")
