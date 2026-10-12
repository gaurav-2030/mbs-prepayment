// node report/build_deck.js  -> outputs/MBS_Prepayment_Deck.pptx (reads outputs/results.json)
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const R = JSON.parse(fs.readFileSync("outputs/results.json"));
const { info, results: res } = R;
const best = res.reduce((a, b) => (b.pr_auc > a.pr_auc ? b : a));
const bestF1 = res.reduce((a, b) => (b.f1 > a.f1 ? b : a));
const syn = info.data === "synthetic";
const NAVY = "1A365D", BLUE = "2B6CB0", LIGHT = "EDF2F7", INK = "1A202C", GREY = "4A5568";
const pres = new pptxgen(); pres.layout = "LAYOUT_16x9";
pres.title = "MBS Prepayment Prediction";

function slide(title, notes) {
  const s = pres.addSlide(); s.background = { color: "FFFFFF" };
  s.addShape(pres.ShapeType.rect, { x: 0, y: 0, w: 10, h: 0.9, fill: { color: NAVY }, line: { color: NAVY } });
  s.addText(title, { x: 0.5, y: 0.1, w: 9, h: 0.7, fontSize: 26, bold: true, color: "FFFFFF", fontFace: "Calibri", isTextBox: true, margin: 0 });
  if (notes) s.addNotes(notes);
  return s;
}
const bullets = (arr, o = {}) => arr.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < arr.length - 1, paraSpaceAfter: 8 } }));

// 1 title
let s = pres.addSlide(); s.background = { color: NAVY };
s.addText("Predicting Mortgage-Backed Securities Prepayment", { x: 0.6, y: 1.3, w: 8.8, h: 1.2, fontSize: 36, bold: true, color: "FFFFFF", fontFace: "Calibri", isTextBox: true });
s.addText("Using Machine Learning Methods", { x: 0.6, y: 2.5, w: 8.8, h: 0.6, fontSize: 24, color: "BEE3F8", fontFace: "Calibri", isTextBox: true });
s.addText("UE24CS352A Machine Learning - Mini-Project  |  Section B  |  Problem #55\nBhoomika Kshatriya (PES1UG24AM068)   |   Gaurav Girish (PES1UG24AM103)", { x: 0.6, y: 3.9, w: 8.8, h: 0.9, fontSize: 16, color: "FFFFFF", fontFace: "Calibri", isTextBox: true });
s.addNotes("Introduce team (Section B, problem 55) and that we follow the Guo/Boccio/Cameron study.");

// 2 problem
s = slide("Problem: will this loan prepay this month?", "Prepayment cuts MBS investor returns. Industry uses pool-level models; we do loan-level classification.");
s.addText(bullets([
  "MBS investors receive interest + principal from a pool of home loans",
  "Prepayment = borrower repays early -> interest stops -> lower returns",
  "Industry: pool-level statistical models with hand-picked factors",
  "Ours: loan-level binary classification (1 = prepaid this month)",
  "Compare Logistic, SVM, GDA and a Neural Network"]), { x: 0.6, y: 1.2, w: 8.8, h: 3.8, fontSize: 20, color: INK, fontFace: "Calibri", valign: "top", isTextBox: true });

// 3 dataset
s = slide("Dataset and features", "Freddie Mac SF loan-level data: origination + monthly performance merged on loan id. Label from zero-balance code 01.");
s.addText(bullets([
  "Freddie Mac Single-Family Loan-Level data: ~45.5M loans, 1999-2020",
  "Origination (static): credit score, DTI, LTV, rate, UPB, occupancy, purpose",
  "Performance (monthly): current UPB, loan age, rate, zero-balance code",
  "Macro: mortgage rate, house-price change, unemployment",
  "Label = 1 if zero-balance code 01 (voluntary payoff)"]), { x: 0.5, y: 1.1, w: 5.4, h: 3.9, fontSize: 16, color: INK, fontFace: "Calibri", valign: "top", isTextBox: true });
const stat = (x, y, big, small) => { s.addShape(pres.ShapeType.roundRect, { x, y, w: 3.2, h: 1.15, fill: { color: LIGHT }, line: { color: LIGHT }, rectRadius: 0.1 });
  s.addText(big, { x, y: y + 0.08, w: 3.2, h: 0.65, align: "center", fontSize: 28, bold: true, color: BLUE, fontFace: "Calibri", isTextBox: true });
  s.addText(small, { x, y: y + 0.72, w: 3.2, h: 0.35, align: "center", fontSize: 12, color: GREY, fontFace: "Calibri", isTextBox: true }); };
stat(6.3, 1.2, info.rows.toLocaleString(), "loan-month rows" + (syn ? " (synthetic)" : ""));
stat(6.3, 2.55, info.imbalance, "class imbalance");
stat(6.3, 3.9, String(info.features), "features after encoding");

// 4 data pipeline
s = slide("Data pipeline: leakage-safe and tested", "Why each step exists: 30-year filter (code 01 also means matured), start-of-month balance (UPB is 0 in the payoff month), split by loan (no loan in two splits). Tests in tests/ check every one of these.");
s.addText(bullets([
  "Merge origination + monthly performance files on loan id",
  "Only 30-year loans: code 01 means prepaid OR matured",
  "Leakage fix: balance is 0 in the payoff month, so we use the start-of-month balance",
  "Split 60/20/20 by loan: no loan appears in two splits",
  "Macro features from FRED (src/macro.py): 30-yr mortgage rate, FHFA house prices, BLS unemployment",
  "Missing codes 999 / 9999 -> median; categoricals one-hot encoded"]), { x: 0.5, y: 1.1, w: 5.6, h: 4.2, fontSize: 15, color: INK, fontFace: "Calibri", valign: "top", isTextBox: true });
const nTests = ["tests/test_data.py", "tests/test_macro.py"].map(f => (fs.readFileSync(f, "utf8").match(/^def test_/gm) || []).length).reduce((a, b) => a + b, 0);
stat(6.3, 1.2, "60 / 20 / 20", "train / val / test, grouped by loan");
stat(6.3, 2.55, String(nTests), "automated tests (python -m pytest)");
stat(6.3, 3.9, "3", "public macro series joined by month");

// 5 approach
s = slide("Approach: four model families", "Explain each briefly; note SVM subset and NN oversampling.");
const cards = [["Logistic", "Baseline, L1 (selection), L2 (multicollinearity)"], ["SVM", "Poly / RBF / sigmoid kernels on a training subset"],
  ["GDA", "Linear (shared cov.) vs Quadratic (per-class cov.)"], ["Neural Net", "MLP 128-256-128, ReLU, Adam; positives oversampled 10x"]];
cards.forEach((c, i) => { const x = 0.5 + (i % 2) * 4.6, y = 1.15 + Math.floor(i / 2) * 1.85;
  s.addShape(pres.ShapeType.roundRect, { x, y, w: 4.3, h: 1.65, fill: { color: LIGHT }, line: { color: LIGHT }, rectRadius: 0.1 });
  s.addText(c[0], { x: x + 0.2, y: y + 0.1, w: 3.9, h: 0.5, fontSize: 20, bold: true, color: NAVY, fontFace: "Calibri", isTextBox: true });
  s.addText(c[1], { x: x + 0.2, y: y + 0.65, w: 3.9, h: 0.9, fontSize: 14, color: INK, fontFace: "Calibri", valign: "top", isTextBox: true }); });
s.addText("Split 60/20/20 grouped by loan  |  threshold tuned on validation (max F1)  |  PR-AUC for imbalance", { x: 0.5, y: 4.85, w: 9, h: 0.4, fontSize: 13, color: GREY, fontFace: "Calibri", isTextBox: true });

// 6 results chart
s = slide("Results: F1 and PR-AUC on the test set", `${syn ? "These are synthetic-data results - replace by re-running on Freddie Mac. " : ""}Accuracy alone is misleading at this imbalance.`);
// chart is an image (outputs/metrics_bar.png) so it renders identically in PowerPoint, Keynote and Google Slides
s.addImage({ path: "outputs/metrics_bar.png", x: 0.4, y: 1.0, w: 9.2, h: 3.9 });
s.addText(`${syn ? "Synthetic data. " : ""}Best PR-AUC: ${best.model} (${best.pr_auc.toFixed(3)}); best F1: ${bestF1.model} (${bestF1.f1.toFixed(3)})`,
  { x: 0.5, y: 4.95, w: 9, h: 0.4, fontSize: 13, bold: true, color: NAVY, fontFace: "Calibri", isTextBox: true });

// 7 table
s = slide("Detailed metrics", "Walk through precision vs recall trade-off.");
const hdr = ["Model", "Acc.", "Prec.", "Recall", "F1", "ROC-AUC", "PR-AUC"].map(t => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: NAVY }, align: "center" } }));
const rows = [hdr, ...res.map(r => [r.model, r.accuracy.toFixed(4), r.precision.toFixed(3), r.recall_prepay.toFixed(3), r.f1.toFixed(3), r.roc_auc.toFixed(3), r.pr_auc.toFixed(3)]
  .map((t, i) => ({ text: t, options: { align: i ? "center" : "left", color: INK } })))];
s.addTable(rows, { x: 0.5, y: 1.1, w: 9, colW: [2.6, 1, 1, 1, 0.9, 1.2, 1.3], fontSize: 12, fontFace: "Calibri", border: { type: "solid", color: "CBD5E0", pt: 0.5 }, rowH: 0.34 });

// 8 insights
s = slide("Insights", "Rate incentive is the main driver; accuracy is misleading; leakage caveat.");
const top = fs.readFileSync("outputs/feature_importance.csv", "utf8").split("\n")[1].split(",")[0];
s.addText(bullets([
  `Top L1-logistic feature: ${top.replace(/_/g, " ")} - borrowers refinance when market rates fall below their rate`,
  `Accuracy hides failure: predicting 'never prepay' already scores ${((1 - info.prepay_rate) * 100).toFixed(1)}% at ${info.imbalance}`,
  "Regularisation barely moves logistic performance, but L1 prunes features",
  "Kernel SVMs limited to a subset; NN needs positive reweighting to detect prepayments",
  "Leakage fix: UPB is 0 in the payoff month, so we use start-of-month balance"]), { x: 0.5, y: 1.15, w: 5.2, h: 3.9, fontSize: 15, color: INK, fontFace: "Calibri", valign: "top", isTextBox: true });
s.addImage({ path: "outputs/feature_importance.png", x: 5.8, y: 1.3, w: 3.9, h: 2.6 });

// 9 EDA and tuning
s = slide("EDA and hyper-parameter tuning", "EDA: notebooks/eda.ipynb. Tuning: python -m src.tune - each value is chosen on the validation split, then scored once on test.");
const tt = fs.readFileSync("outputs/tuning_test.csv", "utf8").trim().split("\n").slice(1).map(l => l.split(","));
const tune = m => { const r = tt.filter(x => x[0] === m); return { cv: r[0][3], cs: +r[0][5], dv: r[1][3], ds: +r[1][5] }; };
const q = tune("GDA (Quadratic)"), l1 = tune("Logistic (L1)");
s.addText(bullets([
  "EDA: the refinancing incentive is the dominant, monotone driver; loan age rises then falls (seasoning, then burnout)",
  "Near-duplicate features (loan age vs months to maturity, r = -1) make QDA's covariance matrices singular",
  `QDA reg_param ${q.dv} -> ${q.cv} (chosen on validation): test PR-AUC ${q.ds.toFixed(3)} -> ${q.cs.toFixed(3)}; logistic C barely matters (${l1.ds.toFixed(3)} vs ${l1.cs.toFixed(3)})`]),
  { x: 0.5, y: 1.05, w: 9, h: 1.75, fontSize: 14, color: INK, fontFace: "Calibri", valign: "top", isTextBox: true });
s.addImage({ path: "outputs/tuning.png", x: 0.5, y: 2.85, w: 9, h: 2.35 });

// 10 demo
s = slide("Live demo", "Run: python -m src.demo --random 6 ; then a custom loan with --loan.");
s.addShape(pres.ShapeType.rect, { x: 0.5, y: 1.2, w: 9, h: 2.4, fill: { color: "1A202C" }, line: { color: "1A202C" } });
s.addText("$ python -m src.train --data synthetic\n$ python -m src.demo --random 6\n$ python -m src.demo --loan examples/loan.json\n    (credit 770, loan rate 7.0% vs market 4.5%, UPB 60k, age 48)",
  { x: 0.7, y: 1.3, w: 8.6, h: 2.2, fontSize: 14, color: "9AE6B4", fontFace: "Courier New", valign: "top", isTextBox: true });
s.addText("Shows: score, tuned threshold, predicted vs actual on held-out test loan-months", { x: 0.5, y: 3.9, w: 9, h: 0.5, fontSize: 16, color: GREY, fontFace: "Calibri", isTextBox: true });

// 11 conclusions
s = slide("Conclusions and future work", `Close with limitations and next steps. The matrix shows the best model (${best.model}) on the test set: it catches ${best.TP} of ${best.TP + best.FN} prepayments.`);
s.addText(bullets([
  "Prepayment recast as loan-level classification; 9 models compared fairly",
  `Best PR-AUC: ${best.model}; best F1: ${bestF1.model}`,
  "Imbalance-aware metrics + validation-tuned thresholds give honest comparison",
  "Future: time-based validation, ensemble of models, pool-level baseline"]), { x: 0.5, y: 1.1, w: 5.1, h: 3.2, fontSize: 16, color: INK, fontFace: "Calibri", valign: "top", isTextBox: true });
s.addImage({ path: "outputs/best_confusion.png", x: 5.8, y: 1.05, w: 3.9, h: 3.34 });
s.addText("Thank you - Questions?", { x: 0.5, y: 4.45, w: 5.1, h: 0.45, fontSize: 20, bold: true, color: BLUE, fontFace: "Calibri", isTextBox: true });
s.addText("Reference: C. Guo, J. Boccio, C. Cameron, \"Predicting Mortgage Backed Securities Prepayment Using Machine Learning Methods\", CS229 project report, Stanford University, 2021. cs229.stanford.edu/proj2021spr/report2/81995903.pdf",
  { x: 0.5, y: 4.95, w: 9.2, h: 0.5, fontSize: 9, italic: true, color: GREY, fontFace: "Calibri", isTextBox: true });

pres.writeFile({ fileName: "outputs/MBS_Prepayment_Deck.pptx" }).then(() => console.log("deck written"));
