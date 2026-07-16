from c4fairness.result_viz import classify_column


def test_sensitive_label_override():
    fam, kind, lbl = classify_column("race_gap", sensitive_labels={"race": "Ethnicity"})
    assert fam == "sensitive" and kind == "value" and lbl == "Ethnicity gap"


def test_sensitive_label_override_pvalue_and_cat():
    assert classify_column("race_gap_sig", sensitive_labels={"race": "Ethnicity"})[2] == "Ethnicity gap sig."
    assert classify_column("race_cat", sensitive_labels={"race": "Ethnicity"})[2] == "Ethnicity cat."


def test_sensitive_label_defaults_to_column_name():
    assert classify_column("race_gap")[2] == "race gap"


def test_error_label_unaffected():
    assert classify_column("error_gap", error_label="FP Rate")[2] == "FP Rate gap"
