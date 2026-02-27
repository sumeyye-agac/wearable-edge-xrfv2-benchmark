from xrfv2_edge_tal.reproduce import extract_run_dir


def test_extract_run_dir_train_output() -> None:
    text = "Event training run dir: runs/20260227_021605_0bc9e9f1\nProfile: earbuds_glasses\n"
    assert extract_run_dir(text) == "runs/20260227_021605_0bc9e9f1"


def test_extract_run_dir_calibration_output() -> None:
    text = "Event calibration run dir: runs/20260227_030614_5a32e2cf\n"
    assert extract_run_dir(text) == "runs/20260227_030614_5a32e2cf"
