def test_joint_dice_weighting():
    full = 0.8
    missing = 0.6
    joint = 0.5 * full + 0.5 * missing
    assert joint == 0.7


def test_best_joint_not_full_only():
    best_full = 0.9
    best_missing = 0.1
    full = 0.95
    missing = 0.05
    joint = 0.5 * full + 0.5 * missing
    assert joint < full
    assert joint > best_missing
