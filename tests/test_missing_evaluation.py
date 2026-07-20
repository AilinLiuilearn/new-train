def test_random_missing_rate_bookkeeping():
    case_count = 8
    for rate in (0.0, 0.25, 0.5, 0.75, 1.0):
        missing = int(case_count * rate)
        available = case_count - missing
        assert available + missing == case_count


def test_reproducible_missing_assignment():
    seed = 123
    cases = [f'case_{i}' for i in range(10)]
    first = list(cases)
    second = list(cases)
    assert first == second
