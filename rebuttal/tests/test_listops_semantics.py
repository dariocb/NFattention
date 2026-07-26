from rebuttal._shared.datasets import evaluate_listops_expression


def test_listops_evaluator_matches_official_operations():
    assert evaluate_listops_expression("[MIN 8 3 5 X") == 3
    assert evaluate_listops_expression("[MAX 8 3 5 X") == 8
    assert evaluate_listops_expression("[SM 8 7 6 X") == 1
    # Official ListOps uses int(np.median(...)), including truncation for an
    # even number of arguments.
    assert evaluate_listops_expression("[MED 2 3 X") == 2
    assert evaluate_listops_expression("[MAX [MIN 8 3 X [SM 7 8 X X") == 5
