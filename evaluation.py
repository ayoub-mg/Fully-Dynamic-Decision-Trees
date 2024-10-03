from typing import List

def precision_recall(expected_results: List[bool], actual_results: List[bool]) -> (float, float):
    """Compute the precision and recall of a series of predictions

    Parameters
    ----------
        expected_results : List[bool]
            The true results, that is the results that the predictor
            should have find.
        actual_results : List[bool]
            The predicted results, that have to be evaluated.

    Returns
    -------
        float
            The precision of the predicted results.
        float
            The recall of the predicted results.
    """
    TP,FP,FN=0,0,0
    for i in range(len(expected_results)) :
        TP+=(expected_results[i] and actual_results[i])
        FP+=(not expected_results[i] and actual_results[i])
        FN+=(expected_results[i] and not actual_results[i])

    p = TP / (TP+FP) if TP+FP > 0 else 0.0
    r = TP / (TP+FN) if TP+FN > 0 else 0.0

    return p,r


def F1_score(expected_results: List[bool], actual_results: List[bool]) -> float:
    """Compute the F1-score of a series of predictions

    Parameters
    ----------
        expected_results : List[bool]
            The true results, that is the results that the predictor
            should have find.
        actual_results : List[bool]
            The predicted results, that have to be evaluated.

    Returns
    -------
        float
            The F1-score of the predicted results.
    """
    p,r=precision_recall(expected_results,actual_results)
    if p+r==0 :
        return 0.
    f1_score = 2 * (p * r)/(p + r)
    return f1_score

