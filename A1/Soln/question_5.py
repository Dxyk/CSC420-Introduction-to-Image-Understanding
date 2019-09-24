import numpy as np


# ==================== Question 5 ====================
def is_separable_filter(h: np.ndarray) -> bool:
    """
    Return True if the given filter h is separable, false otherwise.
    If the filter h is separable, then also print the corresponding horizontal
    and vertical filters.

    :param h: the given filter
    :return: True if the filter is separable, False otherwise
    """
    u, s, v = np.linalg.svd(h, full_matrices=True, compute_uv=True)
    # since we're using float instead of fractions, we will use
    # a relatively small epsilon (e^{-12}) to check for rank
    k = np.nonzero(s < 1e-12)[0][0]
    if k == 1:
        # Use np slicing instead of indexing to retain matrix shape
        vertical_filter = u[:, :1] * s[0] ** .5
        horizontal_filter = v[:1, :] * s[0] ** .5
        print(vertical_filter)
        print(horizontal_filter)
    return k == 1


if __name__ == '__main__':
    print("{0} Question 5 {0}".format("=" * 20))
    separable_filter = np.asarray([[1, 2, 1],
                                   [2, 4, 2],
                                   [1, 2, 1]]) / 16
    inseparable_filter = np.asarray([[0, -1, 0],
                                     [-1, 5, -1],
                                     [0, -1, 0]])
    print("{0} Separable {0}".format("=" * 10))
    print(is_separable_filter(separable_filter))
    print("{0} Inseparable {0}".format("=" * 10))
    print(is_separable_filter(inseparable_filter))