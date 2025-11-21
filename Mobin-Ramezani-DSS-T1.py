import numpy as np

# --------------------- AHP ---------------------
def pairwise_matrix(n, comparisons):
    A = np.ones((n, n), dtype=float)
    for i, j, val in comparisons:
        A[i, j] = float(val)
        A[j, i] = 1.0 / float(val)
    return A

def ahp_weights(A, max_iter=100):
    A = np.array(A, dtype=float)
    n = A.shape[0]
    w_first = np.ones(n) / n
    for _ in range(max_iter):
        B = A @ A
        row_sums = B.sum(axis=1)
        w_new = row_sums / row_sums.sum()
        if np.round(w_new, 4).tolist() == np.round(w_first, 4).tolist():
            return w_new
        A = B
        w_first = w_new
    return w_first

# --------------------- TOPSIS ---------------------a
def topsis(X, weights, is_benefit):
    X = np.array(X, dtype=float)
    w = np.array(weights, dtype=float)
    is_benefit = np.array(is_benefit, dtype=bool)

    R = X / (np.sqrt((X ** 2).sum(axis=0)))
    V = R * w

    A_pos = np.where(is_benefit, V.max(axis=0), V.min(axis=0))
    A_neg = np.where(is_benefit, V.min(axis=0), V.max(axis=0))

    S_pos = np.sqrt(((V - A_pos) ** 2).sum(axis=1))
    S_neg = np.sqrt(((V - A_neg) ** 2).sum(axis=1))
    C = S_neg / (S_neg + S_pos)

    ranking = np.argsort(C)[::-1]
    return C, ranking

# --------------------- main ---------------------
if __name__ == "__main__":
    comparisons = [
        (0,1,1/5), (0,2,1/9), (0,3,7), (0,4,7),
        (1,2,1/9), (1,3,5), (1,4,5),
        (2,3,9), (2,4,9),
        (3,4,1),
    ]
    A = pairwise_matrix(5, comparisons)
    w = ahp_weights(A)

    print("Weights (AHP):")
    for i, weight in enumerate(w):
        print(f"C{i+1}: {weight:.2f}")

    # Price, Battery, CPU, Weight, Display
    X = np.array([
        [950, 8, 7.5, 1.6, 8],   # A
        [1200, 10, 9.0, 1.8, 9],   # B
        [800, 6, 6.5, 1.4, 7],   # C
        [1000, 9, 8.0, 1.5, 8.5], # D
        [1100, 7, 8.5, 1.9, 8],   # E
    ])
    is_benefit = np.array([False, True, True, False, True])

    C, ranking = topsis(X, w, is_benefit)

    alternatives = np.array(["A", "B", "C", "D", "E"])

    print("\nRelative Closeness (TOPSIS):")
    for i in range(len(C)):
        print(f"Laptop {alternatives[i]}: {C[i]:.2f}")

    print("\nRanking:")
    for rank, index in enumerate(ranking, start=1):
        print(f"Laptop {alternatives[index]}: {C[index]:.2f} -> {rank} رتبه")

