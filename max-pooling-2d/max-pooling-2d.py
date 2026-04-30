def max_pooling_2d(X, pool_size):
    """
    Apply 2D max pooling with non-overlapping windows.
    """
    # Write code here
    out_h = len(X) // pool_size
    out_w = len(X[0]) // pool_size

    output = []

    for i in range(out_h):
        row = []
        for j in range(out_w):
            start_i = i * pool_size
            start_j = j * pool_size

            max_val = float('-inf')
            
            for a in range(pool_size):
                for b in range(pool_size):
                    val = X[start_i + a][start_j + b]
                    if val > max_val:
                        max_val = val

            row.append(max_val)
        output.append(row)
    return output