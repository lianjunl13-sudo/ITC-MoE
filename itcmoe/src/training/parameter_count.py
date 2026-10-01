def parameter_count(shape, ranks):
    e, o, i = shape
    re, ro, ri = ranks
    return re * ro * ri + e * re + o * ro + i * ri
