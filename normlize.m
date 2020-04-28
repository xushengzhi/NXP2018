function A = normlize(B)

max_value = max(max(B));
A = B - max_value;