import warnings


def check_and_warn_input_range(tensor, min_value, max_value, name):
    actual_min = tensor.min()
    actual_max = tensor.max()
    if actual_min < min_value or actual_max > max_value:
        warnings.warn(
            f"{name} must be in {min_value}..{max_value} range, but it ranges {actual_min}..{actual_max}"
        )
