# Define longitude boundaries for different datasets
lon_size = {
    "BJ13": [115.25, 116.25],
    "TaxiNYC": [-74.3, -73.7],
    "Bike": [73.73, 74.22]
}

# Define latitude boundaries for different datasets
lat_size = {
    "BJ13": [39.26, 40.089],
    "TaxiNYC": [40.4, 40.9],
    "Bike": [40.49, 40.90],
}

def pro_norm(val, min_val, max_val):
    """
    Normalize a value to the range [-0.5, 0.5] using min-max normalization.

    Args:
        val (float): Original value.
        min_val (float): Minimum value of the range.
        max_val (float): Maximum value of the range.

    Returns:
        float: Normalized and clipped value in [-0.5, 0.5].
    """
    val = (val - min_val) / (max_val - min_val) - 0.5
    return min(max(val, -0.5), 0.5)

def space_features(coords, dataset):
    """
    Normalize spatial coordinates (longitude and latitude) for a given dataset.

    Args:
        coords (list of [lon, lat]): List of coordinate pairs.
        dataset (str): Dataset name ("BJ13", "TaxiNYC", or "Bike").

    Returns:
        list of [lon_n, lat_n]: Normalized coordinates in [-0.5, 0.5].
    """
    out = []
    for lon, lat in coords:
        lon_n = pro_norm(lon, *lon_size[dataset])
        lat_n = pro_norm(lat, *lat_size[dataset])
        out.append([lon_n, lat_n])
    return out
