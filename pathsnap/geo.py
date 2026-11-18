"""Per-trip metric projection (UTM from centroid) and network cropping.

Owner: P1
"""

def utm_epsg(lon, lat):
    """EPSG code of the UTM zone containing (lon, lat)."""
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def project(lon, lat, epsg):
    raise NotImplementedError  # pyproj.Transformer.from_crs(4326, epsg, always_xy=True)


def crop_network(nodes_xy, arcs, track_xy, margin_m=200):
    raise NotImplementedError
