"""Search regions: worldwide, continents and countries, as latitude/longitude boxes.

Boxes are (lat_min, lat_max, lon_min, lon_max). They are rough on purpose
(a box always takes in some neighbouring land or sea); restricting the search
to one cuts the work and the false matches a lot. Regions that cross the
180° meridian are cut at it.
"""

REGIONS = {
    "Worldwide": None,
    # continents
    "North America": (7.0, 72.0, -170.0, -52.0),
    "Central America and Caribbean": (7.0, 27.0, -118.0, -59.0),
    "South America": (-56.0, 13.0, -82.0, -34.0),
    "Europe": (34.0, 72.0, -25.0, 45.0),
    "Africa": (-35.0, 38.0, -18.0, 52.0),
    "Middle East": (12.0, 42.0, 34.0, 63.0),
    "Asia": (-11.0, 78.0, 25.0, 180.0),
    "Oceania": (-48.0, 0.0, 110.0, 180.0),
    # countries
    "United States (contiguous)": (24.5, 49.5, -125.0, -66.5),
    "Alaska": (51.0, 72.0, -170.0, -129.0),
    "Hawaii": (18.5, 22.5, -160.5, -154.5),
    "Canada": (41.5, 72.0, -141.0, -52.0),
    "Mexico": (14.5, 33.0, -118.5, -86.5),
    "Brazil": (-34.0, 5.5, -74.0, -34.5),
    "Argentina": (-55.5, -21.5, -73.6, -53.5),
    "United Kingdom and Ireland": (49.8, 61.0, -10.7, 2.0),
    "France": (41.3, 51.2, -5.3, 9.7),
    "Germany": (47.2, 55.1, 5.8, 15.1),
    "Italy": (36.5, 47.1, 6.6, 18.6),
    "Spain and Portugal": (35.9, 43.9, -9.6, 4.4),
    "Benelux": (49.4, 53.6, 2.5, 7.3),
    "Switzerland and Austria": (45.8, 49.1, 5.9, 17.2),
    "Scandinavia": (54.5, 71.5, 4.5, 31.6),
    "Poland": (49.0, 54.9, 14.1, 24.2),
    "Greece and Turkey": (34.8, 42.2, 19.3, 44.9),
    "India": (6.5, 35.7, 68.0, 97.5),
    "China": (18.0, 53.6, 73.5, 135.1),
    "Japan": (24.0, 45.6, 122.9, 146.0),
    "Australia": (-44.0, -10.0, 112.9, 154.0),
    "New Zealand": (-47.4, -34.3, 166.3, 178.6),
    "South Africa": (-35.0, -22.0, 16.4, 33.0),
}


def region_bbox(name):
    """The box of a region name (None for worldwide); unknown names raise ValueError."""
    if name in (None, "", "Worldwide"):
        return None
    if name not in REGIONS:
        raise ValueError(f"unknown region: {name}")
    return REGIONS[name]
