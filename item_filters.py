"""
Item filters for recommendations.
"""


def get_stop_list():
    """Return a set of items that should be excluded from recommendations"""
    return {
        "bærepose",
        "pose",
        "plastpose",
        "handlepose",
        "Barepose"
    }


def should_exclude_item(item_name):
    """Check if the item name is in the stoplist."""
    stopList = get_stop_list()
    # Case-insensitive check
    return item_name.lower() in [s.lower() for s in stopList]
