"""Model metadata merge rules."""


def merge_model_entries(existing: dict, incoming: dict) -> dict:
    """Merge model entries from multiple parsers. Order-independent."""
    mid = incoming["model_id"]
    result = dict(existing) if existing else {
        "model_id": mid,
        "primary_category_id": incoming.get("primary_category_id", ""),
        "category_ids": [],
        "supports_penalty": False,
        "supports_inference": False,
    }

    # category_ids: union
    cat_set = set(result.get("category_ids", []))
    cat_set.update(incoming.get("category_ids", []))
    result["category_ids"] = sorted(cat_set)

    # primary_category_id: first non-empty wins (central registry preferred)
    if not result.get("primary_category_id") and incoming.get("primary_category_id"):
        result["primary_category_id"] = incoming["primary_category_id"]

    # supports_*: logical OR
    if incoming.get("supports_penalty", False):
        result["supports_penalty"] = True
    if incoming.get("supports_inference", False):
        result["supports_inference"] = True

    return result
