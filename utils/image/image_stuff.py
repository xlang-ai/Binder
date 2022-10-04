import json


def get_caption_map(file_path=None):
    """
    Get the caption map.
    """
    if not file_path:
        file_path = 'mmqa_captions.json'

    with open(file_path, "r") as f:
        caption_map = json.load(f)
    return caption_map


def get_caption(id):
    """
    Get the caption of the picture by id.
    """
    with open("mmqa_captions.json", "r") as f:
        caption = json.load(f)
    if id in caption.keys():
        return caption[id]
    else:
        return ""
