"""
Builds the structured attribute layer: for each image, extracts garment
category, dominant color (via bbox pixel sampling), and style attributes
from Fashionpedia's COCO-style annotations.
"""
import json
import numpy as np
from PIL import Image
from collections import defaultdict

NAMED_COLORS = {
    'black': (0,0,0), 'white': (255,255,255), 'grey': (128,128,128),
    'red': (220,20,20), 'maroon': (128,0,0), 'pink': (255,192,203),
    'orange': (255,140,0), 'yellow': (255,215,0), 'gold': (212,175,55),
    'green': (34,139,34), 'olive': (128,128,0), 'teal': (0,128,128),
    'blue': (30,60,200), 'navy blue': (0,0,128), 'purple': (128,0,128),
    'brown': (139,69,19), 'beige': (222,196,176), 'tan': (210,180,140),
    'cream': (255,253,208),
}

def closest_color_name(rgb):
    r, g, b = rgb
    best_name, best_dist = None, float('inf')
    for name, (cr, cg, cb) in NAMED_COLORS.items():
        dist = (r-cr)**2 + (g-cg)**2 + (b-cb)**2
        if dist < best_dist:
            best_dist, best_name = dist, name
    return best_name

def get_dominant_color(image_path, bbox):
    x, y, w, h = [int(v) for v in bbox]
    img = Image.open(image_path).convert('RGB')
    crop = img.crop((x, y, x + max(w,1), y + max(h,1))).resize((30, 30))
    pixels = np.array(crop).reshape(-1, 3)
    median_color = tuple(np.median(pixels, axis=0).astype(int))
    return closest_color_name(median_color), median_color

def build_image_metadata(annotation_json_path, images_dir, selected_image_ids):
    """
    Given a Fashionpedia-style annotation file and a list of image IDs,
    returns {image_id: {file_name, garments: [{category, color, attributes, bbox}]}}
    """
    with open(annotation_json_path, 'r') as f:
        data = json.load(f)

    anns_by_image = defaultdict(list)
    for ann in data['annotations']:
        anns_by_image[ann['image_id']].append(ann)

    cat_id_to_name = {c['id']: c['name'] for c in data['categories']}
    attr_id_to_info = {a['id']: (a['name'], a['supercategory']) for a in data['attributes']}
    img_id_to_info = {img['id']: img for img in data['images']}

    image_metadata = {}
    for img_id in selected_image_ids:
        img_info = img_id_to_info[img_id]
        file_name = img_info['file_name']
        img_path = f"{images_dir}/{file_name}"

        garments = []
        for ann in anns_by_image[img_id]:
            try:
                color_name, _ = get_dominant_color(img_path, ann['bbox'])
            except Exception:
                color_name = None
            garment_type = cat_id_to_name[ann['category_id']]
            style_attrs = [attr_id_to_info[aid][0] for aid in ann.get('attribute_ids', []) if aid in attr_id_to_info]
            garments.append({
                'category': garment_type, 'color': color_name,
                'attributes': style_attrs, 'bbox': ann['bbox'],
            })

        image_metadata[img_id] = {'file_name': file_name, 'garments': garments}

    return image_metadata
