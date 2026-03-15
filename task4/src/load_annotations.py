import xml.etree.ElementTree as ET

ANNOTATIONS_PATH = 'annotations.xml'

Point = tuple[float, float]
Annotation = tuple[Point, Point]


def load_annotations() -> dict[str, Annotation | None]:
    """
    Loads annotations from a CVAT XML file, and returns a dictionary
    mapping image filenames to their 2-point annotations.
    """
    tree = ET.parse(ANNOTATIONS_PATH)
    root = tree.getroot()

    annotations_dict = {}

    # Iterate through all <image> elements in the XML
    for image in root.findall('image'):
        filename = image.get('name')

        if not filename:
            continue

        # Find the polyline element
        polyline = image.find('polyline')

        if polyline is not None:
            coords_str = polyline.get('points')

            if coords_str:
                # coords_str looks like "50.38,297.31;49.60,535.06"
                # Split by ';' to get the two points, then by ',' for x and y
                points = coords_str.split(';')

                if len(points) == 2:
                    x1_str, y1_str = points[0].split(',')
                    x2_str, y2_str = points[1].split(',')

                    x1, y1 = float(x1_str), float(y1_str)
                    x2, y2 = float(x2_str), float(y2_str)

                    # Note: If you still need coordinates normalized to [0, 1],
                    # extract width/height from the <image> tag and divide here:
                    width, height = float(image.get('width')), float(image.get('height'))
                    x1, y1 = x1 / width, y1 / height
                    x2, y2 = x2 / width, y2 / height

                    annotations_dict[filename] = ((x1, y1), (x2, y2))
                else:
                    # Fallback if there aren't exactly 2 points in the polyline
                    annotations_dict[filename] = None
            else:
                annotations_dict[filename] = None
        else:
            # If the image is unlabeled
            annotations_dict[filename] = None

    return annotations_dict
