import xml.etree.ElementTree as ET

ANNOTATIONS_PATH = 'annotations.xml'

Point = tuple[float, float]
Annotation = tuple[Point, Point, Point, Point, Point, Point]


def load_annotations() -> dict[str, Annotation | None]:
    """
    Loads annotations from a CVAT XML file, normalizes the coordinates to [0, 1],
    and returns a dictionary mapping image filenames to their 6-point annotations.
    """
    tree = ET.parse(ANNOTATIONS_PATH)
    root = tree.getroot()

    annotations_dict = {}

    # Iterate through all <image> elements in the XML
    for image in root.findall('image'):
        filename = image.get('name')

        # Safely get dimensions (avoiding DivisionByZero just in case)
        width_str = image.get('width')
        height_str = image.get('height')

        if not filename or not width_str or not height_str:
            continue

        width = float(width_str)
        height = float(height_str)

        points_map = {}
        skeleton = image.find('skeleton')

        # Extract labeled points if the skeleton exists
        if skeleton is not None:
            for pts in skeleton.findall('points'):
                label = pts.get('label')
                coords = pts.get('points')

                if label and coords:
                    x_str, y_str = coords.split(',')
                    x, y = float(x_str), float(y_str)
                    # Normalize the coordinates
                    points_map[label] = (x / width, y / height)

        # Ensure we have exactly labels "1" through "6"
        expected_labels = ['1', '2', '3', '4', '5', '6']
        if all(label in points_map for label in expected_labels):
            # Construct the tuple in the strict order of 1 to 6
            annotation_tuple = (
                points_map['1'],
                points_map['2'],
                points_map['3'],
                points_map['4'],
                points_map['5'],
                points_map['6']
            )
            annotations_dict[filename] = annotation_tuple
        else:
            # If any point is missing or image is unlabeled, return None
            annotations_dict[filename] = None

    return annotations_dict
