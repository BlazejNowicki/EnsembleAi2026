import xml.etree.ElementTree as ET

ANNOTATIONS_PATH = 'annotations.xml'

Point = tuple[float, float]
Annotation = tuple[Point, Point, Point]


def load_annotations() -> dict[str, Annotation | None]:
    """
    Loads annotations from a CVAT XML file, normalizes coordinates to [0, 1],
    and returns a dictionary mapping image filenames to their 3-point annotations
    (two from the polyline, one from the 'end' point).
    """
    tree = ET.parse(ANNOTATIONS_PATH)
    root = tree.getroot()

    annotations_dict = {}

    # Iterate through all <image> elements in the XML
    for image in root.findall('image'):
        filename = image.get('name')

        if not filename:
            continue

        width_str = image.get('width')
        height_str = image.get('height')

        if not width_str or not height_str:
            annotations_dict[filename] = None
            continue

        width, height = float(width_str), float(height_str)

        # Find the polyline element
        polyline = image.find('polyline')

        # Find the point element with label="end"
        end_point = None
        for pts in image.findall('points'):
            if pts.get('label') == 'end':
                end_point = pts
                break

        # Proceed only if both annotations exist in the image
        if polyline is not None and end_point is not None:
            poly_coords_str = polyline.get('points')
            end_coords_str = end_point.get('points')

            if poly_coords_str and end_coords_str:
                # poly_coords_str looks like "50.38,297.31;49.60,535.06"
                poly_points = poly_coords_str.split(';')

                if len(poly_points) == 2:
                    # Extract the two points from the polyline
                    x1_str, y1_str = poly_points[0].split(',')
                    x2_str, y2_str = poly_points[1].split(',')

                    # Extract the single point from the 'end' label
                    # end_coords_str looks like "100.5,200.0"
                    x3_str, y3_str = end_coords_str.split(',')

                    # Parse to floats and normalize to [0, 1]
                    x1, y1 = float(x1_str) / width, float(y1_str) / height
                    x2, y2 = float(x2_str) / width, float(y2_str) / height
                    x3, y3 = float(x3_str) / width, float(y3_str) / height

                    annotations_dict[filename] = ((x1, y1), (x2, y2), (x3, y3))
                else:
                    # Fallback if there aren't exactly 2 points in the polyline
                    annotations_dict[filename] = None
            else:
                # Missing coordinate strings
                annotations_dict[filename] = None
        else:
            # If either the polyline or the 'end' point is missing
            annotations_dict[filename] = None

    return annotations_dict
