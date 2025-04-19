import os
import xml.etree.ElementTree as ET
import argparse
import shutil

def load_classes(classes_file):
    with open(classes_file, 'r') as f:
        classes = f.read().strip().split()
    return classes

def convert_voc_to_yolo(images_folder, labels_folder, output_folder, classes_file):
    classes = load_classes(classes_file)
    classes_set = set(classes)

    output_images_folder = os.path.join(output_folder, 'images')
    output_labels_folder = os.path.join(output_folder, 'labels')

    os.makedirs(output_images_folder, exist_ok=True)
    os.makedirs(output_labels_folder, exist_ok=True)

    for label_file in filter(lambda f: f.endswith('.xml'), os.listdir(labels_folder)):
        image_file = label_file.replace('.xml', '.jpg')
        image_path = os.path.join(images_folder, image_file)
        label_path = os.path.join(labels_folder, label_file)

        if not os.path.exists(image_path):
            continue

        tree = ET.parse(label_path)
        root = tree.getroot()

        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        yolo_annotations = []

        for obj in root.iter('object'):
            class_name = obj.find('name').text
            if class_name not in classes_set:
                continue
            class_id = classes.index(class_name)

            bndbox = obj.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            center_x = (xmin + xmax) / 2.0 / width
            center_y = (ymin + ymax) / 2.0 / height
            box_width = (xmax - xmin) / width
            box_height = (ymax - ymin) / height

            yolo_annotations.append(f"{class_id} {center_x} {center_y} {box_width} {box_height}")

        output_image_path = os.path.join(output_images_folder, image_file)
        shutil.copy(image_path, output_image_path)

        yolo_label_path = os.path.join(output_labels_folder, f"{os.path.splitext(image_file)[0]}.txt")
        with open(yolo_label_path, 'w') as f:
            f.write('\n'.join(yolo_annotations))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PascalVOC annotations to YOLOv8 format")
    parser.add_argument('--images_folder', type=str, required=True, help='Path to the folder containing images')
    parser.add_argument('--labels_folder', type=str, required=True, help='Path to the folder containing PascalVOC annotation XML files')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to the folder to save YOLOv8 formatted annotations and images')
    parser.add_argument('--classes_file', type=str, required=True, help='Path to the file containing class names')

    args = parser.parse_args()

    convert_voc_to_yolo(args.images_folder, args.labels_folder, args.output_folder, args.classes_file)

# python pascalVOC_to_yolo.py --images_folder path/to/images --labels_folder path/to/labels --output_folder path/to/output --classes_file path/to/classes.txt
