import os
import cv2
import argparse
from ultralytics import YOLO

def detect_and_label_ppe(input_dir, output_dir, person_det_model, ppe_detection_model):
    # Load YOLOv8 models
    person_detector = YOLO(person_det_model)
    ppe_detector = YOLO(ppe_detection_model)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate through images in the input directory
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        image = cv2.imread(img_path)

        # Person detection
        results = person_detector(image)
        person_boxes = results[0].boxes.xyxy.cpu().numpy()
        person_scores = results[0].boxes.conf.cpu().numpy()

        for i, box in enumerate(person_boxes):
            x1, y1, x2, y2 = map(int, box[:4])
            if person_scores[i] > 0.5:  # Confidence threshold for person detection
                # Crop the person from the image
                cropped_person = image[y1:y2, x1:x2]

                # PPE detection
                ppe_results = ppe_detector(cropped_person)
                ppe_boxes = ppe_results[0].boxes.xyxy.cpu().numpy()
                ppe_scores = ppe_results[0].boxes.conf.cpu().numpy()
                ppe_classes = ppe_results[0].boxes.cls.cpu().numpy()

                # Iterate over the PPE detections and draw bounding boxes and labels
                for j, ppe_box in enumerate(ppe_boxes):
                    ppe_x1, ppe_y1, ppe_x2, ppe_y2 = map(int, ppe_box[:4])
                    if ppe_scores[j] > 0.5:  # Confidence threshold for PPE detection
                        label = ppe_detector.names[int(ppe_classes[j])]

                        # Adjust the text position
                        font_scale = 0.45
                        font_thickness = 2
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
                        text_x = max(ppe_x1 - text_width - 10, 0)
                        text_y = ppe_y1 + text_height // 2

                        # Draw the bounding box and label on the cropped person
                        color = (255, 0, 0) if label == 'hard-hat' else (0, 255, 0)
                        cv2.rectangle(cropped_person, (ppe_x1, ppe_y1), (ppe_x2, ppe_y2), color, 2)
                        cv2.putText(cropped_person, label, (text_x, text_y), font, font_scale, color, font_thickness)

                # Restore the cropped and labeled person into the original image
                image[y1:y2, x1:x2] = cropped_person

        # Save the processed image to the output directory
        output_path = os.path.join(output_dir, img_name)
        cv2.imwrite(output_path, image)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Detect and label PPE in images.')
    parser.add_argument('input_dir', type=str, help='Path to the input directory containing images.')
    parser.add_argument('output_dir', type=str, help='Path to the output directory to save labeled images.')
    parser.add_argument('person_det_model', type=str, help='Path to the YOLOv8 model for person detection.')
    parser.add_argument('ppe_detection_model', type=str, help='Path to the YOLOv8 model for PPE detection.')

    args = parser.parse_args()

    detect_and_label_ppe(args.input_dir, args.output_dir, args.person_det_model, args.ppe_detection_model)

# python inference.py --input_dir path/to/input_dir --output_dir path/to/output_dir --person_det_model path/to/person_det.pt --ppe_detection_model path/to/ppe_det.pt
    
