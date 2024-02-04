import cv2


def draw_boxes(
    pred, image, img, names, original_size, faces, scale_coords, plot_one_box
):
    boxes = []  # List to store the bounding boxes
    # Process the predictions
    for det in pred:
        if len(det):
            # Rescale the coordinates to the original image size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()

            # Draw the bounding boxes on the image
            for *xyxy, conf, cls in reversed(det):
                if conf > 0.4:
                    x1, y1, x2, y2 = xyxy
                    label = f"{names[int(cls)]} {conf:.2f}"
                    face = image[int(y1) : int(y2), int(x1) : int(x2)]
                    faces.append(face)  # Add the detected face to the list
                    # _, face = cv2.imencode(".jpeg", face)
                    boxes.append((x1, y1, x2, y2))  # Add the bounding box to the list
                    plot_one_box(
                        xyxy, image, label=label, color=(255, 0, 0), line_thickness=3
                    )  # Draw the bounding box on the original image

    # Resize the image back to its original size
    image = cv2.resize(image, (original_size[1], original_size[0]))
    return (
        faces,
        boxes,
        image,
    )  # Return the list of detected faces and the original image with bounding boxes
