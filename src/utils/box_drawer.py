import cv2
import numpy as np


def blur_faces(image, boxes):
    x1, y1, x2, y2 = boxes
    face = image[int(y1) : int(y2), int(x1) : int(x2)]
    blurred_face = cv2.GaussianBlur(face, (21, 21), 30)
    image[int(y1) : int(y2), int(x1) : int(x2)] = blurred_face
    return image


def whiten_background(image, boxes):

    # Create a mask of the same size as the image
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    x1, y1, x2, y2 = boxes
    mask[int(y1) : int(y2), int(x1) : int(x2)] = 255

    # Create a white image of the same size as the image
    white_image = np.ones_like(image) * 255

    # Use the mask to segment the faces from the image
    faces = cv2.bitwise_and(image, image, mask=mask)

    # Use the inverse of the mask to segment the background from the white image
    mask_inv = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(white_image, white_image, mask=mask_inv)

    # Combine the faces and background to create the new image
    new_image = cv2.add(faces, background)

    return new_image


def draw_boxes(
    pred,
    image,
    img,
    names,
    original_size,
    faces,
    scale_coords,
    plot_one_box,
    strategies,
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
                    for strategy in strategies:
                        if strategy == "blur_faces":
                            image = blur_faces(image, (x1, y1, x2, y2))
                        if strategy == "whiten_background":
                            image = whiten_background(image, (x1, y1, x2, y2))
                    # Add the bounding box to the list
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
