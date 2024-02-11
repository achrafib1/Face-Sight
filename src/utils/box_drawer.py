import cv2
import numpy as np


def blur_faces(image, boxes):
    for box in boxes:
        x1, y1, x2, y2 = box
        face = image[int(y1) : int(y2), int(x1) : int(x2)]
        blurred_face = cv2.GaussianBlur(face, (21, 21), 30)
        image[int(y1) : int(y2), int(x1) : int(x2)] = blurred_face
    return image


def change_background(image, boxes, background):

    # Create a mask of the same size as the image
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for box in boxes:
        x1, y1, x2, y2 = box
        mask[int(y1) : int(y2), int(x1) : int(x2)] = 255

    if isinstance(background, str):

        background = background.lstrip("#")
        background = tuple(int(background[i : i + 2], 16) for i in (0, 2, 4))
        # background = background[::-1]  # Reverse the tuple to get BGR
        background_image = np.ones_like(image) * np.array(background, dtype=np.uint8)
    else:

        background_image = cv2.resize(background, (image.shape[1], image.shape[0]))

    # Use the mask to segment the faces from the image
    faces = cv2.bitwise_and(image, image, mask=mask)

    # Use the inverse of the mask to segment the background from the background image
    mask_inv = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(background_image, background_image, mask=mask_inv)

    # Combine the faces and background to create the new image
    new_image = cv2.add(faces, background)

    return new_image


def change_face_color(image, boxes, color):

    hex_color = color.lstrip("#")
    bgr_color = np.array(
        [[[int(hex_color[i : i + 2], 16) for i in (4, 2, 0)]]], dtype=np.uint8
    )  # Reverse the tuple to get BGR and convert to 3D numpy array
    desired_hsv = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2HSV)[0, 0]

    for box in boxes:
        x1, y1, x2, y2 = box
        face_region = image[int(y1) : int(y2), int(x1) : int(x2)]

        # Convert the face region to HSV
        hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)

        # Change the hue and saturation to the desired color
        hsv[..., 0] = desired_hsv[0]
        hsv[..., 1] = desired_hsv[1]

        # Convert to RGB
        face_region = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # Replace the face region in the original image
        image[int(y1) : int(y2), int(x1) : int(x2)] = face_region

    return image


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
    background="#56ecd5",
    color="#56ecd5",
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
                    # Add the bounding box to the list
                    boxes.append((x1, y1, x2, y2))
                    plot_one_box(
                        xyxy, image, label=label, color=(255, 0, 0), line_thickness=3
                    )  # Draw the bounding box on the original image
            for strategy in strategies:
                if strategy == "blur_faces":
                    image = blur_faces(image, boxes)
                if strategy == "Change Background":
                    image = change_background(image, boxes, background)
                if strategy == "change_face_color":
                    image = change_face_color(image, boxes, color)

    # Resize the image back to its original size
    image = cv2.resize(image, (original_size[1], original_size[0]))
    return (
        faces,
        boxes,
        image,
    )  # Return the list of detected faces and the original image with bounding boxes
