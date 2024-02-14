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
    """
    Changes the color of the detected faces in the image.

    Parameters:
    - image: The original image.
    - boxes: A list of bounding boxes for each detected face. Each box is a tuple (x1, y1, x2, y2).
    - color: The desired color for the faces in hexadecimal format.

    Returns:
    - The image with the detected faces changed to the desired color.
    """

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


def replace_faces(image, boxes, replacement):
    if replacement is not None and type(replacement) == np.ndarray:
        for box in boxes:
            x1, y1, x2, y2 = box
            face_width = int(x2) - int(x1)
            face_height = int(y2) - int(y1)

            # Resize the replacement image to fit the face region
            replacement_resized = cv2.resize(replacement, (face_width, face_height))

            # Replace the face region with the replacement image
            image[int(y1) : int(y2), int(x1) : int(x2)] = replacement_resized

    return image


def highlight_edges(image, boxes, face_color="#56ecd5"):

    hex_color = face_color.lstrip("#")
    face_color = tuple(
        int(hex_color[i : i + 2], 16) for i in (4, 2, 0)
    )  # Reverse the tuple to get BGR
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect edges in the image
    edges = cv2.Canny(gray, 50, 150)

    # Create a copy of the original image to draw the edges on
    # edge_image = image.copy()
    edge_image = np.zeros_like(image)

    # Draw the edges in white
    edge_image[edges != 0] = (255, 255, 255)

    # Highlight the edges of the detected faces with the chosen color
    for box in boxes:
        x1, y1, x2, y2 = box
        face_edges = edges[int(y1) : int(y2), int(x1) : int(x2)]
        edge_image[int(y1) : int(y2), int(x1) : int(x2)][face_edges != 0] = face_color

    return edge_image


def pixelate_faces(image, boxes, pixel_size=10):
    """
    Pixelates the detected faces in the image.

    Parameters:
    - image: The original image.
    - boxes: A list of bounding boxes for each detected face. Each box is a tuple (x1, y1, x2, y2).
    - pixel_size: The size of the pixels for the pixelation effect.

    Returns:
    - The image with the detected faces pixelated.
    """

    # Create a copy of the original image to draw on
    pixelated_image = image.copy()

    for box in boxes:
        x1, y1, x2, y2 = box
        face = pixelated_image[int(y1) : int(y2), int(x1) : int(x2)]
        # Resize the face to a smaller size to create the pixelation effect
        small = cv2.resize(
            face, (pixel_size, pixel_size), interpolation=cv2.INTER_LINEAR
        )
        # Resize the small image back to the size of the face
        face_pixelated = cv2.resize(
            small, (face.shape[1], face.shape[0]), interpolation=cv2.INTER_NEAREST
        )
        # Replace the face in the image with the pixelated version
        pixelated_image[int(y1) : int(y2), int(x1) : int(x2)] = face_pixelated

    return pixelated_image


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
    image_replacement=None,
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
                if strategy == "change face_color":
                    image = change_face_color(image, boxes, color)
                if strategy == "replace faces":
                    image = replace_faces(image, boxes, image_replacement)
                if strategy == "highlight_edges":
                    image = highlight_edges(image, boxes)
                if strategy == "pixelate_faces":
                    image = pixelate_faces(image, boxes)

    # Resize the image back to its original size
    image = cv2.resize(image, (original_size[1], original_size[0]))
    return (
        faces,
        boxes,
        image,
    )  # Return the list of detected faces and the original image with bounding boxes
