import cv2
from typing import List, Tuple, Optional, Union, Callable
import numpy as np
import torch


def blur_faces(image: np.ndarray, boxes: List[Tuple[int, int, int, int]]) -> np.ndarray:
    """
    Blurs the detected faces in the image.

    Parameters:
    - image: The original image.
    - boxes: A list of bounding boxes for each detected face. Each box is a tuple (x1, y1, x2, y2).

    Returns:
    - The image with the detected faces blurred.
    """

    # For each bounding box in the list of boxes
    for box in boxes:
        # Unpack the bounding box coordinates
        x1, y1, x2, y2 = box
        # Extract the face region from the image using the bounding box coordinates
        face = image[int(y1) : int(y2), int(x1) : int(x2)]
        # Apply a Gaussian blur to the face region
        # The kernel size is 21x21 and the standard deviation in the x and y directions is 30
        blurred_face = cv2.GaussianBlur(face, (21, 21), 30)
        # Replace the face region in the original image with the blurred face
        image[int(y1) : int(y2), int(x1) : int(x2)] = blurred_face
    # Return the image with the blurred faces
    return image


def change_background(
    image: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    background: Union[str, np.ndarray],
):
    """
    Changes the background of the detected faces in the image.

    Parameters:
    - image: The original image.
    - boxes: A list of bounding boxes for each detected face. Each box is a tuple (x1, y1, x2, y2).
    - background: The desired background color in hexadecimal format or an image as a numpy array.

    Returns:
    - The image with the background of the detected faces changed.
    """

    # Create a mask of the same size as the image
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    # For each bounding box in the list of boxes
    for box in boxes:
        x1, y1, x2, y2 = box
        # Set the pixels within the bounding box in the mask to 255 (white)
        mask[int(y1) : int(y2), int(x1) : int(x2)] = 255

    # If the background is a string (indicating a color in hexadecimal format)
    if isinstance(background, str):

        # Remove the '#' from the start of the string
        background = background.lstrip("#")
        # Convert the hexadecimal color to a tuple of RGB values
        background_rgb = tuple(int(background[i : i + 2], 16) for i in (0, 2, 4))
        # background = background[::-1]  # Reverse the tuple to get BGR
        # Create a new image of the same size as the original image, filled with the background color
        background_image = np.ones_like(image) * np.array(
            background_rgb, dtype=np.uint8
        )
    else:

        # If the background is not a string, it is an image
        # Resize the background image to the same size as the original image
        background_image = cv2.resize(background, (image.shape[1], image.shape[0]))

    # Use the mask to segment the faces from the image
    faces = cv2.bitwise_and(image, image, mask=mask)

    # Use the inverse of the mask to segment the background from the background image
    mask_inv = cv2.bitwise_not(mask)
    background = cv2.bitwise_and(background_image, background_image, mask=mask_inv)

    # Combine the faces and background to create the new image
    new_image = cv2.add(faces, background)

    return new_image


def change_face_color(
    image: np.ndarray, boxes: List[Tuple[int, int, int, int]], color: str
) -> np.ndarray:
    """
    Changes the color of the detected faces in the image.

    Parameters:
    - image: The original image.
    - boxes: A list of bounding boxes for each detected face. Each box is a tuple (x1, y1, x2, y2).
    - color: The desired color for the faces in hexadecimal format.

    Returns:
    - The image with the detected faces changed to the desired color.
    """
    # Remove the '#' from the start of the color string
    hex_color = color.lstrip("#")
    # Convert the hexadecimal color to a tuple of BGR values and convert it to a 3D numpy array
    bgr_color = np.array(
        [[[int(hex_color[i : i + 2], 16) for i in (4, 2, 0)]]], dtype=np.uint8
    )  # Reverse the tuple to get BGR and convert to 3D numpy array
    desired_hsv = cv2.cvtColor(bgr_color, cv2.COLOR_BGR2HSV)[0, 0]

    # For each bounding box in the list of boxes
    for box in boxes:
        x1, y1, x2, y2 = box
        # Extract the face region from the image using the bounding box coordinates
        face_region = image[int(y1) : int(y2), int(x1) : int(x2)]

        # Convert the face region from BGR to HSV
        hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)

        # Change the hue and saturation to the desired color
        hsv[..., 0] = desired_hsv[0]
        hsv[..., 1] = desired_hsv[1]

        # Convert to RGB
        face_region = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        # Replace the face region in the original image
        image[int(y1) : int(y2), int(x1) : int(x2)] = face_region

    return image


def replace_faces(
    image: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    replacement: Optional[np.ndarray],
) -> np.ndarray:
    """
    Replaces the detected faces in the image with a given replacement image.

    Parameters:
    - image: The original image.
    - boxes: A list of bounding boxes for each detected face. Each box is a tuple (x1, y1, x2, y2).
    - replacement: The replacement image as a numpy array.

    Returns:
    - The image with the detected faces replaced by the replacement image.
    """

    # Check if the replacement image is not None and is a numpy array
    if replacement is not None and type(replacement) == np.ndarray:
        # For each bounding box in the list of boxes
        for box in boxes:
            x1, y1, x2, y2 = box
            # Calculate the width and height of the face region
            face_width = int(x2) - int(x1)
            face_height = int(y2) - int(y1)

            # Resize the replacement image to fit the face region
            replacement_resized = cv2.resize(replacement, (face_width, face_height))

            # Replace the face region with the replacement image
            image[int(y1) : int(y2), int(x1) : int(x2)] = replacement_resized

    # Return the image with the replaced faces
    return image


def highlight_edges(
    image: np.ndarray,
    boxes: List[Tuple[int, int, int, int]],
    face_color: str = "#56ecd5",
) -> np.ndarray:
    """
    Highlights the edges of the detected faces in the image.

    Parameters:
    - image: The original image.
    - boxes: A list of bounding boxes for each detected face. Each box is a tuple (x1, y1, x2, y2).
    - face_color: The color to use for highlighting the edges of the faces, in hexadecimal format.

    Returns:
    - The image with the edges of the detected faces highlighted.
    """

    # Remove the '#' from the start of the color string
    hex_color = face_color.lstrip("#")
    # Convert the hexadecimal color to a tuple of BGR values
    face_color_bgr = tuple(
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
    # For each bounding box in the list of boxes
    for box in boxes:
        x1, y1, x2, y2 = box
        # Extract the edges within the bounding box
        face_edges = edges[int(y1) : int(y2), int(x1) : int(x2)]
        # Set the pixels in the edge image where the face edges are detected to the desired color
        edge_image[int(y1) : int(y2), int(x1) : int(x2)][
            face_edges != 0
        ] = face_color_bgr

    # Return the edge image with the highlighted edges
    return edge_image


def pixelate_faces(
    image: np.ndarray, boxes: List[Tuple[int, int, int, int]], pixel_size: int = 10
) -> np.ndarray:
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

    # For each bounding box in the list of boxes
    for box in boxes:
        # Unpack the bounding box coordinates
        x1, y1, x2, y2 = box
        # Extract the face region from the image using the bounding box coordinates
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

    # Return the image with the pixelated faces
    return pixelated_image


def draw_boxes(
    pred: np.ndarray,
    image: np.ndarray,
    img: torch.Tensor,
    names: List[str],
    original_size: Tuple[int, int],
    faces: List[np.ndarray],
    scale_coords: Callable,
    plot_one_box: Callable,
    strategies: List[str],
    background: str = "#56ecd5",
    color: str = "#56ecd5",
    image_replacement: Optional[np.ndarray] = None,
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]], np.ndarray]:
    """
    Draws bounding boxes on the detected faces in the image and applies various strategies.

    Parameters:
    - pred: The model's predictions.
    - image: The original image.
    - img: The reshaped and normalized image tensor.
    - names: The names of the classes.
    - original_size: The original size of the image.
    - faces: A list to store the detected faces.
    - scale_coords: The function to rescale the coordinates to the original image size.
    - plot_one_box: The function to draw a bounding box on the image.
    - strategies: A list of strategies to apply to the image.
    - background: The desired background color in hexadecimal format.
    - color: The desired color for the faces in hexadecimal format.
    - image_replacement: The replacement image as a numpy array.

    Returns:
    - faces: The list of detected faces.
    - boxes: The list of bounding boxes for each detected face.
    - image: The image with the bounding boxes drawn and the strategies applied.
    """

    boxes = []  # List to store the bounding boxes
    # Process the predictions
    for det in pred:
        if len(det):
            # Rescale the coordinates to the original image size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()

            # Draw the bounding boxes on the image
            for *xyxy, conf, cls in reversed(det):
                if conf > 0.4:
                    # Unpack the coordinates of the bounding box
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
            # Apply the selected strategies to the detected faces
            for strategy in strategies:
                if strategy == "blur faces":
                    image = blur_faces(image, boxes)
                if strategy == "Change Background":
                    image = change_background(image, boxes, background)
                if strategy == "change face color":
                    image = change_face_color(image, boxes, color)
                if strategy == "replace faces":
                    image = replace_faces(image, boxes, image_replacement)
                if strategy == "highlight edges":
                    image = highlight_edges(image, boxes)
                if strategy == "pixelate faces":
                    image = pixelate_faces(image, boxes)

    # Resize the image back to its original size
    image = cv2.resize(image, (original_size[1], original_size[0]))
    return (
        faces,
        boxes,
        image,
    )  # Return the list of detected faces and the original image with bounding boxes
