import cv2
import git
import os


def load_model(model_path):
    # Check if the repository exists locally. If not, clone it.
    if not os.path.exists("yolov7"):
        try:
            git.Repo.clone_from("https://github.com/WongKinYiu/yolov7.git", "yolov7")
        except git.GitCommandError as e:
            print(f"Error occurred while cloning the repository: {e}")
            return None

    # Now that the repository is cloned, we can import the utility functions.
    from yolov7.models.experimental import attempt_load
    from yolov7.utils.general import scale_coords, non_max_suppression
    from yolov7.utils.plots import plot_one_box

    model = attempt_load(model_path)
    names = model.module.names if hasattr(model, "module") else model.names
    return model, names, scale_coords, non_max_suppression, plot_one_box
