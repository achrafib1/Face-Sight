import os
import git


def clone_repo():
    # Check if the repository exists locally. If not, clone it.
    if not os.path.exists("src/yolov7"):
        try:
            git.Repo.clone_from(
                "https://github.com/WongKinYiu/yolov7.git", "src/yolov7"
            )
        except git.GitCommandError as e:
            print(f"Error occurred while cloning the repository: {e}")
            return None


clone_repo()
