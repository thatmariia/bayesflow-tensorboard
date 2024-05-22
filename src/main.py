import os
import datetime

from src.minimal_example import run_minimal_example


if __name__ == '__main__':

    # A directory for all the logs across all runs
    general_log_dir = os.path.join("logs", "fit")

    # A directory for the current run
    now_subdir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    current_log_dir = os.path.join(general_log_dir, now_subdir)
    os.makedirs(current_log_dir, exist_ok=True)  # Create the directory if it does not exist
    print("Logging to:", current_log_dir)

    # check if current_log_dir is created
    if os.path.exists(current_log_dir):
        print("current_log_dir is created")
    else:
        print("current_log_dir is not created")

    run_minimal_example(current_log_dir)



