import logging
import os


def _custom_path_filter(path):
    project_root = "react-video-agent"
    idx = path.find(project_root)
    if idx != -1:
        path = path[idx + len(project_root):]
    return path


class CustomLogRecord(logging.LogRecord):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pathname = _custom_path_filter(self.pathname)


def setup_logger(log_filename="app.log", log_dir="logs"):
    os.makedirs(log_dir, exist_ok=True)
    log_filepath = os.path.join(log_dir, log_filename)

    logging.setLogRecordFactory(CustomLogRecord)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(module)s] [%(pathname)s]: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_filepath),
        ],
    )
    return logging.getLogger()


logger = setup_logger()
