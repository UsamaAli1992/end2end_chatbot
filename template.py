import logging
import os
from pathlib import Path

# See https://docs.python.org/3/library/logging.html#logging.basicConfig
logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s")

list_of_files = [
    "src/__init__.py",
    "src/helper.py",
    "src/prompt.py",
    ".env",
    "setup.py",
    "research/trials.ipynb",
    "app.py",
    "store_index.py",
    "static",
    "templates/chat.html",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info("Created directory: %s for the file: %s", filedir, filename)

    # Paths like "static" are folders: last segment has no "." (files like .env still have a dot).
    is_dir_only = "." not in filename
    if is_dir_only:
        filepath.mkdir(parents=True, exist_ok=True)
        logging.info("Created directory: %s", filepath)
        continue

    if (not filepath.is_file()) or filepath.stat().st_size == 0:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.touch()
        logging.info("Creating empty file: %s", filepath)
    else:
        logging.info("File already exists: %s", filepath)

logging.info("All files are created successfully.")
