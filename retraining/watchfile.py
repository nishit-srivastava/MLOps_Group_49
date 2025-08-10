import time
import subprocess
import os
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Configuration
FOLDER_TO_WATCH = "../new_data/"
CURL_COMMAND = [
    "curl", "-X", "POST", "http://localhost:5001/retrain"  # Your Flask retrain endpoint
]

# Lock to prevent overlapping retraining
retrain_lock = threading.Lock()

def file_fully_written(file_path, check_interval=1, checks=3):
    last_size = -1
    stable_count = 0
    while stable_count < checks:
        try:
            current_size = os.path.getsize(file_path)
        except FileNotFoundError:
            return False
        if current_size == last_size:
            stable_count += 1
        else:
            stable_count = 0
        last_size = current_size
        time.sleep(check_interval)
    return True

class WatcherHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            file_path = event.src_path
            print(f"New file detected: {file_path}")

            if file_fully_written(file_path):
                print(f"File complete: {file_path}")
                with retrain_lock:  # Ensure only one retrain at a time
                    print("Triggering retraining...")
                    subprocess.run(CURL_COMMAND)
                    print("Retraining finished.")
            else:
                print(f"File disappeared before completion: {file_path}")

if __name__ == "__main__":
    event_handler = WatcherHandler()
    observer = Observer()
    observer.schedule(event_handler, FOLDER_TO_WATCH, recursive=False)
    observer.start()
    print(f"Monitoring folder: {FOLDER_TO_WATCH}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
