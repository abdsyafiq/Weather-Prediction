from datetime import datetime
import re


def log(message: str) -> None:
    """This function logs the mentioned message at a given stage of the code execution to a log file.
    Function returns nothing"""

    now = datetime.now().strftime("%d-%h-%Y %H:%M:%S")
    with open("./logger/process.log", "a") as f: 
        f.write(f"[{now}]" + " " + message + "\n")

    if re.search(r"Error ", message):
        with open("./logger/error.log", "a") as f: 
            f.write(f"[{now}]" + " " + message + "\n")
