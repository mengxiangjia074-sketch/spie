from datetime import datetime


def utc_timestamp():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def local_timestamp():
    return datetime.now().replace(microsecond=0).isoformat()


def run_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

