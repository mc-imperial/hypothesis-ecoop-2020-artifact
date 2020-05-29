import os
import hashlib
from reducereval import WORKING
import time


def mkdirp(dirname):
    try:
        os.makedirs(dirname)
    except FileExistsError:
        pass
    return dirname


def file_age_in_seconds(pathname):
    return time.time() - os.stat(pathname).st_mtime


def lockfile_for_key(key):
    return os.path.join(
        mkdirp(os.path.join(WORKING, "locks")),
        hashlib.sha1(key.encode("utf-8")).hexdigest()[:16],
    )


def claim_lock(key):
    lockfile = lockfile_for_key(key)
    try:
        if file_age_in_seconds(lockfile) >= 300:
            os.unlink(lockfile)
    except FileNotFoundError:
        pass

    try:
        with open(lockfile, "x"):
            return True
    except FileExistsError:
        return False


def release_lock(key):
    lockfile = lockfile_for_key(key)
    try:
        os.unlink(lockfile)
    except FileNotFoundError:
        pass
