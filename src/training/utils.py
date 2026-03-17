import subprocess

import yaml


def get_git_revision_short_hash() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )


def get_dvc_hash(dvc_file_path):
    with open(dvc_file_path, "r") as f:
        data = yaml.safe_load(f)

    try:
        md5_hash = data["outs"][0]["md5"]
        return md5_hash
    except (KeyError, IndexError):
        return "Hash not found. Ensure this is a valid .dvc file."
