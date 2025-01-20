import json


class Jsonify:
    @staticmethod
    def save(content: str, file_dir: str):
        with open(file_dir, "w") as f:
            json.dump(content, f)
        print("Saved to:'{}'".format(file_dir))

    @staticmethod
    def load(file_dir: str):
        with open(file_dir, "r") as f:
            file = json.load(f)
        return file
