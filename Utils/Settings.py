import json

class ProjectSettings:
    @staticmethod
    def Settings():
        with open('./Core/Settings.json') as f:
            return json.load(f)