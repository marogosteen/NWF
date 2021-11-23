import json
from config import Config

config = Config()
print(type(config))
print(dir(config))
dir(config)
print(config.epochs)
d = config.__dict__


with open("hoge.json", mode="w") as f:
    json.dump(d, f, indent=1)
