from dotenv import load_dotenv, dotenv_values
import os
import pathlib
import io
import glob

config = dotenv_values(".local.env")
print(f"Loaded configuration is {config}")

PATH_MODEL = pathlib.Path(config.get('PATH_MODEL', ''))
PATH_DATA = pathlib.Path(config.get('PATH_DATA', ''))
SYMBOL_MAPPING = {index: char for index, char in enumerate(set([chr(i) for i in range(1, 3000) if len(chr(i)) == 1]))}

# assert PATH_DATA.exists(), f"Data: Path to {PATH_DATA.absolute()} does not exist"
assert PATH_MODEL.exists(), f"Models: Path to {PATH_MODEL.absolute()} does not exist"

if __name__ == '__main__':
    config = dotenv_values(".env")
    print(config)