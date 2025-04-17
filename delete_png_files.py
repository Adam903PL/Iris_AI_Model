import os

import os


directory = './'

for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith('.png'):
            file_path = os.path.join(root, file)
            os.remove(file_path)
            print(f'Usunięto: {file_path}')
