#! /usr/bin/env python3

"""Document for saving data to txt file
"""


class DataLogger():
    def __init__(self, save_path):
        self.save_path = save_path
        with open(save_path, 'w') as f: f.write('')

    def write_data(self, line_number, data):
        txt = str(line_number)
        for d in data:
            txt += f',{d}'
        with open(self.save_path, 'a') as writer: # Appends to the file
            writer.write(f'{txt}\n')
