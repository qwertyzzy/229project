# import numpy as np
# import os
# import pandas as pd
import sqlite3


class DataLoader:
    def __init__(self, root, db):
        self.root = root
        self.db = db


    def get_image_paths(self):
        paths = []

        conn = sqlite3.connect(self.db)
        cur = conn.cursor()
        cur.execute("SELECT video_id, frame, class, object_id FROM main")
        rs = cur.fetchall()

        for row in rs:
            video_id = str(row[0])
            frame = str(row[1])
            classNo = str(row[2])
            object_id = str(row[3])
            path = self.root + classNo + '/' + video_id + '_' + frame + '_' + classNo + '_' + object_id + '.jpg'
            paths.append(path)

        return paths

    
    def get_id_class(self, image_id):
        conn = sqlite3.connect(self.db)
        cur = conn.cursor()
        cur.execute("SELECT actual_class FROM main WHERE image_id=?", (int(image_id), ))
        res = cur.fetchall()[0][0]
        return res
        

    def get_db(self):
        return self.db

