import os
import sqlite3

conn = sqlite3.connect('cluster.db')
cur = conn.cursor()

cur.execute('''
          CREATE TABLE IF NOT EXISTS "main"
          (
            video_id VARCHAR, 
            frame INTEGER, 
            image_id INTEGER NOT NULL,
            class INTEGER, 
            object_id INTEGER,
            object_presence VARCHAR,
            xmin INTEGER, 
            xmax INTEGER, 
            ymin INTEGER, 
            ymax INTEGER, 
            actual_class INTEGER,
            train BIT,
            PRIMARY KEY (image_id)
            )
          ''')

conn.commit()
conn.close()

