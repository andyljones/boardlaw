from . import backblaze

def database():
    backblaze.upload('output/arena.sql', 'output/arena.sql')