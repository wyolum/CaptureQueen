import time

while 1:
    f = open('fens.txt')
    lines = f.readlines()
    for line in lines:
        line = line.strip()
        with open('.fen', 'w') as fen:
            print(line)
            print(line, file=fen, flush=True)
            time.sleep(1)
    time.sleep(5)
