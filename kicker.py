import subprocess
import os
from datetime import datetime

path = '/home/pi/code/CaptureQueen'

def getTasks(script):
    cmd = f'ps -ef | grep {script} | grep -v grep'
    r = os.popen(cmd).read().strip().split('\n')
    r = [x for x in r if x]
    return r

def kick(script, args=None):
    if args is None:
        args = []
    if not getTasks(script):
        start_time = datetime.now()
        timestamp = start_time.strftime("%y_%m_%d_T%H:%M:%S")
        log_fn = f'{path}/logs/{timestamp}_{script.split("/")[-1][:-2]}log'
        log = open(log_fn, 'w')
        
        subprocess.Popen(["python3",f'{path}/{script}', ' '.join(args)], stdout=log, stderr=log)
        print(f'{script} kicked off in background, logging to {log_fn}')
        
        
if __name__ == '__main__':
    '''
    This code checks tasklist, and will print the status of a code
    '''

    script = 'pyqt_mqtt_chess.py'
    kick(script)
