import os,psutil,time
import subprocess
CARLA_PATH = '/home/ylh/CARLA/CARLA_0.9.13'

operating_system='windows' if os.name=='nt' else 'linux'

def get_binary(show):
    if show:
        return 'CarlaUE4.exe' if operating_system=='windows' else 'CarlaUE4.sh'
    else:
        return 'CarlaUE4.exe' if operating_system=='windows' else 'CarlaUE4.sh -RenderOffScreen '

def get_exec_command(show):
    binary=get_binary(show)
    exec_command=binary if operating_system=='windows' else ('./'+binary)

    return binary,exec_command

def kill_process():
    binary=get_binary(show=False)
    for process in psutil.process_iter():
        if process.name().lower().startswith(binary.split('.')[0].lower()):
            try:
                process.terminate()
            except:
                pass

    still_alive=[]
    for process in psutil.process_iter():
        if process.name().lower().startswith(binary.split('.')[0].lower()):
            still_alive.append(process)

    if len(still_alive):
        for process in still_alive:
            try:
                process.kill()
            except:
                pass
        psutil.wait_procs(still_alive)

# Starts Carla simulator
def start_process(show  = True):
    # Kill Carla processes if there are any and start simulator
    print('Starting Carla...')
    kill_process()
    subprocess.Popen(get_exec_command(show)[1],cwd=CARLA_PATH, shell=True)
    time.sleep(5)