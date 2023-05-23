import subprocess

# outfit_generation.sh 및 make_set.py 실행 명령어
command = "./outfit_generation.sh && python3 make_set.py"

# 명령어 실행
subprocess.run(command, shell=True)