import os
parent_dir = os.path.dirname(__file__)

os.system('sudo docker run -it -v ' + parent_dir + ':/pytest elliptic_scy:latest -c "cd/pytest; bash"')
# os.system('sudo docker run -it -v ' + parent_dir + ':/pytest elliptic_scy:latest bash -c "cd /pytest/preprocessor; python3 definicoes.py"')

# sudo docker run -it -v  $PWD:/pytest elliptic_scy:latest bash -c "cd /pytest/preprocessor; bash"
