import os
import subprocess
import time


# ray blast #################################################################


def ray_blast(folder, out, now_path):
    file_box = []
    out_box = []
    file_mid = []
    out_mid = []
    t = 0
    for f in os.listdir(os.path.join(os.path.join(now_path, 'Reads'), folder)):
        tp_path = os.path.join(os.path.join(now_path, 'Reads'), folder)
        pp_path = os.path.join(os.path.join(now_path, 'PSSMs'), out)
        if out not in os.listdir(os.path.join(now_path, 'PSSMs')):
            os.makedirs(pp_path)
        t += 1
        if t <= 20:
            file_mid.append(os.path.join(tp_path, f))
            out_mid.append(os.path.join(pp_path, f.split('.')[0]))
        else:
            file_box.append(file_mid)
            out_box.append(out_mid)
            file_mid = []
            out_mid = []
            t = 0
    start = time.time()
    for k in range(len(file_box)):
        out_file = ''
        for m in range(len(file_box[k])):
            out_file += file_box[k][m] + '@@' + out_box[k][m] + '\n'
        file_name = os.path.join(now_path, folder) + '_' + str(k)
        if folder + '_' + str(k) not in os.listdir(now_path):
            print(file_name)
            with open(file_name, 'w', encoding='UTF-8') as f:
                f.write(out_file)
            command = 'python Ray_blast.py ' + file_name
            subprocess.Popen(command, shell=True).communicate()
        else:
            pass
    if 'A' in os.listdir(now_path):
        os.remove('A')
    print("共计用时: {}s".format(time.time() - start))


def ray_supplement(folder, out, now_path):
    file_box = []
    out_box = []
    for f in os.listdir(os.path.join(os.path.join(now_path, 'Reads'), folder)):
        tp_path = os.path.join(os.path.join(now_path, 'Reads'), folder)
        pp_path = os.path.join(os.path.join(now_path, 'PSSMs'), out)
        if out not in os.listdir(os.path.join(now_path, 'PSSMs')):
            os.makedirs(pp_path)
        if f.split('.')[0] not in os.listdir(pp_path):
            file_box.append(os.path.join(tp_path, f))
            out_box.append(os.path.join(pp_path, f.split('.')[0]))
    start = time.time()
    out_file = ''
    for m in range(len(file_box)):
        out_file += file_box[m] + '@@' + out_box[m] + '\n'
    file_name = os.path.join(now_path, folder) + '_sup'
    if folder + '_sup' not in os.listdir(now_path):
        print(file_name)
        with open(file_name, 'w', encoding='UTF-8') as f:
            f.write(out_file)
        ray_command = os.path.join('rpct', 'Ray_blast.py')
        command = 'python ' + ray_command + ' ' + file_name
        subprocess.Popen(command, shell=True).communicate()
    else:
        pass
    if 'A' in os.listdir(now_path):
        os.remove('A')
    print("共计用时: {}s".format(time.time() - start))
