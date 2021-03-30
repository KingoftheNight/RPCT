#  导入库  ###############################################################


import os
trainbe_path = os.path.dirname(__file__)
grid_path = os.path.join(os.path.join(os.path.join(trainbe_path, 'libsvm-3.24'), 'tools'), 'grid.py')
import math
import random
from math import sqrt
import shutil
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import subprocess
import sys
sys.path.append(trainbe_path)
from ReadBE import extract_time
sys.path.append(os.path.join(os.path.join(trainbe_path, 'libsvm-3.24'), 'python'))
from svmutil import svm_read_problem, svm_train, svm_save_model, svm_load_model, svm_predict


# 核心程序 ###############################################################


# 训练模型
def train_command(file, c_number, gamma, out):
    y, x = svm_read_problem(file)
    model = svm_train(y, x, '-s 0 -t 2 -c ' + c_number + ' -g ' + gamma)
    svm_save_model(out, model)
    return y, x, model


# 预测模型
def predict_command(file, model):
    y_p, x_p = svm_read_problem(file)
    model = svm_load_model(model)
    p_label, p_acc, p_val = svm_predict(y_p, x_p, model)
    return y_p, p_label, p_acc, p_val


# class TEP ###################################################################

# 构建超参数文件
def make_hys(folder, c, g, out, now_path):
    out_file = ''
    folder = os.path.join(now_path, folder)
    out = os.path.join(now_path, out)
    for i in os.listdir(folder):
        out_file += i + '	C_numbr: ' + c + '	Gamma: ' + g + '\n'
    with open(out, 'w') as f:
        f.write(out_file)
        f.close()


# 超参数文件读取
def train_cg(path):
    with open(path, 'r') as f:
        data = f.readlines()
        f.close()
    cg_box = {}
    for i in data:
        cg_box[i.split('\t')[0]] = [i.split('\t')[1].split(': ')[-1], i.split('\t')[2].split(': ')[-1]]
    return cg_box


# 数据均分
def eval_index(data, crossV):
    if crossV != '-1':
        for x in range(len(data)):
            if float(data[x].split(' ')[0]) != 0:
                middle = x  # 分开两类数据
                break
        Pos_index = []
        Neg_index = []
        for i in range(int(crossV)):
            Pos_index.append([])
            Neg_index.append([])
        for i in range(middle):
            if (i % (2 * int(crossV))) == 0:
                Pos_index[0].append(i)
            if (i % (2 * int(crossV))) == 1:
                Pos_index[0].append(i)
            if (i % (2 * int(crossV))) == 2:
                Pos_index[1].append(i)
            if (i % (2 * int(crossV))) == 3:
                Pos_index[1].append(i)
            if (i % (2 * int(crossV))) == 4:
                Pos_index[2].append(i)
            if (i % (2 * int(crossV))) == 5:
                Pos_index[2].append(i)
            if (i % (2 * int(crossV))) == 6:
                Pos_index[3].append(i)
            if (i % (2 * int(crossV))) == 7:
                Pos_index[3].append(i)
            if (i % (2 * int(crossV))) == 8:
                Pos_index[4].append(i)
            if (i % (2 * int(crossV))) == 9:
                Pos_index[4].append(i)
        for i in range(middle, len(data)):
            if (i % int(crossV)) == 0:
                Neg_index[0].append(i)
            if (i % (2 * int(crossV))) == 1:
                Neg_index[0].append(i)
            if (i % (2 * int(crossV))) == 2:
                Neg_index[1].append(i)
            if (i % (2 * int(crossV))) == 3:
                Neg_index[1].append(i)
            if (i % (2 * int(crossV))) == 4:
                Neg_index[2].append(i)
            if (i % (2 * int(crossV))) == 5:
                Neg_index[2].append(i)
            if (i % (2 * int(crossV))) == 6:
                Neg_index[3].append(i)
            if (i % (2 * int(crossV))) == 7:
                Neg_index[3].append(i)
            if (i % (2 * int(crossV))) == 8:
                Neg_index[4].append(i)
            if (i % (2 * int(crossV))) == 9:
                Neg_index[4].append(i)
        Pos_files = []
        Neg_files = []
        for i in range(len(Pos_index)):
            mid_file = ''
            for j in Pos_index[i]:
                mid_file += data[j]
            Pos_files.append(mid_file)
            mid_file = ''
            for j in Neg_index[i]:
                mid_file += data[j]
            Neg_files.append(mid_file)
        return Pos_files, Neg_files
    else:
        Pos_files = []
        Neg_files = []
        for i in range(len(data)):
            extract_time(i, len(data))
            Pos_files.append(data[i])
            mid_file = ''
            for j in range(len(data)):
                if j != i:
                    mid_file += data[j]
            Neg_files.append(mid_file)
        return Pos_files, Neg_files


# 分割文件
def eval_part(Pos_files, Neg_files, i, path, crossV):
    if crossV != '-1':
        result_s = Pos_files[i] + Neg_files[i]
        result_a = ''
        for j in range(len(Pos_files)):
            if j != i:
                result_a += Pos_files[j]
        for j in range(len(Neg_files)):
            if j != i:
                result_a += Neg_files[j]
        test_name = os.path.join(path, 'test')
        train_name = os.path.join(path, 'train')
        file1 = open(test_name, 'w')
        file1.write(result_s)
        file1.close()
        file2 = open(train_name, 'w')
        file2.write(result_a)
        file2.close()
    else:
        result_s = Pos_files[i]
        result_a = Neg_files[i]
        test_name = os.path.join(path, 'test')
        train_name = os.path.join(path, 'train')
        file1 = open(test_name, 'w')
        file1.write(result_s)
        file1.close()
        file2 = open(train_name, 'w')
        file2.write(result_a)
        file2.close()
    return [test_name, train_name]


# 判定并得到指标
def eval_standerd(true, tfpn, step):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for j in range(step):
        if tfpn[j] == 0.0 and true[j] == 0.0:
            tp += 1
        if tfpn[j] == 1.0 and true[j] == 1.0:
            tn += 1
        if tfpn[j] == 0.0 and true[j] == 1.0:
            fp += 1
        if tfpn[j] == 1.0 and true[j] == 0.0:
            fn += 1
    acc = (tp + tn) / (tp + tn + fp + fn)
    if (tp + fn) == 0 or (tn + fp) == 0 or (tp + fp) == 0 or (tn + fn) == 0:
        return tp, tn, fp, fn, float('%.3f' % acc), float('%.3f' % 0), float('%.3f' % 0), float('%.3f' % 0), float(
            '%.3f' % 0)
    else:
        if step == 1:
            return tp, tn, fp, fn, float('%.3f' % acc), float('%.3f' % 0), float('%.3f' % 0), float('%.3f' % 0), float(
                '%.3f' % 0)
        else:
            sn = tp / (tp + fn)
            sp = tn / (tn + fp)
            ppv = tp / (tp + fp)
            mcc = (tp * tn - fp * fn) / sqrt((tp + fp) * (tn + fn) * (tp + fn) * (tn + fp))
            return tp, tn, fp, fn, float('%.3f' % acc), float('%.3f' % sn), float('%.3f' % sp), float(
                '%.3f' % ppv), float('%.3f' % mcc)


# 求平均
def eval_average(standerd_box):
    new_list = []
    for each_number in range(len(standerd_box[0])):
        m = 0
        for part in standerd_box:
            if len(part) != 0:
                m += part[each_number]
            else:
                m = 0
        m = m / len(standerd_box)
        new_list.append(m)
    new_list[0] = int(new_list[0])
    new_list[1] = int(new_list[1])
    new_list[2] = int(new_list[2])
    new_list[3] = int(new_list[3])
    new_list[4] = float('%.4f' % new_list[4])
    new_list[5] = float('%.4f' % new_list[5])
    new_list[6] = float('%.4f' % new_list[6])
    new_list[7] = float('%.4f' % new_list[7])
    new_list[8] = float('%.4f' % new_list[8])
    return new_list


# 交叉验证
def eval_cv(c_number, gamma, crossV, file):
    if 'mid-cv' not in os.listdir(os.path.split(file)[0]):
        path = os.path.join(os.path.split(file)[0], 'mid-cv')
        os.makedirs(path)
    else:
        path = os.path.join(os.path.split(file)[0], 'mid-cv')
    with open(file, 'r') as inf:  # 读取文件
        data = inf.readlines()
        inf.close()
    Pos_files, Neg_files = eval_index(data, crossV)  # 获取索引
    standerd_box = []
    for i in range(len(Pos_files)):  # 交叉验证
        files = eval_part(Pos_files, Neg_files, i, path, crossV)  # 分割文件
        outfile = files[1] + '.model'
        y, x, model = train_command(files[1], c_number, gamma, outfile)
        y_p, p_label, p_acc, p_val = predict_command(files[0], outfile)
        standerd_num = eval_standerd(y_p, p_label, len(y_p))  # 类名比对，得到指标
        standerd_box.append(standerd_num)
    standerd_list = eval_average(standerd_box)
    shutil.rmtree(path)  # 删除文件夹
    time.sleep(0.05)
    return standerd_list  # 返回指标


# 聚类
def eval_cluster(evaluate_score, evaluate_key):
    cluster_t = []
    cluster_s = []
    t_index = []
    s_index = []
    t_number = []
    s_number = []
    acc_score = {}
    for i in range(len(evaluate_score)):
        acc_score[evaluate_key[i]] = float('%.3f' % evaluate_score[i][4])
    for key in evaluate_key:
        t_tp = key.split('s')[0]
        s_tp = 's' + key.split('s')[1]
        if t_tp not in t_index:
            t_index.append(t_tp)
        if s_tp not in s_index:
            s_index.append(s_tp)
    for i in t_index:
        cluster_t.append(0)
        t_number.append(0)
    for i in s_index:
        cluster_s.append(0)
        s_number.append(0)
    for key in acc_score:
        t_tp = key.split('s')[0]
        s_tp = 's' + key.split('s')[1]
        cluster_t[t_index.index(t_tp)] += acc_score[key]
        cluster_s[s_index.index(s_tp)] += acc_score[key]
        t_number[t_index.index(t_tp)] += 1
        s_number[s_index.index(s_tp)] += 1
    for i in range(len(cluster_t)):
        cluster_t[i] = float('%.4f' % (cluster_t[i] / t_number[i]))
    for i in range(len(cluster_s)):
        cluster_s[i] = float('%.4f' % (cluster_s[i] / s_number[i]))
    return [cluster_t, cluster_s, t_index, s_index]


# 文件训练主程序
def train_main(file, c_number, gamma, out):
    y, x, model = train_command(file, c_number, gamma, out)


# 文件夹训练主程序
def train_f_main(folder, cg_path, out, now_path):
    file_path = os.path.join(now_path, folder)
    if out not in os.listdir(now_path):
        out_folder = os.path.join(now_path, out)
        os.makedirs(out_folder)
    else:
        out_folder = os.path.join(now_path, out)
    cg_box = train_cg(cg_path)
    start_e = 0
    for eachfile in os.listdir(file_path):
        start_e += 1
        extract_time(start_e, len(os.listdir(file_path)))
        out_path = os.path.join(out_folder, eachfile + '.model')
        y, x, model = train_command(folder + eachfile, cg_box[eachfile][0], cg_box[eachfile][-1].strip('\n'), out_path)


# 柱状图
def eval_histogram(data, d_index, d_class, out):
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 汉显
    plt.rcParams['axes.unicode_minus'] = False  # 汉显
    plt.xlabel(d_class, fontsize=10)  # X轴标题
    plt.ylabel('ACC(%)', fontsize=10)  # Y轴标题
    plt.bar(d_index, data)  # 数据
    plt.title('ACC of each ' + d_class)  # 标题
    plt.grid(axis="y", c='g', linestyle='dotted')
    plt.savefig(os.path.join(out, 'ACC_' + d_class + '.png'), dpi=300)  # 保存
    plt.show()


# 密度图
def eval_density(evaluate_score, out):
    values = []
    for key in evaluate_score:
        values.append(float(key[4]) * 100)
    s = pd.Series(values)
    sns.distplot(s, bins=10, hist=False, kde=True, axlabel='ACC')
    plt.savefig(os.path.join(out, 'ACC_Density.png'), dpi=300)
    plt.show()


# 热力图
def eval_heatmap(evaluate_score, evaluate_key, out, t_index, s_index):
    data = {}
    for i in range(len(evaluate_score)):
        data[evaluate_key[i]] = float('%.4f' % (evaluate_score[i][4]))
    map_box = []
    min_num = float(data['t0s20']) * 100
    for key in data:
        if float(data[key]) * 100 > min_num:
            pass
        else:
            min_num = float(data[key]) * 100
    for s in s_index:
        mid_box = []
        for t in t_index:
            if t + s in data:
                mid_box.append(float('%.4f' % data[t + s]) * 100)
            else:
                mid_box.append(min_num - 10)
        map_box.append(mid_box)
    f, ax = plt.subplots(figsize=(12, 12))
    x = np.array(map_box)
    ax.set_title('ACC_Heatmap')
    ax.set_ylabel('Size')
    ax.set_xlabel('Type')
    # cmap = sns.cubehelix_palette(start = 1, rot = 3, gamma=1.1, as_cmap = True)
    sns.heatmap(x, cmap='YlGnBu', annot=True, mask=(x < min_num), vmax=100, linewidths=0.1, square=False,
                xticklabels=True, yticklabels=True)
    ax.set_xticklabels(t_index)
    ax.set_yticklabels(s_index)
    plt.savefig(os.path.join(out, 'ACC_Heatmap.png'), dpi=400)
    plt.show()


# 绘图
def draw_pictures(cluster_t, t_index, out, cluster_s, s_index, evaluate_score, evaluate_key):
    # 柱状图
    eval_histogram(cluster_t, t_index, 'Type', out)
    eval_histogram(cluster_s, s_index, 'Size', out)
    # 密度图
    eval_density(evaluate_score, out)
    # 热图
    eval_heatmap(evaluate_score, evaluate_key, out, t_index, s_index)


# 文件夹交叉验证主程序
def eval_f_main(folder, cg_path, crossV, out, now_path):
    print('')
    in_folder = os.path.join(now_path, folder)
    if out not in os.listdir(now_path):
        out_folder = os.path.join(now_path, out)
        os.makedirs(out_folder)
    cg_box = train_cg(cg_path)
    start_e = 0
    evaluate_score = []
    evaluate_key = []
    for eachfile in os.listdir(in_folder):
        start_e += 1
        extract_time(start_e, len(os.listdir(in_folder)))
        outLine = eval_cv(cg_box[eachfile][0], cg_box[eachfile][-1].strip('\n'), crossV,
                          os.path.join(folder, eachfile))  # 调用k_cv返回交叉验证结果
        evaluate_score.append(outLine)
        evaluate_key.append(eachfile.split('_')[0])
    # 输出文件
    out_lines = ''
    for i in range(len(evaluate_score)):
        line = evaluate_score[i]
        mid_line = evaluate_key[i]
        for j in line:
            mid_line += ',' + str(j)
        out_lines += mid_line + '\n'
    out_lines = 'index,tp,tn,fp,fn,acc,sn,sp,ppv,mcc\n' + out_lines
    with open(os.path.join(out, 'Features_eval.csv'), 'w') as f2:
        f2.write(out_lines)
        f2.close()
    # 聚类
    cluster = eval_cluster(evaluate_score, evaluate_key)
    cluster_t = cluster[0]
    cluster_s = cluster[1]
    t_index = cluster[2]
    s_index = cluster[3]
    draw_pictures(cluster_t, t_index, out, cluster_s, s_index, evaluate_score, evaluate_key)


# 文件交叉验证主程序
def eval_main(file, c_number, gamma, crossV, out, now_path):
    print('')
    out_path = os.path.join(now_path, out + '-eval.txt')
    outLine = eval_cv(c_number, gamma, crossV, file)  # 调用k_cv返回交叉验证结果
    line_num = ''
    for li in range(len(outLine)):
        each_num = outLine[li]
        if li < 4:
            line_num += str(int(each_num)) + '\t'
        else:
            line_num += str('%.3f' % each_num) + '\t'
    lines = 'Model Evaluation\n\ntp\ttn\tfp\tfn\tacc\tsn\tsp\tppv\tmcc\n' + line_num[:-1]
    lines += '\n\n预测成功率(Accuracy, Acc):\n\nAcc = (TP + TN) / (TP + FP + TN + FN)\n\n'
    lines += '敏感度(Sensitivity, SN)也称召回率(Recall, RE):\n\nSn = Recall = TP / (TP + FN)\n\n'
    lines += '特异性(Specificity, SP):\n\nSp = TN / (TN + FP)\n\n'
    lines += '精确率(Precision, PR)也称阳极预测值(Positive Predictive Value, PPV):\n\nPrecision= PPV = TP / (TP + FP)\n\n'
    lines += "Matthew 相关系数(Matthew's correlation coefficient, Mcc):\n\nMCC = (TP*TN- FP*FN)/sqrt((TP + FP)*(TN + FN)*(TP + FN)*(TN + FP)).其中sqrt代表开平方."
    with open(out_path, 'w') as f1:
        f1.write(lines)
        f1.close()


# 文件超参数寻优主程序
def search_main(file):
    command = 'python ' + grid_path + ' ' + file
    pi = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    result = str(pi.stdout.read(), encoding="utf-8").split('  ')
    c_num = result[0].strip('\n').split(' ')[-1]
    gamma = result[1].strip('\n').split(' ')[-1]
    print('\n>>>C_numbr: ' + str(c_num) + '\tGamma: ' + str(gamma) + '\n')


# 文件夹超参数寻优主程序
def search_f_main(folder, now_path):
    print('')
    search_path = os.path.join(now_path, folder)
    key = ''
    start_e = 0
    for eachfile in os.listdir(folder):
        start_e += 1
        extract_time(start_e, len(os.listdir(folder)))
        command = 'python ' + grid_path + ' ' + os.path.join(search_path, eachfile)
        pi = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        result = str(pi.stdout.read(), encoding="utf-8").split('  ')
        c_num = result[0].strip('\n').split(' ')[-1]
        gamma = result[1].strip('\n').split(' ')[-1]
        key += eachfile + '\tC_numbr: ' + str(c_num) + '\tGamma: ' + str(gamma) + '\n'
    with open(os.path.join(now_path, 'Hyperparameter.txt'), 'w') as f2:
        f2.write(key)
        f2.close()


# class FILTER ################################################################

# 欧氏距离
def ojld_distance(p_data, n_data, test_data, number):
    p_distance = []
    n_distance = []
    for line in p_data:
        p_distance.append(math.pow((test_data[number] - line[number]), 2))
    for line in n_data:
        n_distance.append(math.pow((test_data[number] - line[number]), 2))
    return [min(p_distance), min(n_distance)]


# 概率计算
def relief(number, feature_class, feature_line, cycle):
    feature_standard = []
    for line in feature_line:
        mid_box = []
        for i in line:
            mid_box.append(float(i.split(':')[-1]))
        feature_standard.append(mid_box)
    p_data = []
    n_data = []
    for i in range(len(feature_standard)):
        if feature_class[i] == '0':
            p_data.append(feature_standard[i])
        if feature_class[i] == '1':
            n_data.append(feature_standard[i])
    weight = 0
    for m in range(cycle):
        rand_num = random.randint(0, len(feature_standard) - 1)
        if feature_class[rand_num] == '0':
            distance_box = ojld_distance(p_data, n_data, feature_standard[rand_num], number)
            weight += -distance_box[0] + distance_box[1]
        if feature_class[rand_num] == '1':
            distance_box = ojld_distance(p_data, n_data, feature_standard[rand_num], number)
            weight += -distance_box[1] + distance_box[0]
    aver_weight = weight / (m + 1)
    aver_weight = 1 / (1 + math.exp(-aver_weight))
    return aver_weight


# 概率计算
def filter_possible(number, feature_class, feature_line):
    type_both = 0
    type_a = 0
    type_b = 0
    t0 = 0
    t1 = 0
    for i in range(len(feature_class)):
        if feature_class[i] == '0':
            type_a += float(feature_line[i][number].split(':')[-1])
            t0 += 1
        else:
            type_b += float(feature_line[i][number].split(':')[-1])
            t1 += 1
        type_both += float(feature_line[i][number].split(':')[-1])
    avg_0 = type_a / t0
    avg_1 = type_b / t1
    avg_both = type_both / len(feature_class)
    F_son = math.pow(avg_0 - avg_both, 2) + math.pow(avg_1 - avg_both, 2)
    avg_m_0 = 0
    avg_m_1 = 0
    for i in range(len(feature_class)):
        if feature_class[i] == '0':
            avg_m_0 += (math.pow(float(feature_line[i][number].split(':')[-1]) - avg_0, 2))
        else:
            avg_m_1 += (math.pow(float(feature_line[i][number].split(':')[-1]) - avg_1, 2))
    F_mother = avg_m_0 / (t0 - 1) + avg_m_1 / (t1 - 1)
    if F_mother != 0:
        f_score = F_son / F_mother
    else:
        f_score = -0.1
    return f_score


# 排序
def selection_sort(data):
    arr = []
    for i in data:
        arr.append(i)
    index = []
    for i in range(len(arr)):
        index.append(i)
    for i in range(len(arr) - 1):
        min_index = i
        for j in range(i + 1, len(arr)):
            if arr[j] < arr[min_index]:
                min_index = j
        index[min_index], index[i] = index[i], index[min_index]
        arr[min_index], arr[i] = arr[i], arr[min_index]
    # 倒序输出
    re_index = []
    for i in range(len(index) - 1, -1, -1):
        re_index.append(index[i])
    return re_index


# 特征组合测试
def feature_test(d, feature, c_number, gamma, crossV):
    fs_acc = []
    filter_data = []
    for k in d:
        filter_data.append(k[0])
    start_e = 0
    for i in range(len(feature)):
        start_e += 1
        key = feature[i]
        for j in range(len(d)):
            filter_data[j] += ' ' + str(i + 1) + ':' + d[j].split(' ')[key + 1].strip('\n').split(':')[-1]
        out_content = ''
        for n in filter_data:
            out_content += n + '\n'
        with open('mid-ifs', 'w') as ot:
            ot.write(out_content)
            ot.close()
        standard_num = eval_cv(c_number, gamma, crossV, './mid-ifs')
        single_acc = str('%.4f' % (standard_num[4]))
        fs_acc.append(single_acc)
        os.remove('./mid-ifs')
        print('>>>' + str(start_e) + '~' + str(len(feature)))
    return fs_acc


# 绘制折线图
def plot_pl(data, out, type_p):
    x = []
    y = []
    for i in range(len(data)):
        x.append(i + 1)
    for j in data:
        y.append(float(j))
    plt.figure()
    plt.plot(x, y)
    plt.xlabel("Feature")
    plt.ylabel("Acc")
    plt.title(type_p)
    max_x = y.index(max(y))
    max_y = max(y)
    plt.text(max_x, max_y, str(max_x + 1) + '(' + str(max_y * 100) + '%)', fontsize=10)


# 检查数据
def filter_check(data, num):
    mid_box = data
    num = int(num)
    if len(mid_box) != num:
        mb = []
        for i in mid_box:
            mb.append(i.split(':')[0])
        for i in range(num, 0, -1):
            if str(i) not in mb:
                mid_box.insert(mb.index(str(i + 1)), str(i) + ':0')
                mb.insert(mb.index(str(i + 1)), str(i))
    return mid_box


# 特征筛选
def filter_pro(in_path, out_path, c_number, gamma, crossV, cycle, now_path):
    in_path = os.path.join(now_path, in_path)
    out_path = os.path.join(now_path, out_path)
    with open(in_path, 'r') as in_file:  # 读取文件
        d = in_file.readlines()
        in_file.close()
    feature_class = []
    feature_line = []
    for lines in d:
        lines = lines.strip('\n')
        feature_class.append(lines.split(' ')[0])
        mid_box = lines.split(' ')[1:]
        if len(mid_box[-1]) == 0:
            mid_box = mid_box[:-1]
        #mid_box = filter_check(mid_box, num)
        feature_line.append(mid_box)
        out = lines.split(' ')[0] + ' '
        for i in mid_box:
            out += i + ' '
        d[d.index(lines + '\n')] = out + '\n'
    relief_list = []
    start_num = 0
    for each_number in range(len(feature_line[0])):
        start_num += 1
        print('\r>>>' + str(start_num) + '~' + str(len(feature_line[0])), end='', flush=True)
        type_relief = relief(each_number, feature_class, feature_line, int(cycle))  # 求得每个特征relief
        type_fscore = filter_possible(each_number, feature_class, feature_line)  # 求得每个特征f-score
        complex_num = 1 / (math.exp(-type_relief) + 1) + 1 / (math.exp(-type_fscore) + 1)
        relief_list.append(complex_num)
    relief_pool = selection_sort(relief_list)  # 排序
    fs_acc = feature_test(d, relief_pool, c_number, gamma, crossV)
    print('\n特征筛选完成，导出结果中...')
    filter_plot_main(in_path, relief_pool, fs_acc, out_path)
    ifs_content = 'IFS特征排序： '
    for i in relief_pool:
        ifs_content += str(i + 1) + ' '
    with open(out_path + '-ifs.txt', 'w') as f4:
        f4.write(ifs_content)
        f4.close()


# class FILTER PLOT ###########################################################

# 特征分类
def filter_class(standard, size_fs):
    pssm_value = []
    aac_value = []
    kmer_value = []
    kpssm_value = []
    saac_value = []
    sw_value = []
    psepssm_value = []
    pmv = 0
    krv = 0
    acv = 0
    kmv = 0
    sav = 0
    swv = 0
    pev = 0
    for i in standard:
        if i < math.pow(size_fs, 2):
            pmv += 1
        if i >= math.pow(size_fs, 2) and i < (math.pow(size_fs, 2) + 20):
            acv += 1
        if i >= (math.pow(size_fs, 2) + 20) and i < (2 * math.pow(size_fs, 2) + 20):
            krv += 1
        if i >= (2 * math.pow(size_fs, 2) + 20) and i < (3 * math.pow(size_fs, 2) + 20):
            kmv += 1
        if i >= (3 * math.pow(size_fs, 2) + 20) and i < (3 * math.pow(size_fs, 2) + 80):
            sav += 1
        if i >= (3 * math.pow(size_fs, 2) + 80) and i < (4 * math.pow(size_fs, 2) + 80):
            swv += 1
        if i >= (4 * math.pow(size_fs, 2) + 80):
            pev += 1
        pssm_value.append(pmv / (2 * len(standard)))
        kmer_value.append(krv / (2 * len(standard)))
        aac_value.append(acv / (2 * len(standard)))
        kpssm_value.append(kmv / (2 * len(standard)))
        saac_value.append(sav / (2 * len(standard)))
        psepssm_value.append(pev / (2 * len(standard)))
        sw_value.append(swv / (2 * len(standard)))
    return pssm_value, aac_value, kmer_value, kpssm_value, saac_value, sw_value, psepssm_value


# 绘制折线图
def filter_plot_pl(data, pssm_value, aac_value, kmer_value, kpssm_value, saac_value, sw_value, psepssm_value, type_p):
    x = []
    y = []
    for i in range(len(data)):
        x.append(i + 1)
    for j in data:
        y.append(float(j))
    plt.figure()
    plt.plot(x, y, label='ACC')
    plt.plot(x, pssm_value, color='blue', label='PSSMraa')
    plt.plot(x, kmer_value, color='green', label='Kmer')
    plt.plot(x, aac_value, color='yellow', label='OAAC')
    plt.plot(x, kpssm_value, color='red', label='KPSSM')
    plt.plot(x, saac_value, color='pink', label='SAAC')
    plt.plot(x, sw_value, color='orange', label='SW')
    plt.plot(x, psepssm_value, color='gray', label='PSSM-DT')
    plt.legend(bbox_to_anchor=(0., 1.09, 1., .102), loc=0, ncol=4, mode="expand", borderaxespad=0.)
    plt.xlabel("Feature Number")
    plt.ylabel("Acc")
    plt.title(type_p)
    max_x = y.index(max(y))
    max_y = max(y)
    max_pmv = pssm_value[max_x]
    max_krv = kmer_value[max_x]
    max_acv = aac_value[max_x]
    max_kmv = kpssm_value[max_x]
    max_sav = saac_value[max_x]
    max_swv = sw_value[max_x]
    max_pev = psepssm_value[max_x]
    plt.text(max_x, max_y, str(max_x + 1) + '(' + str(max_y * 100) + '%)', fontsize=10)
    plt.text(max_x, max_pmv, str(int(max_pmv * 2 * len(data))), fontsize=6)
    plt.text(max_x, max_krv, str(int(max_krv * 2 * len(data))), fontsize=6)
    plt.text(max_x, max_acv, str(int(max_acv * 2 * len(data))), fontsize=6)
    plt.text(max_x, max_kmv, str(int(max_kmv * 2 * len(data))), fontsize=6)
    plt.text(max_x, max_sav, str(int(max_sav * 2 * len(data))), fontsize=6)
    plt.text(max_x, max_swv, str(int(max_swv * 2 * len(data))), fontsize=6)
    plt.text(max_x, max_pev, str(int(max_pev * 2 * len(data))), fontsize=6)


def filter_plot_main(in_path, relief_pool, fs_acc, out_path):
    size_fs = int(os.path.split(in_path)[-1].split('_')[0].split('s')[-1])
    # ifs
    pssm_value, aac_value, kmer_value, kpssm_value, saac_value, sw_value, psepssm_value = filter_class(relief_pool,
                                                                                                       size_fs)
    filter_plot_pl(fs_acc, pssm_value, aac_value, kmer_value, kpssm_value, saac_value, sw_value, psepssm_value,
                   'IFS-Acc')
    plt.savefig(out_path + '-ifs.png', dpi=500, bbox_inches='tight')


# class PREDICTION ############################################################

# 预测主程序
def predict_main(file, model, out, now_path):
    file_path = os.path.join(now_path, file)
    out_path = os.path.join(now_path, out)
    y_p, x_p = svm_read_problem(file_path)
    model = svm_load_model(model)
    p_label, p_acc, p_val = svm_predict(y_p, x_p, model)
    result = 'True,Predict\n'
    for i in range(len(y_p)):
        result += str(y_p[i]) + ',' + str(p_label[i]) + '\n'
    with open(out_path + '_predict.csv', 'w', encoding="utf-8") as f:
        f.write(result)
        f.close()