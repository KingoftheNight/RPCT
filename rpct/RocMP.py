# 调用库 ######################################################################


import os
import shutil
import copy
import math
import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn import svm as SVM
from sklearn.metrics import roc_curve, auc
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import sys
rocmp_path = os.path.dirname(__file__)
sys.path.append(rocmp_path)
from TrainFP import train_command, predict_command, eval_cv


# class MCOMBINE ##############################################################

# 路径修改
def mc_folder(path):
    if path[-1] == '/':
        pass
    else:
        path += '/'
    return path


# 排序
def mc_sort(acc_value):
    arr = []
    for i in acc_value:
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


# 特征评估文件读取
def mc_evaluate(eval_file, member, train_fs, predict_fs):
    with open(eval_file, 'r') as ef:
        data = ef.readlines()
        ef.close()
    eval_key = []
    acc_value = []
    for each in data[1:]:
        eachLine = each.split(',')
        eval_key.append(eachLine[0])
        acc_value.append(float(eachLine[5]))
    acc_sort = mc_sort(acc_value)
    t_fs_path = []
    p_fs_path = []
    fs_key = []
    for i in range(member):
        t_fs_path.append(os.path.join(train_fs, eval_key[acc_sort[i]] + '_rpct.fs'))
        p_fs_path.append(os.path.join(predict_fs, eval_key[acc_sort[i]] + '_rpct.fs'))
        fs_key.append(eval_key[acc_sort[i]] + '_rpct.fs')
    return [t_fs_path, p_fs_path, fs_key]


# 超参数文件读取
def mc_hyper(fs_key, cg_file):
    with open(cg_file, 'r') as hf:
        data = hf.readlines()
        hf.close()
    cg_box = []
    for i in fs_key:
        cg_box.append(0)
    for each in data:
        eachLine = each.split('	')
        c_num = float(eachLine[1].split(': ')[-1])
        gamma = float(eachLine[2].strip('\n').split(': ')[-1])
        if eachLine[0] in fs_key:
            cg_box[fs_key.index(eachLine[0])] = [c_num, gamma]
    return cg_box


# 训练模型并复制预测特征文件
def mc_train(t_fs_path, p_fs_path, fs_key, cg_box, now_path):
    if 'Intlen_predict' not in os.listdir(now_path):
        intlen_path = os.path.join(now_path, 'Intlen_predict')
        os.makedirs(intlen_path)
    else:
        intlen_path = os.path.join(now_path, 'Intlen_predict')
    test_model = []
    test_features = []
    for i in range(len(t_fs_path)):
        each_t_fs = t_fs_path[i]
        each_t_ms = os.path.join(intlen_path, fs_key[i] + '.model')
        test_model.append(each_t_ms)
        each_p_fs = p_fs_path[i]
        each_p_ms = os.path.join(intlen_path, fs_key[i])
        test_features.append(each_p_ms)
        each_c_num = cg_box[i][0]
        each_g_num = cg_box[i][1]
        y, x, model = train_command(each_t_fs, str(each_c_num), str(each_g_num), each_t_ms)
        shutil.copyfile(each_p_fs,each_p_ms)
    return test_model, test_features


# 模型预测
def mc_predict(test_model, test_features):
    predict_value = []
    for i in range(len(test_model)):
        each_model = test_model[i]
        each_feature = test_features[i]
        y_p, p_label, p_acc, p_val = predict_command(each_feature, each_model)
        predict_value.append(p_label)
    true_value = y_p
    return true_value, predict_value


# 多数投票
def mc_vote(true_value, predict_value):
    out_value = []
    for i in range(len(true_value)):
        mid_value = []
        for each_value in predict_value:
            mid_value.append(each_value[i])
        number_0 = 0
        number_1 = 0
        for j in mid_value:
            if j == 0.0:
                number_0 += 1
            else:
                number_1 += 1
        if number_0 > number_1:
            out_value.append(0.0)
        else:
            out_value.append(1.0)
    # 集合模型评估
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for j in range(len(true_value)):
        if true_value[j] == 0.0 and out_value[j] == 0.0:
            tp += 1
        if true_value[j] == 1.0 and out_value[j] == 1.0:
            tn += 1
        if true_value[j] == 0.0 and out_value[j] == 1.0:
            fp += 1
        if true_value[j] == 1.0 and out_value[j] == 0.0:
            fn += 1
    acc = (tp + tn) / (tp + tn + fp + fn)
    sn = tp / (tp + fn)
    sp = tn / (tn + fp)
    ppv = tp / (tp + fp)
    mcc = (tp * tn - fp * fn) / math.sqrt((tp + fp) * (tn + fn) * (tp + fn) * (tn + fp))
    # 结果输出
    out_line = str(tp) + '\t' + str(tn) + '\t' + str(fp) + '\t' + str(fn) + '\t' + str('%.3f' % acc) + '\t' + str(
        '%.3f' % sn) + '\t' + str('%.3f' % sp) + '\t' + str('%.3f' % ppv) + '\t' + str('%.3f' % mcc)
    lines = 'Model Evaluation\n\ntp\ttn\tfp\tfn\tacc\tsn\tsp\tppv\tmcc\n' + out_line
    lines += '\n\n预测成功率(Accuracy, Acc):\n\nAcc = (TP + TN) / (TP + FP + TN + FN)\n\n'
    lines += '敏感度(Sensitivity, SN)也称召回率(Recall, RE):\n\nSn = Recall = TP / (TP + FN)\n\n'
    lines += '特异性(Specificity, SP):\n\nSp = TN / (TN + FP)\n\n'
    lines += '精确率(Precision, PR)也称阳极预测值(Positive Predictive Value, PPV):\n\nPrecision= PPV = TP / (TP + FP)\n\n'
    lines += "Matthew 相关系数(Matthew's correlation coefficient, Mcc):\n\nMCC = (TP*TN- FP*FN)/sqrt((TP + FP)*(TN + FN)*(TP + FN)*(TN + FP)).其中sqrt代表开平方."
    return lines


# 模型集合投票主程序
def model_combine_main(train_fs, predict_fs, eval_file, cg_file, member, now_path):
    # 参数格式修改
    train_fs = os.path.join(now_path, train_fs)
    predict_fs = os.path.join(now_path, predict_fs)
    eval_file = os.path.join(now_path, eval_file)
    cg_file = os.path.join(now_path, cg_file)
    member = int(member)
    # 读取评估文件
    features_files_path = mc_evaluate(eval_file, member, train_fs, predict_fs)
    t_fs_path = features_files_path[0]
    p_fs_path = features_files_path[1]
    fs_key = features_files_path[2]
    # 读取超参数文件
    cg_box = mc_hyper(fs_key, cg_file)
    # 训练模型
    test_model, test_features = mc_train(t_fs_path, p_fs_path, fs_key, cg_box, now_path)
    # 模型预测
    true_value, predict_value = mc_predict(test_model, test_features)
    # 多数投票
    out_content = mc_vote(true_value, predict_value)
    with open(os.path.join(now_path, 'Integrated_learning_' + str(member) + '.txt'), 'w') as lf:
        lf.write(out_content)
        lf.close()


# class PCA ###################################################################

# time
def PCA_time(start_e, end_e):
    print('\r>>>' + str(start_e) + "~" + str(end_e), end='', flush=True)


# 读取特征文件
def PCA_read(in_path):
    with open(in_path, 'r') as f1:
        data = f1.readlines()
        f1.close()
    index = []
    out_box = []
    for line in data:
        each_box = line.strip('\n').split(' ')
        index.append(each_box[0])
        mid_box = [float(each_box[0])]
        for i in each_box[1:]:
            mid_box.append(float(i.split(':')[-1]))
        out_box.append(mid_box)
    out_box = np.array(out_box)
    return out_box, index


def PCA_scale_data(X_train):
    min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = min_max_scaler.fit(X_train)
    X_train_ = scaler.transform(X_train)
    return X_train_


def PCA_selection(data):
    X = data.iloc[:, 1:]
    X_s = PCA_scale_data(X)
    pca = PCA(n_components=1)
    pca.fit(X_s)
    pc1_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    pc1_FeatureScore = pd.DataFrame({'Feature': X.columns,
                                     'PC1_loading': pc1_loadings.T[0],
                                     'PC1_loading_abs': abs(pc1_loadings.T[0])})
    pc1_FeatureScore = pc1_FeatureScore.sort_values('PC1_loading_abs', ascending=False)
    data_train = data.reindex(['Label']+list(pc1_FeatureScore['Feature']), axis=1)
    feature_selection = []
    for i in pc1_FeatureScore['Feature']:
        feature_selection.append(i)
    return feature_selection, data_train


# 主成分分析
def PCA_train(features_data):
    features_array = np.array(features_data)
    cols_avg = features_array.mean(axis=0)
    array_avg = np.cov(features_array, rowvar=False)
    for x in range(len(features_array)):
        for y in range(len(features_data[0])):
            features_array[x, y] = features_array[x, y] - cols_avg[y]
    fs_values, fs_matrix = np.linalg.eig(array_avg)
    E1 = np.argsort(fs_values)
    E1 = E1[::-1]
    return features_array, fs_matrix, E1


# 数组转换列表
def PCA_change(result_matrix):
    mid_array = np.array(result_matrix)
    mid_array = list(mid_array)
    mid_list = []
    for line in mid_array:
        mid_box = []
        for j in line:
            mid_box.append(float(round(j, 6)))
        mid_list.append(mid_box)
    out_box = []
    for i in mid_list:
        mid_box = []
        for j in range(len(i)):
            mid_box.append(str(j + 1) + ':' + str(i[j]))
        out_box.append(mid_box)
    return out_box


# 输出降维特征文件
def PCA_file(result_list, features_index, now_path):
    out_file = ''
    for i in range(len(features_index)):
        mid_file = features_index[i]
        for j in result_list[i]:
            mid_file += ' ' + j
        out_file += mid_file + '\n'
    mid_ifs = os.path.join(now_path, 'mid-ifs')
    with open(mid_ifs, 'w') as f2:
        f2.write(out_file)
        f2.close()
    return mid_ifs


# PCA plot
def PCA_plot(all_fs_acc):
    x = []
    y = []
    for i in range(len(all_fs_acc)):
        x.append(i + 1)
    for j in all_fs_acc:
        y.append(float(j))
    plt.figure()
    plt.plot(x, y)
    plt.xlabel("Featurer Dimension")
    plt.ylabel("ACC")
    plt.title('PCA-ACC')
    max_x = y.index(max(y))
    max_y = max(y)
    plt.text(max_x, max_y, str(max_x + 1) + '(' + str(max_y * 100) + '%)', fontsize=10)


# 主成分分析主程序
def PCA_main(in_path, out_path, c_number, gamma, crossV, now_path):
    # 读取文件
    in_path = os.path.join(now_path, in_path)
    out_path = os.path.join(now_path, out_path)
    features_data, feature_index = PCA_read(in_path)
    # PCA
    features_array = pd.DataFrame(features_data)
    fs_matrix, fs_data = PCA_selection(features_array)
    len_fs = len(fs_matrix)
    # 降维
    all_fs_acc = []
    for i in range(1, len_fs+1):
        PCA_time(i, len_fs)
        select_matrix = fs_matrix[0:i]
        result_matrix = fs_data[select_matrix]
        result_list = PCA_change(result_matrix)
        # 输出特征文件
        mid_ifs = PCA_file(result_list, feature_index, now_path)
        # 交叉验证
        standard_num = eval_cv(c_number, gamma, crossV, mid_ifs)
        single_acc = str('%.3f' % (standard_num[4]))
        all_fs_acc.append(single_acc)
        os.remove(mid_ifs)
    PCA_plot(all_fs_acc)
    plt.savefig(out_path + '-pca.png', dpi=400)
    out_file = 'IFS特征排序： '
    for j in fs_matrix:
        out_file += str(j) + ' '
    with open(out_path + '-pca.txt','w',encoding='UTF-8') as f:
        f.write(out_file)
        f.close()


# class ROC ###################################################################

# 检查数据
def roc_check(data):
    mid_box = data
    if len(mid_box) != 400:
        mb = []
        for i in mid_box:
            mb.append(i.split(':')[0])
        for i in range(400, 0, -1):
            if str(i) not in mb:
                mid_box.insert(mb.index(str(i + 1)), str(i) + ':0')
                mb.insert(mb.index(str(i + 1)), str(i))
    return mid_box


# 检查数据
def roc_filter_check(data, num):
    mid_box = data
    if len(mid_box) != int(num):
        mb = []
        for i in mid_box:
            mb.append(i.split(':')[0])
        for i in range(int(num), 0, -1):
            if str(i) not in mb:
                mid_box.insert(mb.index(str(i + 1)), str(i) + ':0')
                mb.insert(mb.index(str(i + 1)), str(i))
    return mid_box


# 读取文件
def roc_read(roc_filename):
    with open(roc_filename, 'r') as f1:
        file = f1.readlines()
        f1.close()
    # 提取特征list
    features = []
    features_label = []
    for i in file:
        line = i.strip('\n').split(' ')
        fs_box = roc_check(line[1:])
        # fs_box = line[1:]
        mid_box = []
        for j in fs_box:
            mid_box.append(float(j.split(':')[-1]))
        features.append(mid_box)
        features_label.append(int(line[0]))
    # 转换为数组
    af_data = np.array(features)
    af_label = np.array(features_label)
    return (af_data, af_label)


# 读取文件
def roc_filter_read(roc_filename):
    with open(roc_filename, 'r') as f1:
        file = f1.readlines()
        f1.close()
    # 提取特征list
    features = []
    features_label = []
    for i in file:
        line = i.strip('\n').split(' ')
        fs_box = line[1:]
        mid_box = []
        for j in fs_box:
            mid_box.append(float(j.split(':')[-1]))
        features.append(mid_box)
        features_label.append(int(line[0]))
    # 转换为数组
    af_data = np.array(features)
    af_label = np.array(features_label)
    return (af_data, af_label)


# svm分类训练
def roc_svm(af_data, af_label, c_number, ga):
    # 分割数据
    train_data, test_data, train_label, test_label = model_selection.train_test_split(af_data, af_label, test_size=.3,
                                                                                      random_state=0)
    # svm分类训练
    roc_svm = SVM.SVC(kernel='rbf', C=c_number, gamma=ga, probability=True)
    test_predict_label = roc_svm.fit(train_data, train_label).decision_function(test_data)
    # roc坐标获取
    fpr, tpr, threshold = roc_curve(test_label, test_predict_label)
    roc_auc = auc(fpr, tpr)
    return (fpr, tpr, roc_auc)


# 绘制roc曲线
def roc_draw(fpr, tpr, roc_auc, out):
    plt.figure()
    plt.figure(figsize=(10, 10))
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig(out + '-roc.png', dpi=300)


# ROC CURVE COMMAND
def roc_graph(file, out, c_number, ga, now_path):
    roc_features = roc_filter_read(os.path.join(now_path, file))
    roc_tube = roc_svm(roc_features[0], roc_features[1], float(c_number), float(ga))
    roc_draw(roc_tube[0], roc_tube[1], roc_tube[2], os.path.join(now_path, out))


# class REDUCE AMINO ACIDS ####################################################

# 欧氏距离
def euclidean_distance(a, b):
    sq = 0
    for i in range(len(a)):
        sq += (a[i] - b[i]) * (a[i] - b[i])
    distance = math.sqrt(sq)
    return distance


# pre_c
def pre_r(euclidean_box, index):
    mid_list = []
    for key in euclidean_box:
        mid_list.append(euclidean_box[key])
    mid_list.sort()
    box = []
    for each_num in mid_list:
        for key in euclidean_box:
            if euclidean_box[key] == each_num:
                if key not in box:
                    box.append(key)
    pre_raac = []
    for m in mid_list:
        for key in euclidean_box:
            if euclidean_box[key] == m:
                if [key.split("&")[0], key.split("&")[1]] not in pre_raac:
                    pre_raac.append([key.split("&")[0], key.split("&")[1]])
    reduce_list = []
    aa_raac = copy.deepcopy(pre_raac)
    aa20 = copy.deepcopy(index)
    for i in aa_raac[:190]:
        if i[0] in aa20 and i[1] in aa20:
            aa20.remove(str(i[0]))
            aa20.remove(str(i[1]))
            aa20.append(i)
        else:
            p = 0
            q = 0
            if i[0] in aa20:
                aa20.remove(str(i[0]))
            if i[1] in aa20:
                aa20.remove(str(i[1]))
            for j in range(len(aa20)):
                if len(aa20[j]) == 1:
                    pass
                else:
                    if i[0] in aa20[j] or i[1] in aa20[j]:
                        p += 1
                        if p == 1:
                            if i[0] not in aa20[j]:
                                aa20[j].append(str(i[0]))
                            if i[1] not in aa20[j]:
                                aa20[j].append(str(i[1]))
                            q = copy.deepcopy(j)
                        else:
                            for k in aa20[j]:
                                if k not in aa20[q]:
                                    aa20[q].append(k)
                            aa20.remove(aa20[j])
                            break
        result = ""
        for amp in aa20:
            if len(amp) != 1:
                for poi in amp:
                    result += poi
                result += "-"
            else:
                result += amp + "-"
        result = result[:-1]
        if result not in reduce_list:
            reduce_list.append(str(result))
    return reduce_list


# 约化氨基酸
def reduce(data):
    index = ['A', 'L', 'R', 'K', 'N', 'M', 'D', 'F', 'C', 'P', 'Q', 'S', 'E', 'T', 'G', 'W', 'H', 'Y', 'I', 'V']
    name = data[0]
    data = data[1:]
    euclidean_box = {}
    print('>>>>>>>>>>>>>>>>>')
    m = 0
    for i in range(20):
        m += 1
        for j in range(m, 20):
            tube_a = []
            tube_b = []
            tube_a.append(float(data[i]))
            tube_b.append(float(data[j]))
            # 欧氏距离
            distance = euclidean_distance(tube_a, tube_b)
            print(index[i] + "&" + index[j] + ' 欧式距离：' + str('%.4f' % distance))
            euclidean_box[index[i] + "&" + index[j]] = str('%.4f' % distance)
            time.sleep(0.01)
    print('>>>>>>>>>>>>>>>>>')
    final = []
    reduce_list = pre_r(euclidean_box, index)
    for y in reduce_list:
        result = name + " 1 size " + str(len(y.split("-"))) + " " + y
        final.insert(0, result)
    final = final[1:]
    return final


# 读取指令
def res_main(resp_id):
    aaindex_path = os.path.join(os.path.join(rocmp_path, 'aaindexDB'), 'AAindex.txt')
    with open(aaindex_path, 'r', encoding='UTF-8') as f:
        resp_data = f.readlines()
        f.close()
    request = ''
    for i in resp_data:
        if resp_id in i:
            request = i
    if len(request) == 0:
        print('We do not find the ID:' + resp_id)
        return
    print("\n约化中...")
    final = reduce(request.split('\t'))
    print("约化完成.")
    time.sleep(0.5)
    # 打印约化列表
    print('>>>>>>>>>>>>>>>>>')
    for j in final:
        time.sleep(0.05)
        print(j)
    print('>>>>>>>>>>>>>>>>>')
    print("结果打印完毕.")