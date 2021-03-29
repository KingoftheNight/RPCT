#   导入库    ##################################################################


import os
import subprocess
import math
import copy
readbe_path = os.path.dirname(__file__)


# class READ ##################################################################


# 检查数据
def read_check(data, lf):
    mid_box = data
    if len(mid_box) != int(lf):
        mb = []
        for i in mid_box:
            mb.append(i.split(':')[0])
        for i in range(int(lf) - 1, 0, -1):
            if str(i) not in mb:
                mid_box.insert(mb.index(str(i + 1)), str(i) + ':0')
                mb.insert(mb.index(str(i + 1)), str(i))
    return mid_box


# 数据库拆分
def read_fasta(file, out, now_path):
    if 'Reads' not in os.listdir(now_path):
        root_read = os.path.join(now_path, 'Reads')
        os.makedirs(root_read)
    else:
        root_read = os.path.join(now_path, 'Reads')
    with open(file, 'r') as u:
        lines = u.readlines()
    result = ''
    for i in lines:
        i = i.strip()
        if i:
            if i[0] == '>':
                result = result + '\n' + i + '\n'
            else:
                result = result + i
    result = result[1:]
    order = 0
    filelist = result.split('\n')
    if out not in os.listdir(root_read):
        Path = os.path.join(root_read, out)
        os.makedirs(Path)
    else:
        Path = os.path.join(root_read, out)
    for line in range(len(filelist)):
        if '>' in filelist[line]:
            order += 1
            if line < len(filelist):
                with open(os.path.join(Path, str(order) + '.fasta'), 'w') as f:
                    f.write(filelist[line] + '\n' + filelist[line + 1])


# 生成特征筛选文件
def read_filter(file, filter_index, out, number, lf, now_path):
    # 读取特征排序以及特征文件
    with open(filter_index, 'r', encoding='UTF-8') as f1:
        data = f1.readlines()
        f1.close()
    index = data[0].split(' ')[1:-1]
    with open(file, 'r', encoding='UTF-8') as f2:
        data = f2.readlines()
        f2.close()
    # 提取矩阵特征
    type_f = []
    matrix = []
    for line in data:
        line = line.split(' ')
        type_f.append(line[0])
        mid_box = read_check(line[1:], lf)
        for i in range(len(mid_box)):
            mid_box[i] = mid_box[i].split(':')[-1]
        matrix.append(mid_box)
    # 生成特征筛选文件
    new_matrix = ''
    for i in range(len(matrix)):
        line = matrix[i]
        mid_file = type_f[i]
        order = 0
        for j in range(0, int(number)):
            key = index[j]
            order += 1
            mid_file += ' ' + str(order) + ':' + line[int(key) - 1].strip('\n')
        new_matrix += mid_file + '\n'
    with open(os.path.join(now_path, out + '-fffs'), 'w') as f3:
        f3.write(new_matrix[:-1])
        f3.close()


# 格式化数据库
def read_madb(file, out):
    command = 'makeblastdb -in ' + file + ' -dbtype prot -out ' + out
    outcode = subprocess.Popen(command, shell=True)
    outcode.wait()


# class PSI-BLAST #############################################################


# 调用psi-blast
def psi_blast(file, database, number, ev, out, now_path):
    file = os.path.join(os.path.join(now_path, 'Reads'), file)
    blast_path = os.path.join(readbe_path, 'blastDB')
    if 'PSSMs' not in os.listdir(now_path):
        root_pssm = os.path.join(now_path, 'PSSMs')
        os.makedirs(root_pssm)
    else:
        root_pssm = os.path.join(now_path, 'PSSMs')
    if out not in os.listdir(root_pssm):
        Path = os.path.join(root_pssm, out)
        os.makedirs(Path)
    else:
        Path = os.path.join(root_pssm, out)
    order = 0
    for f in os.listdir(file):
        order += 1
        command = 'psiblast -query ' + os.path.join(file, f) + ' -db ' + os.path.join(blast_path, database) + ' -num_iterations ' + number + ' -evalue ' + ev + ' -out A' + ' -out_ascii_pssm ' + os.path.join(Path, f.split('.')[0])
        outcode = subprocess.Popen(command, shell=True)
        if outcode.wait() == 0:
            print('\r' + str(order) + '\tCompleted\t', end='', flush=True)
        else:
            print('\r' + str(order) + '\tProblems', end='', flush=True)
    if 'A' in os.listdir(now_path):
        os.remove('A')


# class CTD ###################################################################

# AAC
def AAC(PSSM_aaid):
    aa = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    all_features = []
    for j in range(len(PSSM_aaid)):
        line = PSSM_aaid[j]
        aaBox = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i in line:
            if i in aa:
                aaBox[aa.index(i)] += 1
        all_features.append(aaBox)
    return all_features


# SAAC extract long sequence
def saac_el(eachfile):
    dN = 25
    dC = 10
    dN_1 = eachfile[0:4 * dN]
    dC_1 = eachfile[-dC:]
    dL_1 = eachfile[4 * dN:-dC]
    all_elfs = AAC([dN_1, dL_1, dC_1])
    out_box = []
    for i in all_elfs:
        out_box += i
    return out_box


# SAAC extract middle sequence
def saac_em(eachfile):
    dN = 25
    dC = 10
    dN_1 = eachfile[0:4 * dN]
    dC_1 = eachfile[-dC:]
    dL_1 = eachfile[-(dC + 20):-dC]
    all_elfs = AAC([dN_1, dL_1, dC_1])
    out_box = []
    for i in all_elfs:
        out_box += i
    return out_box


# SAAC extract short sequence
def saac_es(eachfile):
    dC = 10
    dN = int((len(eachfile) - dC) / 2)
    dN_1 = eachfile[0:dN]
    dC_1 = eachfile[-dC:]
    dL_1 = eachfile[dN:-dC]
    all_elfs = AAC([dN_1, dL_1, dC_1])
    out_box = []
    for i in all_elfs:
        out_box += i
    return out_box


# saac mian
def saac_main(PSSM_aaid):
    dN = 25
    dC = 10
    all_features = []
    for eachfile in PSSM_aaid:
        if len(eachfile) >= (4 * dN + dC + 20):
            each_fs = saac_el(eachfile)
        if len(eachfile) > (4 * dN + dC) and len(eachfile) < (4 * dN + dC + 20):
            each_fs = saac_em(eachfile)
        if len(eachfile) <= (4 * dN + dC):
            each_fs = saac_es(eachfile)
        all_features.append(each_fs)
    return all_features


# Sliding window extract
def sw_extract(eachfile, eachaaid, lmda):
    eachmatrix = []
    supNum = int((lmda - 1) / 2)
    supMatrix = []
    supAaid = []
    for j in range(supNum):
        mid_box = []
        for k in eachfile[0]:
            mid_box.append(0.0)
        supMatrix.append(mid_box)
        supAaid.append('X')
    newfile = supMatrix + eachfile + supMatrix
    newaaid = supAaid + eachaaid + supAaid
    for j in range(supNum, len(newfile) - supNum):
        select_box = newfile[j - supNum: j + supNum + 1]
        eachmatrix.append([newaaid[j]] + select_box)
    return eachmatrix


# Sliding window plus
def sw_plus(eachmatrix, lmda):
    aa_index = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    matrix_400 = []
    for j in range(len(aa_index)):
        mid_box = []
        for k in eachmatrix[0][1]:
            mid_box.append(0.0)
        matrix_400.append(mid_box)
    for matrix in eachmatrix:
        if matrix[0] in aa_index:
            for line in matrix[1:]:
                for m in range(len(line)):
                    matrix_400[aa_index.index(matrix[0])][m] += line[m]
    for j in range(len(matrix_400)):
        for k in range(len(matrix_400[j])):
            matrix_400[j][k] = float('%.3f' % matrix_400[j][k])
    return matrix_400


# Sliding window reduce
def sw_reduce(raacode, eachplus):
    aa_index = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    reduce_fs = []
    for raa in raacode[1]:
        raa_box = raacode[0][raa]
        # 行合并
        score_box = []
        for i in raa_box:
            score_box.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        for i in range(len(aa_index)):
            for j in range(len(raa_box)):
                if aa_index[i] in raa_box[j]:
                    score_box[j] = extract_row_plus(score_box[j], eachplus[i])
        # 列合并
        type_box = []
        for i in raa_box:
            mid = []
            for j in raa_box:
                mid.append(0)
            type_box.append(mid)
        for i in range(len(aa_index)):
            for j in range(len(raa_box)):
                if aa_index[i] in raa_box[j]:
                    type_box[j] = extract_col_plus(type_box[j], score_box, i)
        type_box = extract_trans(type_box)
        out_box = []
        for i in type_box:
            out_box += i
        reduce_fs.append(out_box)
    return reduce_fs


# Sliding window main
def sw_main(PSSM_matrixes, PSSM_aaid, raacode, lmda):
    Sw_features = []
    start_e = 0
    for i in range(len(PSSM_matrixes)):
        start_e += 1
        extract_time(start_e, len(PSSM_matrixes))
        eachfile = PSSM_matrixes[i]
        eachaaid = PSSM_aaid[i]
        # 提取 L - lmda 个窗口
        eachmatrix = sw_extract(eachfile, eachaaid, int(lmda))
        # 合并20种aa为20x20维矩阵
        eachplus = sw_plus(eachmatrix, int(lmda))
        # 约化矩阵
        reduce_fs = sw_reduce(raacode, eachplus)
        Sw_features.append(reduce_fs)
    return Sw_features


# class KMER ##################################################################

# index
def kmer_index(kmer, box, midbox):
    if int(kmer) > 1:
        new_box = []
        for i in midbox:
            for j in box:
                new_box.append(i + j)
        box = new_box
        return kmer_index(int(kmer) - 1, box, midbox)
    else:
        return box


# kmer
def kmer_function(kmer, fasta, box):
    dict_kmer = {}
    id_list = kmer_index(kmer, box, box)
    for name in id_list:
        dict_kmer[name] = 0
    for k in range(len(fasta) - (int(kmer) - 1)):
        if fasta[k:k + int(kmer)] in dict_kmer:
            dict_kmer[fasta[k:k + int(kmer)]] = dict_kmer[fasta[k:k + int(kmer)]] + 1
    out_file = []
    for key in dict_kmer:
        out_file.append(dict_kmer[key])
    return out_file


# reduce
def kmer_reduce(raa_box, eachfile):
    out_box = []
    for i in eachfile:
        for j in raa_box:
            if i in j:
                out_box.append(j[0])
    simple_raa = []
    for i in raa_box:
        simple_raa.append(i[0])
    return [out_box, simple_raa]


# kmer主程序
def kmer_main(PSSM_aaid, raacode, kmer):
    kmer_features = []
    start_e = 0
    for eachfile in PSSM_aaid:
        start_e += 1
        extract_time(start_e, len(PSSM_aaid))
        mid_raa = []
        for raa in raacode[1]:
            raa_box = raacode[0][raa]
            reduceBox = kmer_reduce(raa_box, eachfile)
            reducefile = reduceBox[0]
            simple_raa = reduceBox[1]
            line = ''
            for i in reducefile:
                line += i
            each_kmer = kmer_function(kmer, line, simple_raa)
            mid_raa.append(each_kmer)
        kmer_features.append(mid_raa)
    return kmer_features


# kpssm DT
def kpssm_DT(eachfile, k):
    out_box = []
    for i in eachfile[0]:
        out_box.append(0)
    for i in range(0, len(eachfile) - k):
        now_line = eachfile[i]
        next_line = eachfile[i + k]
        for j in range(len(now_line)):
            out_box[j] += now_line[j] * next_line[j]
    for i in range(len(out_box)):
        out_box[i] = out_box[i] / (len(eachfile) - k - 1)
    return out_box


# kpssm DDT
def kpssm_DDT(eachfile, k, aa_index):
    out_box = []
    for i in range((len(aa_index) - 1) * len(aa_index)):
        out_box.append(0)
    for i in range(0, len(eachfile) - k):
        now_line = eachfile[i]
        next_line = eachfile[i + k]
        n = -1
        for j in range(len(aa_index)):
            next_aa = copy.deepcopy(aa_index)
            now_aa = aa_index[j]
            next_aa.pop(next_aa.index(now_aa))
            for m in next_aa:
                n += 1
                out_box[n] += now_line[now_aa] * next_line[m]
    for i in range(len(out_box)):
        out_box[i] = out_box[i] / (len(eachfile) - k - 1)
    return out_box


# kpssm reduce
def kpssm_reduce(eachfile, raa_box):
    aa_index = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    # 列合并
    type_box = []
    for i in eachfile:
        mid = []
        for j in raa_box:
            mid.append(0)
        type_box.append(mid)
    for li in range(len(eachfile)):
        for i in range(len(aa_index)):
            for j in range(len(raa_box)):
                if aa_index[i] in raa_box[j]:
                    type_box[li][j] += eachfile[li][i]
    # 列平均
    for i in range(len(type_box)):
        line = type_box[i]
        for j in range(len(line)):
            type_box[i][j] = type_box[i][j] / len(raa_box[j])
    return type_box


# k_gap_PSSM主程序
def kpssm_main(PSSM_matrixes, raacode):
    kpssm_features = []
    start_e = 0
    for eachfile in PSSM_matrixes:
        start_e += 1
        extract_time(start_e, len(PSSM_matrixes))
        mid_matrix = []
        for raa in raacode[1]:
            raa_box = raacode[0][raa]
            reducefile = kpssm_reduce(eachfile, raa_box)
            ddt_index = []
            for i in range(len(raa_box)):
                ddt_index.append(i)
            K = 3
            # DT
            dt_fs = kpssm_DT(reducefile, K)
            # DDT
            ddt_fs = kpssm_DDT(reducefile, K, ddt_index)
            mid_matrix.append(dt_fs + ddt_fs)
        kpssm_features.append(mid_matrix)
    return kpssm_features


# DT PSSM Reduce index
def dtpssm_reduce(raa_box):
    out_box = []
    for i in raa_box:
        out_box.append(i[0])
    return out_box


# top 1 gram
def dtpssm_top_1(reducefile, reduce_index):
    out_box = []
    for line in reducefile:
        out_box.append(reduce_index[line.index(max(line))])
    return out_box


# d 0
def dtpssm_0(fs_top_1, reduce_index):
    out_box = []
    for i in reduce_index:
        out_box.append(0)
    for i in fs_top_1:
        if i in reduce_index:
            out_box[reduce_index.index(i)] += 1
    return out_box


# d n
def dtpssm_n(fs_top_1, reduce_index, d):
    out_box = []
    out_index = []
    for i in reduce_index:
        for j in reduce_index:
            out_box.append(0)
            out_index.append(i + j)
    for i in range(len(fs_top_1) - d):
        if fs_top_1[i] + fs_top_1[i + d] in out_index:
            out_box[out_index.index(fs_top_1[i] + fs_top_1[i + d])] += 1
    return out_box


# PsePSSM主程序
def DT_PSSM_main(PSSM_matrixes, raacode, D):
    DT_features = []
    start_e = 0
    for eachfile in PSSM_matrixes:
        start_e += 1
        extract_time(start_e, len(PSSM_matrixes))
        mid_matrix = []
        for raa in raacode[1]:
            raa_box = raacode[0][raa]
            reducefile = kpssm_reduce(eachfile, raa_box)
            reduce_index = dtpssm_reduce(raa_box)
            # top_1_gram
            fs_top_1 = dtpssm_top_1(reducefile, reduce_index)
            # d_0
            fs_0 = dtpssm_0(fs_top_1, reduce_index)
            # d_n
            fs_n = []
            for d in range(1, D + 1):
                each_fs_n = dtpssm_n(fs_top_1, reduce_index, d + 1)
                fs_n += each_fs_n
            mid_matrix.append(fs_0 + fs_n)
        DT_features.append(mid_matrix)
    return DT_features


# class EPMF ##################################################################


# 提取约化密码表
def extract_raa(r_path):
    with open(r_path, 'r') as code:
        raaCODE = code.readlines()
        code.close()
    raa_dict = {}
    raa_index = []
    for eachline in raaCODE:
        each_com = eachline.split()
        for command in each_com:
            raa_com = each_com[-1].split('-')
            raa_type = each_com[1]
            raa_size = each_com[3]
            raa_ts = 't' + raa_type + 's' + raa_size
            raa_dict[raa_ts] = raa_com
        raa_index.append(raa_ts)
    return raa_dict, raa_index


# time
def extract_time(start_e, end_e):
    print('\r>>>' + str(start_e) + "~" + str(end_e), end='', flush=True)


# 归一化函数
def extract_mathFuction(nextBox):
    outBox = []
    for j in nextBox:
        a = int(j)
        b = -1 * a
        x = 1 / (math.exp(b) + 1)
        outBox.append('%.3f' % x)
    return outBox


# 读取PSSM矩阵文件
def extract_read(path, PSSM_matrixes, PSSM_aaid, PSSM_type, type_p):
    start_e = 0
    for i in range(len(os.listdir(path))):
        start_e += 1
        extract_time(start_e, len(os.listdir(path)))
        eachfile = os.listdir(path)[i]
        with open(os.path.join(path, eachfile), 'r') as f1:
            data = f1.readlines()
            f1.close()
        matrix = []
        aa_id = []
        for j in data:
            if 'Lambda' in j and 'K' in j:
                end_matrix = data.index(j)
        for eachline in data[3:end_matrix - 1]:
            row = eachline.split()
            newrow = row[0:22]
            for i in range(2, len(newrow)):
                newrow[i] = int(newrow[i])
            nextrow = extract_mathFuction(newrow[2:])
            matrix.append(nextrow)
            aa_id.append(newrow[1])
        PSSM_matrixes.append(matrix)
        PSSM_aaid.append(aa_id)
        PSSM_type.append(type_p)
    original_pssm = (copy.deepcopy(PSSM_matrixes), copy.deepcopy(PSSM_aaid), copy.deepcopy(PSSM_type))
    return original_pssm


# 提取矩阵特征
def extract_features(PSSM_matrixes, PSSM_aaid):
    all_features = []
    start_e = 0
    for i in range(len(PSSM_matrixes)):
        start_e += 1
        extract_time(start_e, len(PSSM_matrixes))
        each_matrix = PSSM_matrixes[i]
        matrix_400 = []
        aa_index = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        for aa in aa_index:
            aa_score = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,0.0]
            for j in range(len(each_matrix)):
                line = each_matrix[j]
                if PSSM_aaid[i][j] == aa:
                    for k in range(len(line)):
                        aa_score[k] = aa_score[k] + line[k]
            matrix_400.append(aa_score)
        all_features.append(matrix_400)
    return all_features


# 矩阵行相加
def extract_row_plus(data1, data2):
    new_data = []
    for i in range(len(data1)):
        new_data.append(data1[i] + data2[i])
    return new_data


# 矩阵列相加
def extract_col_plus(tp_box, score_box, i):
    for j in range(len(score_box)):
        line = score_box[j]
        tp_box[j] = tp_box[j] + line[i]
    return tp_box


# 矩阵行平均
def extract_rc_average(data, raa_box):
    new_data = []
    for x in range(len(raa_box)):
        lenth = len(raa_box[x])
        line = data[x]
        new_line = []
        for i in line:
            new_line.append(i / lenth)
        new_data.append(new_line)
    return new_data


# 转置
def extract_trans(type_box):
    new_box = []
    for i in range(len(type_box)):
        col = []
        for j in type_box:
            col.append(j[i])
        new_box.append(col)
    return new_box


# 约化矩阵
def extract_reduce(pssm_features, raacode, PSSM_type):
    all_features = []
    aa_index = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    start_e = 0
    for raa in raacode[1]:
        start_e += 1
        extract_time(start_e, len(raacode[1]))
        raa_box = raacode[0][raa]
        mid_box = []
        for k in range(len(pssm_features)):
            eachfile = pssm_features[k]
            eachtype = PSSM_type[k]
            # 行合并
            score_box = []
            for i in raa_box:
                score_box.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            for i in range(len(aa_index)):
                for j in range(len(raa_box)):
                    if aa_index[i] in raa_box[j]:
                        score_box[j] = extract_row_plus(score_box[j], eachfile[i])
            # 列合并
            type_box = []
            for i in raa_box:
                mid = []
                for j in raa_box:
                    mid.append(0)
                type_box.append(mid)
            for i in range(len(aa_index)):
                for j in range(len(raa_box)):
                    if aa_index[i] in raa_box[j]:
                        type_box[j] = extract_col_plus(type_box[j], score_box, i)
            # 转置
            type_box = extract_trans(type_box)
            mid_box.append([eachtype, type_box])
        all_features.append(mid_box)
    return all_features


# 矩阵转置
def extract_transform(data):
    new_data = []
    for i in data:
        new_data += i
    return new_data


# 归一化
def extract_scale(data):
    new_data = []
    for i in data:
        if i >= - 709:
            f = 1 / (1 + math.exp(-i))
            new_data.append(f)
        else:
            f = 1 / (1 + math.exp(-709))
            new_data.append(f)
        # new_data.append('%.6f' % f)
    return new_data


# str to float
def extract_change(PSSM_matrixes):
    out_box = []
    for i in range(len(PSSM_matrixes)):
        mid_box = []
        for j in range(len(PSSM_matrixes[i])):
            next_box = []
            for k in range(len(PSSM_matrixes[i][j])):
                next_box.append(float(PSSM_matrixes[i][j][k]))
            mid_box.append(next_box)
        out_box.append(mid_box)
    return out_box


# long to short
def extract_short(pssm_features):
    out_box = []
    for i in range(len(pssm_features)):
        mid_box = []
        for j in range(len(pssm_features[i])):
            next_box = []
            for k in range(len(pssm_features[i][j])):
                next_box.append(float('%.3f' % pssm_features[i][j][k]))
            mid_box.append(next_box)
        out_box.append(mid_box)
    return out_box


# AAC开平方
def extract_sqaac(data):
    new_data = []
    for i in data:
        new_data.append(math.sqrt(i))
    return new_data


# 保存特征
def extract_save(raa_features, AAC_features, Kmer_features, Kpssm_features, SAAC_features, Sw_features, dtPSSM_features,
                 outfolder, raa_list):
    start_e = 0
    for k in range(len(raa_features)):
        start_e += 1
        extract_time(start_e, len(raa_features))
        eachraa = raa_features[k]
        out_file = ''
        for i in range(len(eachraa)):
            eachfile = eachraa[i]
            eachaac = AAC_features[i]
            eachkpssm = Kpssm_features[i][k]
            eachkmer = Kmer_features[i][k]
            eachsaac = SAAC_features[i]
            eachsw = Sw_features[i][k]
            eachdtpssm = dtPSSM_features[i][k]
            type_m = eachfile[0]
            data_m = eachfile[1]
            data_m = extract_transform(data_m) + eachaac + eachkmer + eachkpssm + eachsaac + eachsw + eachdtpssm
            # 归一化
            data_m = extract_scale(data_m)
            mid_file = type_m
            for j in range(len(data_m)):
                mid_file += ' ' + str(j + 1) + ':' + str(data_m[j])
            out_file += mid_file + '\n'
        path = os.path.join(outfolder, raa_list[k] + '_rpct.fs')
        with open(path, 'w') as f2:
            f2.write(out_file)
            f2.close()


# 读取指令
def extract_main(positive, negative, outfolder, raa, lmda, now_path):
    # 获取氨基酸约化密码表
    raa_path = os.path.join(readbe_path, 'raacDB')
    raacode = extract_raa(os.path.join(raa_path, raa))
    # 处理地址
    positive = os.path.join(os.path.join(now_path, 'PSSMs'), positive)
    negative = os.path.join(os.path.join(now_path, 'PSSMs'), negative)
    if outfolder not in os.listdir(now_path):
        outfolder = os.path.join(now_path, outfolder)
        os.makedirs(outfolder)
    else:
        outfolder = os.path.join(now_path, outfolder)
    PSSM_matrixes = []
    PSSM_aaid = []
    PSSM_type = []
    # positive
    positive_tube = extract_read(positive, PSSM_matrixes, PSSM_aaid, PSSM_type, '0')
    # negative
    negative_tube = extract_read(negative, positive_tube[0], positive_tube[1], positive_tube[2], '1')
    # PSSM特征提取
    PSSM_matrixes = negative_tube[0]
    PSSM_matrixes = extract_change(PSSM_matrixes)
    PSSM_aaid = negative_tube[1]
    PSSM_type = negative_tube[2]
    pssm_features = extract_features(PSSM_matrixes, PSSM_aaid)
    pssm_features = extract_short(pssm_features)
    # AAC特征提取
    AAC_features = AAC(PSSM_aaid)
    # SAAC特征提取
    SAAC_features = saac_main(PSSM_aaid)
    # kmer特征提取
    Kmer_features = kmer_main(PSSM_aaid, raacode, 2)
    # kpssm特征提取
    Kpssm_features = kpssm_main(PSSM_matrixes, raacode)
    # psepssm特征提取
    dtPSSM_features = DT_PSSM_main(PSSM_matrixes, raacode, 3)
    # PSSM滑窗
    Sw_features = sw_main(PSSM_matrixes, PSSM_aaid, raacode, lmda)
    # 矩阵约化
    raa_features = extract_reduce(pssm_features, raacode, PSSM_type)
    # 生成特征文件
    extract_save(raa_features, AAC_features, Kmer_features, Kpssm_features, SAAC_features, Sw_features, dtPSSM_features,
                 outfolder, raacode[1])
