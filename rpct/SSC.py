from pyecharts.charts import Sankey
from pyecharts import options as opts
import os
from itertools import chain
SSC_path = os.path.dirname(__file__)


# 读取约化文件
def SSC_read(raa_file, type_r):
    with open(raa_file, "r") as f:
        data = f.readlines()
    out_box = []
    for line in data:
        line = line.strip("\n").split(" ")
        if line[1] == type_r:
            out_box.append(line[4])
    all_sq = ""
    for i in out_box[0]:
        if i != "-":
            all_sq += i + "-"
    out_box.append(all_sq[:-1])
    for i in range(len(out_box)):
        out_box[i] = out_box[i].split("-")
    return out_box[::-1]
        

# nodes
def SSC_nodes(linkes):
    name_box = []
    for dic in linkes:
        if dic["source"] not in name_box:
            name_box.append(dic["source"])
        if dic["target"] not in name_box:
            name_box.append(dic["target"])
    nodes = []
    for i in name_box:
        mid_dic = {"name":i}
        nodes.append(mid_dic)
    return nodes


def cluster_link(source, target):
    sl, tl, vl = [], [], []
    for ti, taac in enumerate(target):
        taa_set = set(taac)
        aac_len = len(taac)
        for si, saac in enumerate(source):
            intersect = taa_set & set(saac)
            if intersect:
                sl.append(si)
                tl.append(ti)
                vl.append(len(intersect))
                aac_len -= len(intersect)
            if aac_len == 0:
                break
    return sl, tl, vl 

def type_link(clusters):
    base_idx = 0    
    source_idx, target_idx, values = [], [], []
    for i in range(len(clusters)-1):
        sl, tl, vl = cluster_link(clusters[i], clusters[i+1])
        sidx = [i+base_idx for i in sl]
        base_idx += len(clusters[i])
        tidx = [i+base_idx for i in tl]
        source_idx.extend(sidx)
        target_idx.extend(tidx)
        values.extend(vl)
    return source_idx, target_idx, values


# linkes
def SSC_linkes(labels, source_idx, target_idx, values):
    linkes = []
    for i in range(len(source_idx)):
        x_1 = source_idx[i]
        x_2 = target_idx[i]
        x_3 = values[i]
        mid_dic = {"source": labels[x_1], "target": labels[x_2], "value": x_3}
        if labels[x_1] != labels[x_2] and len(labels[x_1]) < len(labels[x_2]):
            linkes.append(mid_dic)
    return linkes


# 绘制桑葚图
def SSC_plot(nodes, linkes, title_ssc, out):
    c = (
        Sankey()
        .add(
            title_ssc,
            nodes,
            linkes,
            linestyle_opt=opts.LineStyleOpts(opacity=0.2, curve=0.5, color="source"),
            label_opts=opts.LabelOpts(position="right"),
        )
        .set_global_opts(title_opts=opts.TitleOpts(title="约化图谱"))
        .render(out)
    )
    print("约化图谱保存于 " + c)


# 桑葚图主程序
def SSC_main(file, type_r, now_path):
    raa_path = os.path.join(SSC_path, 'raacDB')
    raa_file = os.path.join(raa_path, file)
    raac_list = SSC_read(raa_file, type_r)
    # get linkes
    source_idx, target_idx, values = type_link(raac_list)
    labels = list(chain(*raac_list))
    linkes = SSC_linkes(labels, source_idx, target_idx, values)
    # get nodes
    nodes = SSC_nodes(linkes)
    # plot SSC
    title_ssc = "type" + type_r
    out = file + "_type" + type_r + "_SSC.html"
    out_path = os.path.join(now_path, out)
    SSC_plot(nodes, linkes, title_ssc, out_path)
