# 2021.3.16
# RPCT for python

# import packages #############################################################


from rpct import ReadBE, RocMP, TrainFP, Multprocess, SSC
import argparse
import os
now_path = os.getcwd()


# fuctions ####################################################################


# make database
def parse_makedb(args):
    print('\n>>>Making database...\n')
    ReadBE.read_madb(args.file[0], args.output[0])


# read
def parse_read(args):
    print('\n>>>Reading files...\n')
    if len(args.file) == len(args.output):
        for eachdir in range(len(args.file)):
            ReadBE.read_fasta(args.file[eachdir], args.output[eachdir], now_path)
    else:
        print('\n>>>ERROR: The number of input files and output folders is not equal.\n')


# psi-blast
def parse_blast(args):
    print('\n>>>Blasting PSSM matrix...\n')
    if len(args.folder) == len(args.output):
        for eachdir in range(len(args.folder)):
            ReadBE.psi_blast(args.folder[eachdir], args.database[0], args.num_iterations[0], args.expected_value[0],
                             args.output[eachdir], now_path)
    else:
        print('\n>>>ERROR: The number of input folders and output folders is not equal.\n')


# extract features
def parse_extract(args):
    print('\n>>>Extracting PSSM matrix features...\n')
    if args.reduce_aa:
        ReadBE.extract_main(args.folder[0], args.folder[1], args.output[0], args.reduce_aa[0], args.lmda[0], now_path)
    else:
        ReadBE.extract_d_main(args.folder[0], args.folder[1], args.output[0], args.self_raac[0], args.lmda[0], now_path)

# search best factors
def parse_grid(args):
    if args.document:
        TrainFP.search_main(args.document[0])
    else:
        TrainFP.search_f_main(args.folder[0], now_path)


# train model
def parse_train(args):
    if args.document:
        TrainFP.train_main(args.document[0], args.c_number[0], args.gamma[0], args.output[0])
    else:
        TrainFP.train_f_main(args.folder[0], args.cg_box[0], args.output[0], now_path)


# evaluate model
def parse_evaluate(args):
    if args.document:
        TrainFP.eval_main(args.document[0], args.c_number[0], args.gamma[0], args.cross_validation[0], args.output[0],
                          now_path)
    else:
        TrainFP.eval_f_main(args.folder[0], args.cg_box[0], args.cross_validation[0], args.output[0], now_path)


# ROC cruve
def parse_roc(args):
    RocMP.roc_graph(args.file[0], args.output[0], args.c_number[0], args.gamma[0], now_path)


# filter features
def parse_filter(args):
    TrainFP.filter_pro(args.file[0], args.output[0], args.c_number[0], args.gamma[0], args.cross_validation[0],
                       args.round[0], now_path)


# predict files
def parse_predict(args):
    TrainFP.predict_main(args.file[0], args.model[0], args.output[0], now_path)


# filter features files
def parse_fffs(args):
    ReadBE.read_filter(args.file[0], args.filter_index[0], args.output[0], args.end_feature[0],
                       now_path)


# reduce aa by personal rules
def parse_res(args):
    RocMP.res_main(args.rule_id)


# integrated learning
def parse_intlen(args):
    RocMP.model_combine_main(args.train_features[0], args.predict_features[0], args.eval_file[0], args.cg_file[0],
                             args.member[0], now_path)


# principal component analysis
def parse_pca(args):
    RocMP.PCA_main(args.file[0], args.output[0], args.c_number[0], args.gamma[0], args.cross_validation[0], now_path)


# make hys file
def parse_mhy(args):
    TrainFP.make_hys(args.folder[0], args.c_number[0], args.gamma[0], args.out[0], now_path)


# ray blast
def parse_rblast(args):
    Multprocess.ray_blast(args.folder[0], args.out[0], now_path)


# ray supplement
def parse_rsup(args):
    Multprocess.ray_supplement(args.folder[0], args.out[0], now_path)

# view raac map
def parse_view(args):
    SSC.SSC_main(args.file[0], args.type_raac[0], now_path)


# argparse ####################################################################


def rpct_main():
    parser = argparse.ArgumentParser(description='An Protein Classification Tool Base On RAAC-PSSM Matrix',
                                     fromfile_prefix_chars='@', conflict_handler='resolve')
    subparsers = parser.add_subparsers(help='RPCT help')
    # make database
    parser_ma = subparsers.add_parser('makedb', add_help=False, help='make database')
    parser_ma.add_argument('file', nargs=1, help='fasta database name')
    parser_ma.add_argument('-o', '--output', nargs=1, help='output file name')
    parser_ma.set_defaults(func=parse_makedb)
    # read and segment original files
    parser_re = subparsers.add_parser('read', add_help=False, help='read protein sequences files and segment it')
    parser_re.add_argument('file', nargs='+', help='fasta file paths')
    parser_re.add_argument('-o', '--output', nargs='+', help='output folder')
    parser_re.set_defaults(func=parse_read)
    # blast PSSM matrix
    parser_bl = subparsers.add_parser('blast', add_help=False, help='get PSSM matrix by psi-blast')
    parser_bl.add_argument('folder', nargs='+', help='input a folder containing single sequence files')
    parser_bl.add_argument('-db', '--database', nargs=1, type=str, help='database for blast')
    parser_bl.add_argument('-n', '--num_iterations', nargs=1, type=str, help='number of blast cycles')
    parser_bl.add_argument('-ev', '--expected_value', nargs=1, type=str, help='expected value of blast cycles')
    parser_bl.add_argument('-o', '--output', nargs='+', help='output folder')
    parser_bl.set_defaults(func=parse_blast)
    # extract features
    parser_ex = subparsers.add_parser('extract', add_help=False, help='extract the features of PSSM matrix')
    parser_ex.add_argument('folder', nargs=2, help='input PSSM matrix files folder')
    parser_ex.add_argument('-raa', '--reduce_aa', nargs=1, type=str, help='reduce amino acid file')
    parser_ex.add_argument('-o', '--output', nargs=1, type=str, help='output folder')
    parser_ex.add_argument('-l', '--lmda', nargs=1, type=str, help='sliding window lmda')
    parser_ex.add_argument('-r', '--self_raac', nargs=1, type=str, help='self raac')
    parser_ex.set_defaults(func=parse_extract)
    # grid search
    parser_se = subparsers.add_parser('search', add_help=False, help='search c_number and gamma for training')
    parser_se.add_argument('-d', '--document', nargs=1, help='feature file name')
    parser_se.add_argument('-f', '--folder', nargs=1, help='feature files folder')
    parser_se.set_defaults(func=parse_grid)
    # train
    parser_tr = subparsers.add_parser('train', add_help=False, help='train model by feature file')
    parser_tr.add_argument('-d', '--document', nargs=1, help='feature file name')
    parser_tr.add_argument('-f', '--folder', nargs=1, help='feature files folder')
    parser_tr.add_argument('-c', '--c_number', nargs=1, help='c_number')
    parser_tr.add_argument('-g', '--gamma', nargs=1, help='gamma')
    parser_tr.add_argument('-o', '--output', nargs=1, help='output file')
    parser_tr.add_argument('-cg', '--cg_box', nargs=1, help='c_number and gamma file name')
    parser_tr.set_defaults(func=parse_train)
    # evaluate
    parser_ev = subparsers.add_parser('eval', add_help=False, help='evaluate models by cross-validation')
    parser_ev.add_argument('-d', '--document', nargs=1, help='feature file name')
    parser_ev.add_argument('-f', '--folder', nargs=1, help='feature files folder')
    parser_ev.add_argument('-c', '--c_number', nargs=1, help='c_number')
    parser_ev.add_argument('-g', '--gamma', nargs=1, help='gamma')
    parser_ev.add_argument('-cg', '--cg_box', nargs=1, help='c_number and gamma file name')
    parser_ev.add_argument('-cv', '--cross_validation', nargs=1, help='5: 5-fold cross-validation   -1: jackknife')
    parser_ev.add_argument('-o', '--output', nargs=1, help='output folder')
    parser_ev.set_defaults(func=parse_evaluate)
    # ROC
    parser_ro = subparsers.add_parser('roc', add_help=False, help='draw ROC cruve graph')
    parser_ro.add_argument('file', nargs=1, help='input features file')
    parser_ro.add_argument('-o', '--output', nargs=1, help='output ROC graph')
    parser_ro.add_argument('-c', '--c_number', nargs=1, help='c_number')
    parser_ro.add_argument('-g', '--gamma', nargs=1, help='gamma')
    parser_ro.set_defaults(func=parse_roc)
    # filter
    parser_fi = subparsers.add_parser('filter', add_help=False, help='filter features by IFS')
    parser_fi.add_argument('file', nargs=1, help='input features file')
    parser_fi.add_argument('-c', '--c_number', nargs=1, help='c_number')
    parser_fi.add_argument('-g', '--gamma', nargs=1, help='gamma')
    parser_fi.add_argument('-cv', '--cross_validation', nargs=1, help='5: 5-fold cross-validation   -1: jackknife')
    parser_fi.add_argument('-o', '--output', nargs=1, help='output file name')
    parser_fi.add_argument('-r', '--round', nargs=1, help='the number of test cycle')
    parser_fi.set_defaults(func=parse_filter)
    # predict
    parser_pr = subparsers.add_parser('predict', add_help=False, help='evaluate models by predict files')
    parser_pr.add_argument('file', nargs=1, help='input features file')
    parser_pr.add_argument('-m', '--model', nargs=1, help='input model name')
    parser_pr.add_argument('-o', '--output', nargs=1, help='output file name')
    parser_pr.set_defaults(func=parse_predict)
    # new filter features file
    parser_ff = subparsers.add_parser('fffs', add_help=False, help='set filter features files')
    parser_ff.add_argument('file', nargs=1, help='input features file')
    parser_ff.add_argument('-f', '--filter_index', nargs=1, help='IFS order for each feature')
    parser_ff.add_argument('-n', '--end_feature', nargs=1, help='final feature number')
    parser_ff.add_argument('-o', '--output', nargs=1, help='output file name')
    parser_ff.set_defaults(func=parse_fffs)
    # reduce aa by personal rules
    parser_rs = subparsers.add_parser('res', add_help=False, help='reduce aa by personal rules')
    parser_rs.add_argument('rule_id', nargs='+', help='input aa property ID')
    parser_rs.set_defaults(func=parse_res)
    # integrated learning
    parser_in = subparsers.add_parser('intlen', add_help=False, help='integrated learning with majority vote')
    parser_in.add_argument('-tf', '--train_features', nargs=1, help='input train features file')
    parser_in.add_argument('-pf', '--predict_features', nargs=1, help='input predict features file')
    parser_in.add_argument('-ef', '--eval_file', nargs=1, help='input evaluate file')
    parser_in.add_argument('-cg', '--cg_file', nargs=1, help='input hyperparameter file')
    parser_in.add_argument('-m', '--member', nargs=1, help='choose train models number')
    parser_in.set_defaults(func=parse_intlen)
    # PCA
    parser_pc = subparsers.add_parser('pca', add_help=False, help='principal component analysis')
    parser_pc.add_argument('file', nargs=1, help='input features file')
    parser_pc.add_argument('-c', '--c_number', nargs=1, help='c_number')
    parser_pc.add_argument('-g', '--gamma', nargs=1, help='gamma')
    parser_pc.add_argument('-cv', '--cross_validation', nargs=1, help='5: 5-fold cross-validation   -1: jackknife')
    parser_pc.add_argument('-o', '--output', nargs=1, help='output file name')
    parser_pc.set_defaults(func=parse_pca)
    # mhys
    parser_mh = subparsers.add_parser('mhys', add_help=False, help='make hyperparameter file')
    parser_mh.add_argument('folder', nargs=1, help='input features folder')
    parser_mh.add_argument('-c', '--c_number', nargs=1, help='c_number')
    parser_mh.add_argument('-g', '--gamma', nargs=1, help='gamma')
    parser_mh.add_argument('-o', '--out', nargs=1, help='output file name')
    parser_mh.set_defaults(func=parse_mhy)
    # ray_blast
    parser_rb = subparsers.add_parser('rblast', add_help=False, help='multithreaded blast')
    parser_rb.add_argument('folder', nargs=1, help='input sequence folder')
    parser_rb.add_argument('-o', '--out', nargs=1, help='out folder')
    parser_rb.set_defaults(func=parse_rblast)
    # ray_supplement
    parser_rs = subparsers.add_parser('rsup', add_help=False, help='supplement blast')
    parser_rs.add_argument('folder', nargs=1, help='input sequence folder')
    parser_rs.add_argument('-o', '--out', nargs=1, help='out folder')
    parser_rs.set_defaults(func=parse_rsup)
    # view
    parser_vw = subparsers.add_parser('view', add_help=False, help='view raac map')
    parser_vw.add_argument('file', nargs=1, help='input raac book name')
    parser_vw.add_argument('-t', '--type_raac', nargs=1, help='type of raac')
    parser_vw.set_defaults(func=parse_view)

    args = parser.parse_args()
    try:
        args.func(args)
    except AttributeError:
        pass


# main ########################################################################


if __name__ == '__main__':
    rpct_main()
