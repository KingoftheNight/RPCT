# import packages #############################################################

import tkinter as tk
import time
from tkinter import ttk
import os
from rpct import ReadBE, RocMP, TrainFP, Fuction_win
now_path = os.getcwd()

# create messagebox ###########################################################

def messagebox_help_auther():
    #new window
    help_auther_mg =  tk.Toplevel(window)
    help_auther_mg.title('Auther Informations')
    help_auther_mg.geometry('450x50')
    help_auther_mg.iconbitmap('.\\rpct\\bin\\logo.ico')
    #display box
    mha = tk.Text(help_auther_mg)
    mha.pack(fill="both")
    mha.insert('end','Auther:\tYuChao Liang\nEmail:\t1694822092@qq.com\nAddress:\tCollege of Life Science, Inner Mongolia University')

def messagebox_help_precaution():
    #new window
    help_precaution_mg =  tk.Toplevel(window)
    help_precaution_mg.title('Command Precaution')
    help_precaution_mg.geometry('1300x300')
    help_precaution_mg.iconbitmap('.\\rpct\\bin\\logo.ico')
    #display box
    precaution = Fuction_win.read_precaution()
    mhp = tk.Text(help_precaution_mg)
    mhp.pack(fill="both")
    mhp.insert('end',precaution)

def messagebox_help_Multprocess():
    #fuction
    def gui_exit():
        mmw.destroy()
        mmw.quit()
    def gui_mmb():
        mm_t = e_mm_t.get()
        mm_p = e_mm_p.get()
        if len(mm_t) != 0 and len(mm_p) != 0:
            print('\n>>>Multprocess...\n')
            Fuction_win.ray_blast(mm_t, mm_p, now_path)
            v_command = 'rblast -in ' + mm_t + ' -o ' + mm_p
            var.set(v_command)
            o_m = n_var.get()
            n_var.set(o_m + '\n' + v_command)
    def gui_mms():
        mm_e = e_mm_e.get()
        mm_g = e_mm_g.get()
        if len(mm_e) != 0 and len(mm_g) != 0:
            print('\n>>>Multprocess...\n')
            Fuction_win.ray_supplement(mm_e, mm_g, now_path)
            v_command = 'rsup -in ' + mm_e + ' -o ' + mm_g
            var.set(v_command)
            o_m = n_var.get()
            n_var.set(o_m + '\n' + v_command)
    
    #new window
    mmw =  tk.Toplevel(window) 
    mmw.title('Multprocess')
    mmw.geometry('560x80')
    mmw.iconbitmap('.\\rpct\\bin\\logo.ico')
    #multprocess
    mmf_1 = tk.Frame(mmw)
    mmf_2 = tk.Frame(mmw)
    mmf_1.pack(side='top',fill='x')
    mmf_2.pack(side='bottom',fill='x')
    mmf_2_1 = tk.Frame(mmf_2)
    mmf_2_1.pack(side='top',fill='x')
    mmf_2_2 = tk.Frame(mmf_2)
    mmf_2_2.pack(side='bottom',fill='x')
    ######folder
    tk.Label(mmf_1,text='rblast:  fasta folder',width=16,anchor='w').pack(side='left')
    e_mm_t = tk.Entry(mmf_1,show=None,width=15,font=('SimHei', 11))
    e_mm_t.pack(side='left')
    tk.Label(mmf_1,text='',width=2,anchor='w').pack(side='left')
    ######out
    tk.Label(mmf_1,text='out folder',width=11,anchor='w').pack(side='left')
    e_mm_p = tk.Entry(mmf_1,show=None,width=15,font=('SimHei', 11))
    e_mm_p.pack(side='left')
    tk.Label(mmf_1,text='',width=2,anchor='w').pack(side='left')
    ######button
    b_mm = tk.Button(mmf_1,text='run',font=('SimHei', 11),width=5,height=1,command=gui_mmb)
    b_mm.pack(side='right')
    ######folder
    tk.Label(mmf_2_1,text='rcheck: fasta folder',width=16,anchor='w').pack(side='left')
    e_mm_e = tk.Entry(mmf_2_1,show=None,width=15,font=('SimHei', 11))
    e_mm_e.pack(side='left')
    tk.Label(mmf_2_1,text='',width=2,anchor='w').pack(side='left')
    ######out
    tk.Label(mmf_2_1,text='out folder',width=11,anchor='w').pack(side='left')
    e_mm_g = tk.Entry(mmf_2_1,show=None,width=15,font=('SimHei', 11))
    e_mm_g.pack(side='left')
    tk.Label(mmf_2_1,text='',width=2,anchor='w').pack(side='left')
    ######button
    b_mm = tk.Button(mmf_2_1,text='run',font=('SimHei', 11),width=5,height=1,command=gui_mms)
    b_mm.pack(side='right')
    #exit
    b_mmw_back = tk.Button(mmf_2_2,text='Exit',font=('SimHei',11),bg='#75E4D7',relief='flat',
                       height=1,command=gui_exit)
    b_mmw_back.pack(side='bottom',fill='x')
    mmw.mainloop()

def messagebox_intlen():
    #fuction
    def gui_exit():
        miw.destroy()
        miw.quit()
    def gui_mil():
        mil_t = e_mil_t.get()
        mil_p = e_mil_p.get()
        mil_e = e_mil_e.get()
        mil_cg = e_mil_cg.get()
        mil_m = e_mil_m.get()
        if len(mil_t) != 0 and len(mil_p) != 0 and len(mil_e) != 0 and len(mil_cg) != 0 and len(mil_m) != 0:
            print('\n>>>Integrated Learning...\n')
            RocMP.model_combine_main(mil_t, mil_p, mil_e, mil_cg, mil_m, now_path)
            v_command = 'intlen -tf ' + mil_t + ' -pf ' + mil_p + ' -ef ' + mil_e + ' -cg ' + mil_cg + ' -m ' + mil_m
            var.set(v_command)
            o_m = n_var.get()
            n_var.set(o_m + '\n' + v_command)
    #new window
    miw =  tk.Toplevel(window) 
    miw.title('Integrated Learning')
    miw.geometry('560x80')
    miw.iconbitmap('.\\rpct\\bin\\logo.ico')
    #integrated learning
    milf_1 = tk.Frame(miw)
    milf_2 = tk.Frame(miw)
    milf_1.pack(side='top',fill='x')
    milf_2.pack(side='bottom',fill='x')
    milf_2_1 = tk.Frame(milf_2)
    milf_2_1.pack(side='top',fill='x')
    milf_2_2 = tk.Frame(milf_2)
    milf_2_2.pack(side='bottom',fill='x')
    ######train file
    tk.Label(milf_1,text='train features',width=11,anchor='w').pack(side='left')
    e_mil_t = tk.Entry(milf_1,show=None,width=10,font=('SimHei', 11))
    e_mil_t.pack(side='left')
    tk.Label(milf_1,text='',width=2,anchor='w').pack(side='left')
    ######predict file
    tk.Label(milf_1,text='predict features',width=13,anchor='w').pack(side='left')
    e_mil_p = tk.Entry(milf_1,show=None,width=10,font=('SimHei', 11))
    e_mil_p.pack(side='left')
    tk.Label(milf_1,text='',width=2,anchor='w').pack(side='left')
    ######eval file
    tk.Label(milf_1,text='evaluate file',width=10,anchor='w').pack(side='left')
    e_mil_e = tk.Entry(milf_1,show=None,width=10,font=('SimHei', 11))
    e_mil_e.pack(side='left')
    tk.Label(milf_1,text='',width=2,anchor='w').pack(side='left')
    ######cg file
    tk.Label(milf_2_1,text='cg file',width=5,anchor='w').pack(side='left')
    e_mil_cg = tk.Entry(milf_2_1,show=None,width=20,font=('SimHei', 11))
    e_mil_cg.pack(side='left')
    tk.Label(milf_2_1,text='',width=2,anchor='w').pack(side='left')
    ######member
    tk.Label(milf_2_1,text='member',width=7,anchor='w').pack(side='left')
    e_mil_m = tk.Entry(milf_2_1,show=None,width=4,font=('SimHei', 11))
    e_mil_m.pack(side='left')
    tk.Label(milf_2_1,text='',width=2,anchor='w').pack(side='left')
    ######button
    b_mil = tk.Button(milf_2_1,text='run',font=('SimHei', 11),width=5,height=1,command=gui_mil)
    b_mil.pack(side='right')
    #exit
    b_miw_back = tk.Button(milf_2_2,text='Exit',font=('SimHei',11),bg='#75E4D7',relief='flat',
                       height=1,command=gui_exit)
    b_miw_back.pack(side='bottom',fill='x')
    miw.mainloop()

def messagebox_pca():
    #fuction
    def gui_exit():
        mpw.destroy()
        mpw.quit()
    def gui_mpl():
        mpl_f = e_mpl_f.get()
        mpl_o = e_mpl_o.get()
        mpl_c = e_mpl_c.get()
        mpl_g = e_mpl_g.get()
        mpl_cv = e_mpl_cv.get()
        if len(mpl_f) != 0 and len(mpl_o) != 0 and len(mpl_c) != 0 and len(mpl_g) != 0 and len(mpl_cv) != 0:
            print('\n>>>Principal Component Analysis...\n')
            RocMP.PCA_main(mpl_f, mpl_o, mpl_c, mpl_g, mpl_cv, now_path)
            v_command = 'pca ' + mpl_f + ' -o ' + mpl_o + ' -c ' + mpl_c + ' -g ' + mpl_g + ' -cv ' + mpl_cv
            var.set(v_command)
            o_m = n_var.get()
            n_var.set(o_m + '\n' + v_command)
    #new window
    mpw =  tk.Toplevel(window)
    mpw.title('Principal Component Analysis')
    mpw.geometry('560x80')
    mpw.iconbitmap('.\\rpct\\bin\\logo.ico')
    #pca
    mplf_1 = tk.Frame(mpw)
    mplf_2 = tk.Frame(mpw)
    mplf_1.pack(side='top',fill='x')
    mplf_2.pack(side='bottom',fill='x')
    mplf_2_1 = tk.Frame(mplf_2)
    mplf_2_1.pack(side='top',fill='x')
    mplf_2_2 = tk.Frame(mplf_2)
    mplf_2_2.pack(side='bottom',fill='x')
    ######features file
    tk.Label(mplf_1,text='features file',width=11,anchor='w').pack(side='left')
    e_mpl_f = tk.Entry(mplf_1,show=None,width=20,font=('SimHei', 11))
    e_mpl_f.pack(side='left')
    tk.Label(mplf_1,text='',width=2,anchor='w').pack(side='left')
    ######out file
    tk.Label(mplf_1,text='out',width=4,anchor='w').pack(side='left')
    e_mpl_o = tk.Entry(mplf_1,show=None,width=20,font=('SimHei', 11))
    e_mpl_o.pack(side='left')
    tk.Label(mplf_1,text='',width=2,anchor='w').pack(side='left')
    ######c_number
    tk.Label(mplf_2_1,text='c_number',width=8,anchor='w').pack(side='left')
    e_mpl_c = tk.Entry(mplf_2_1,show=None,width=5,font=('SimHei', 11))
    e_mpl_c.pack(side='left')
    tk.Label(mplf_2_1,text='',width=2,anchor='w').pack(side='left')
    ######gamma
    tk.Label(mplf_2_1,text='g',width=2,anchor='w').pack(side='left')
    e_mpl_g = tk.Entry(mplf_2_1,show=None,width=5,font=('SimHei', 11))
    e_mpl_g.pack(side='left')
    tk.Label(mplf_2_1,text='',width=2,anchor='w').pack(side='left')
    ######crossV
    tk.Label(mplf_2_1,text='cv',width=3,anchor='w').pack(side='left')
    e_mpl_cv = tk.Entry(mplf_2_1,show=None,width=4,font=('SimHei', 11))
    e_mpl_cv.pack(side='left')
    tk.Label(mplf_2_1,text='',width=2,anchor='w').pack(side='left')
    ######button
    b_mpl = tk.Button(mplf_2_1,text='run',font=('SimHei', 11),width=5,height=1,command=gui_mpl)
    b_mpl.pack(side='right')
    #exit
    b_mpw_back = tk.Button(mplf_2_2,text='Exit',font=('SimHei',11),bg='#75E4D7',relief='flat',
                       height=1,command=gui_exit)
    b_mpw_back.pack(side='bottom',fill='x')
    mpw.mainloop()

def messagebox_sethys():
    #fuction
    def gui_exit():
        msw.destroy()
        msw.quit()
    def gui_msl():
        msl_f = e_msl_f.get()
        msl_o = e_msl_o.get()
        msl_c = e_msl_c.get()
        msl_g = e_msl_g.get()
        if len(msl_f) != 0 and len(msl_o) != 0 and len(msl_c) != 0 and len(msl_g) != 0:
            print('\n>>>Setting Hyperparameters File...\n')
            TrainFP.make_hys(msl_f, msl_c, msl_g, msl_o, now_path)
            v_command = 'makehys ' + msl_f + ' -o ' + msl_o + ' -c ' + msl_c + ' -g ' + msl_g
            var.set(v_command)
            o_m = n_var.get()
            n_var.set(o_m + '\n' + v_command)
    #new window
    msw =  tk.Toplevel(window) 
    msw.title('Set Hyperparameters File')
    msw.geometry('420x80')
    msw.iconbitmap('.\\rpct\\bin\\logo.ico')
    #pca
    mslf_1 = tk.Frame(msw)
    mslf_2 = tk.Frame(msw)
    mslf_1.pack(side='top',fill='x')
    mslf_2.pack(side='bottom',fill='x')
    mslf_2_1 = tk.Frame(mslf_2)
    mslf_2_1.pack(side='top',fill='x')
    mslf_2_2 = tk.Frame(mslf_2)
    mslf_2_2.pack(side='bottom',fill='x')
    ######features file
    tk.Label(mslf_1,text='folder',width=7,anchor='w').pack(side='left')
    e_msl_f = tk.Entry(mslf_1,show=None,width=15,font=('SimHei', 11))
    e_msl_f.pack(side='left')
    tk.Label(mslf_1,text='',width=2,anchor='w').pack(side='left')
    ######out file
    tk.Label(mslf_1,text='out',width=4,anchor='w').pack(side='left')
    e_msl_o = tk.Entry(mslf_1,show=None,width=15,font=('SimHei', 11))
    e_msl_o.pack(side='left')
    tk.Label(mslf_1,text='',width=2,anchor='w').pack(side='left')
    ######c_number
    tk.Label(mslf_2_1,text='c_number',width=8,anchor='w').pack(side='left')
    e_msl_c = tk.Entry(mslf_2_1,show=None,width=10,font=('SimHei', 11))
    e_msl_c.pack(side='left')
    tk.Label(mslf_2_1,text='',width=2,anchor='w').pack(side='left')
    ######gamma
    tk.Label(mslf_2_1,text='g',width=2,anchor='w').pack(side='left')
    e_msl_g = tk.Entry(mslf_2_1,show=None,width=10,font=('SimHei', 11))
    e_msl_g.pack(side='left')
    tk.Label(mslf_2_1,text='',width=2,anchor='w').pack(side='left')
    ######button
    b_msl = tk.Button(mslf_2_1,text='run',font=('SimHei', 11),width=5,height=1,command=gui_msl)
    b_msl.pack(side='right')
    #exit
    b_msw_back = tk.Button(mslf_2_2,text='Exit',font=('SimHei',11),bg='#75E4D7',relief='flat',
                       height=1,command=gui_exit)
    b_msw_back.pack(side='bottom',fill='x')
    msw.mainloop()

def messagebox_makedb():
    #fuction
    def gui_makedb():
        m_f = e_m_f.get()
        m_o = e_m_o.get()
        if len(m_f) != 0 and len(m_o) != 0:
            print('\n>>>Making database...\n')
            Fuction_win.read_madb(m_f, m_o)
            v_command = 'makedb ' + m_f + ' -o ' + m_o
            var.set(v_command)
            o_m = n_var.get()
            n_var.set(o_m + '\n' + v_command)
    def gui_exit():
        mmw.destroy()
        mmw.quit()
    #new window
    mmw =  tk.Toplevel(window) 
    mmw.title('Make Blast Database')
    mmw.geometry('380x60')
    mmw.iconbitmap('.\\rpct\\bin\\logo.ico')
    #make database
    mmlf_1 = tk.Frame(mmw)
    mmlf_2 = tk.Frame(mmw)
    mmlf_1.pack(side='top',fill='x')
    mmlf_2.pack(side='bottom',fill='x')
    ######file
    tk.Label(mmlf_1,text='file name',width=8,anchor='w').pack(side='left')
    e_m_f = tk.Entry(mmlf_1,show=None,width=10,font=('SimHei', 11))
    e_m_f.pack(side='left')
    tk.Label(mmlf_1,text='',width=2,anchor='w').pack(side='left')
    ######out
    tk.Label(mmlf_1,text='out name',width=8,anchor='w').pack(side='left')
    e_m_o = tk.Entry(mmlf_1,show=None,width=10,font=('SimHei', 11))
    e_m_o.pack(side='left')
    tk.Label(mmlf_1,text='',width=2,anchor='w').pack(side='left')
    ######button
    b_makedb = tk.Button(mmlf_1,text='run',font=('SimHei', 11),width=5,height=1,command=gui_makedb)
    b_makedb.pack(side='right')
    #exit
    b_mmw_back = tk.Button(mmlf_2,text='Exit',font=('SimHei',11),bg='#75E4D7',relief='flat',
                       height=1,command=gui_exit)
    b_mmw_back.pack(side='bottom',fill='x')
    mmw.mainloop()

def messagebox_raa():
    #fuction
    def gui_res():
        var.set('Reduce Amino Acid by private rules')
        res_p = e_res_p.get()
        if len(res_p) != 0:
            print('\n>>>Reducing Amino Acid...\n')
            RocMP.res_main(res_p)
            v_command = 'res ' + res_p
            var.set(v_command)
            o_m = n_var.get()
            n_var.set(o_m + '\n' + v_command)
    def gui_exit():
        mrw.destroy()
        mrw.quit()
    #new window
    mrw =  tk.Toplevel(window) 
    mrw.title('Reduce Amino Acids')
    mrw.geometry('640x60')
    mrw.iconbitmap('.\\rpct\\bin\\logo.ico')
    #res
    mrlf_1 = tk.Frame(mrw)
    mrlf_2 = tk.Frame(mrw)
    mrlf_1.pack(side='top',fill='x')
    mrlf_2.pack(side='bottom',fill='x')
    ######property
    tk.Label(mrlf_1,text='property',width=8,anchor='w').pack(side='left')
    e_res_p = tk.Entry(mrlf_1,show=None,width=60,font=('SimHei', 11))
    e_res_p.pack(side='left')
    tk.Label(mrlf_1,text='',width=2,anchor='w').pack(side='left')
    ######button
    b_predict = tk.Button(mrlf_1,text='run',font=('SimHei', 11),width=5,height=1,command=gui_res)
    b_predict.pack(side='right')
    #exit
    b_mrw_back = tk.Button(mrlf_2,text='Exit',font=('SimHei',11),bg='#75E4D7',relief='flat',
                       height=1,command=gui_exit)
    b_mrw_back.pack(side='bottom',fill='x')
    mrw.mainloop()

def messagebox_edit_raac():
    #fuction
    raa_path = '.\\rpct\\raacDB\\'
    raaBook = []
    for i in os.listdir(raa_path):
        raaBook.append(i)
    def gui_raac_read():
        value = merl_lb.get(merl_lb.curselection())
        with open(raa_path + value,'r',encoding='GB18030') as rf:
            data = rf.readlines()
            rf.close()
        view_list = ''
        for line in data:
            view_list += line
        raa_code.delete(1.0,'end')
        raa_code.insert('end',view_list)
        var.set(value)
    def gui_exit():
        out_file = raa_code.get('0.0','end')
        out_file = out_file[:-1]
        value = var.get()
        with open(raa_path + value,'w',encoding='GB18030') as rf:
            rf.write(out_file)
            rf.close()
        merw.destroy()
        merw.quit()
    #new window
    merw =  tk.Toplevel(window) 
    merw.title('Edit Reduce Amino Acids database')
    merw.geometry('440x400')
    merw.iconbitmap('.\\rpct\\bin\\logo.ico')
    #edit raac code
    merlf_1 = tk.Frame(merw)
    merlf_2 = tk.Frame(merw)
    merlf_1.pack(side='top',fill='x')
    merlf_2.pack(side='bottom',fill='x')
    merlf_2_1 = tk.Frame(merlf_2)
    merlf_2_1.pack(side='top',fill='x')
    merlf_2_2 = tk.Frame(merlf_2)
    merlf_2_2.pack(side='bottom',fill='x')
    #list
    merl_lb = tk.Listbox(merlf_1,width=38)
    for item in raaBook:
        merl_lb.insert('end',item)
    merl_lb.pack(side='left',fill='y')
    #select
    b_merw_select = tk.Button(merlf_1,text='Select RAAC Database',font=('SimHei',11),
                              height=3,command=gui_raac_read)
    b_merw_select.pack(side='right')
    #code
    raa_code = tk.Text(merlf_2_1,show=None,height=12,font=('SimHei', 11),width=20)
    raa_code.pack(fill='both')
    #exit
    b_merw_back = tk.Button(merlf_2_2,text='OK',font=('SimHei',11),bg='#75E4D7',relief='flat',
                       height=1,command=gui_exit)
    b_merw_back.pack(side='bottom',fill='x')
    merw.mainloop()

def messagebox_help_aaindex():
    #new window
    meaw =  tk.Toplevel(window) 
    meaw.title('Read AAindex Database')
    meaw.geometry('640x220')
    meaw.iconbitmap('.\\rpct\\bin\\logo.ico')
    #fuction
    with open('.\\rpct\\aaindexDB\\AAindex.txt','r',encoding='GB18030') as af:
        data = af.readlines()
        af.close()
    def gui_exit():
        meaw.destroy()
        meaw.quit()
    #read aaindex book
    mealf_1 = tk.Frame(meaw)
    mealf_2 = tk.Frame(meaw)
    mealf_1.pack(side='top',fill='x')
    mealf_2.pack(side='bottom',fill='x')
    #list
    aaindex = tk.Text(mealf_1,show=None,height=12,font=('SimHei', 11))
    aaindex.pack(fill='x')
    view_list = ''
    for line in data:
        view_list += line
    aaindex.insert('end',view_list)
    #exit
    b_meaw_back = tk.Button(mealf_2,text='OK',font=('SimHei',11),bg='#75E4D7',relief='flat',
                       height=2,command=gui_exit)
    b_meaw_back.pack(fill='x')
    meaw.mainloop()


# Function Class ##############################################################

#read

def gui_read():
    var.set('Read protein sequences and split it to single files')
    r_f = e_r_f.get()
    r_o = e_r_o.get()
    if len(r_f) != 0 and len(r_o) != 0:
        print('\n>>>Reading files...\n')
        ReadBE.read_fasta(r_f, r_o, now_path)
        v_command = 'read ' + r_f + ' -o ' + r_o
        var.set(v_command)
        o_m = n_var.get()
        n_var.set(o_m + '\n' + v_command)

#blast

def gui_blast():
    var.set('PSI-Blast protein sequence and get its PSSM matrix')
    b_f = e_b_f.get()
    b_o = e_b_o.get()
    b_db = e_b_db.get()
    b_n = e_b_n.get()
    b_ev = e_b_ev.get()
    if len(b_f) != 0 and len(b_o) != 0 and len(b_db) != 0 and len(b_n) != 0 and len(b_ev) != 0:
        print('\n>>>Blasting PSSM matrix...\n')
        Fuction_win.psi_blast(b_f, b_db, b_n, b_ev, b_o, now_path)
        v_command = 'blast ' + b_f + ' -db ' + b_db + ' -n ' + b_n + ' -ev ' + b_ev + ' -o ' + b_o
        var.set(v_command)
        o_m = n_var.get()
        n_var.set(o_m + '\n' + v_command)

#extact

def gui_extract():
    var.set('Extact features by PSSM-RAAC method')
    ex_f1 = e_ex_f1.get()
    ex_f2 = e_ex_f2.get()
    ex_r = e_ex_r.get()
    ex_o = e_ex_o.get()
    ex_l = e_ex_l.get()
    ex_s = e_ex_s.get()
    if len(ex_f1) != 0 and len(ex_f2) != 0 and len(ex_r) != 0 and len(ex_o) != 0 and len(ex_l) != 0 and len(ex_s) == 0:
        print('\n>>>Extracting PSSM matrix features...\n')
        ReadBE.extract_main(ex_f1, ex_f2, ex_o, ex_r, ex_l, now_path)
        v_command = 'extract ' + ex_f1 + ' ' + ex_f2 + ' -raa .\\rpct\\raacDB\\' + ex_r + ' -o ' + ex_o + ' -l ' + ex_l
        var.set(v_command)
        o_m = n_var.get()
        n_var.set(o_m + '\n' + v_command)
    if len(ex_f1) != 0 and len(ex_f2) != 0 and len(ex_r) == 0 and len(ex_o) != 0 and len(ex_l) != 0 and len(ex_s) != 0:
        print('\n>>>Extracting PSSM matrix features...\n')
        ReadBE.extract_main(ex_f1, ex_f2, ex_o, ex_s, ex_l, now_path)
        v_command = 'extract ' + ex_f1 + ' ' + ex_f2 + ' -selfraac ' + ex_r + ' -o ' + ex_o + ' -l ' + ex_l
        var.set(v_command)
        o_m = n_var.get()
        n_var.set(o_m + '\n' + v_command)

#search

def gui_search():
    var.set('Search LIBSVM Hyperparameters')
    s_d = e_s_d.get()
    s_f = e_s_f.get()
    if len(s_d) != 0 and len(s_f) != 0:
        var.set('You can only choose one between file and folder mode!')
    else:
        if len(s_d) == 0 and len(s_f) != 0:
            print('\n>>>Searching LIBSVM Hyperparameters...\n')
            TrainFP.search_f_main(s_f, now_path)
            v_command = 'search -f ' + s_f
            var.set(v_command)
            o_m = n_var.get()
            n_var.set(o_m + '\n' + v_command)
        else:
            if len(s_d) != 0 and len(s_f) == 0:
                print('\n>>>Searching LIBSVM Hyperparameters...\n')
                TrainFP.search_f_main(s_d, now_path)
                v_command = 'search -d ' + s_d
                var.set(v_command)
                o_m = n_var.get()
                n_var.set(o_m + '\n' + v_command)

#filter

def gui_filter():
    var.set('Filter features by IFS based on Relief method')
    fi_f = e_fi_f.get()
    fi_c = e_fi_c.get()
    fi_g = e_fi_g.get()
    fi_cv = e_fi_cv.get()
    #fi_n = e_fi_n.get()
    fi_o = e_fi_o.get()
    fi_r = e_fi_r.get()
    if len(fi_f) != 0 and len(fi_c) != 0 and len(fi_g) != 0 and len(fi_cv) != 0 and len(fi_o) != 0 and len(fi_r) != 0:
        print('\n>>>Filter Features...\n')
        TrainFP.filter_pro(fi_f, fi_o, fi_c, fi_g, fi_cv, fi_r, now_path)
        v_command = 'filter ' + fi_f + ' -c ' + fi_c + ' -g ' + fi_g + ' -cv ' + fi_cv + ' -o ' + fi_o + ' -r ' + fi_r
        var.set(v_command)
        o_m = n_var.get()
        n_var.set(o_m + '\n' + v_command)

#filter features file setting

def gui_fffs():
    var.set('Filter features file setting')
    fs_f = e_fs_f.get()
    fs_i = e_fs_i.get()
    fs_n = e_fs_n.get()
    fs_o = e_fs_o.get()
    if len(fs_f) != 0 and len(fs_i) != 0 and len(fs_n) != 0 and len(fs_o) != 0:
        print('\n>>>Filter Features File Setting...\n')
        ReadBE.read_filter(fs_f, fs_i, fs_o, fs_n, now_path)
        v_command = 'fffs ' + fs_f + ' -f ' + fs_i + ' -n ' + fs_n + ' -o ' + fs_o
        var.set(v_command)
        o_m = n_var.get()
        n_var.set(o_m + '\n' + v_command)

#train

def gui_train():
    var.set('Train features file to model by LIBSVM')
    t_d = e_t_d.get()
    t_f = e_t_f.get()
    t_c = e_t_c.get()
    t_g = e_t_g.get()
    t_o = e_t_o.get()
    t_cg = e_t_cg.get()
    if len(t_d) != 0 or len(t_f) != 0:
        if len(t_d) != 0 and len(t_c) != 0 and len(t_g) != 0 and len(t_o) != 0 and len(t_cg) == 0 and len(t_f) == 0:
            print('\n>>>Training Features File...\n')
            TrainFP.train_main(t_d, t_c, t_g, t_o)
            v_command = 'train -d ' + t_d + ' -c ' + t_c + ' -g ' + t_g + ' -o ' + t_o + '.model'
            var.set(v_command)
            o_m = n_var.get()
            n_var.set(o_m + '\n' + v_command)
        else:
            if len(t_f) != 0 and len(t_o) != 0 and len(t_cg) != 0 and len(t_d) == 0 and len(t_c) == 0 and len(t_g) == 0:
                print('\n>>>Training Features Files...\n')
                TrainFP.train_f_main(t_f, t_cg, t_o, now_path)
                v_command = 'train -f ' + t_f + ' -cg ' + t_cg + ' -o ' + t_o
                var.set(v_command)
                o_m = n_var.get()
                n_var.set(o_m + '\n' + v_command)
            else:
                var.set('You can only choose one between file and folder mode!')

#eval

def gui_eval():
    var.set('Evaluate features file by Cross-validation')
    e_d = e_e_d.get()
    e_f = e_e_f.get()
    e_c = e_e_c.get()
    e_g = e_e_g.get()
    e_cv = e_e_cv.get()
    e_o = e_e_o.get()
    e_cg = e_e_cg.get()
    if len(e_d) != 0 or len(e_f) != 0:
        if len(e_d) != 0 and len(e_c) != 0 and len(e_g) != 0 and len(e_cv) != 0 and len(e_o) != 0 and len(e_cg) == 0 and len(e_f) == 0:
            print('\n>>>Evaluating Features File...\n')
            TrainFP.eval_main(e_d, e_c, e_g, e_cv, e_o, now_path)
            v_command = 'eval -d ' + e_d + ' -c ' + e_c + ' -g ' + e_g + ' -cv ' + e_cv + ' -o ' + e_o
            var.set(v_command)
            o_m = n_var.get()
            n_var.set(o_m + '\n' + v_command)
        else:
            if len(e_f) != 0 and len(e_cv) != 0 and len(e_o) != 0 and len(e_cg) != 0 and len(e_d) == 0 and len(e_c) == 0 and len(e_g) == 0:
                print('\n>>>Evaluating Features Files...\n')
                TrainFP.eval_f_main(e_f, e_cg, e_cv, e_o, now_path)
                v_command = 'eval -f ' + e_f + ' -cg ' + e_cg + ' -cv ' + e_cv + ' -o ' + e_o
                var.set(v_command)
                o_m = n_var.get()
                n_var.set(o_m + '\n' + v_command)
            else:
                var.set('You can only choose one between file and folder mode!')

#ROC

def gui_roc():
    var.set('Draw ROC curve')
    roc_f = e_roc_f.get()
    #roc_n = e_roc_n.get()
    roc_o = e_roc_o.get()
    roc_c = e_roc_c.get()
    roc_g = e_roc_g.get()
    if len(roc_f) != 0 and len(roc_o) != 0 and len(roc_c) != 0 and len(roc_g) != 0:
        print('\n>>>Drawing ROC curve...\n')
        RocMP.roc_graph(roc_f, roc_o, roc_c, roc_g, now_path)
        v_command = 'roc ' + roc_f + ' -o ' + roc_o + ' -c ' + roc_c + ' -g ' + roc_g
        var.set(v_command)
        o_m = n_var.get()
        n_var.set(o_m + '\n' + v_command)

#predict

def gui_predict():
    var.set('Evaluate model by Predict file')
    p_f = e_p_f.get()
    p_m = e_p_m.get()
    p_o = e_p_o.get()
    if len(p_f) != 0 and len(p_o) != 0 and len(p_m) != 0:
        TrainFP.predict_main(p_f, p_m, p_o, now_path)
        v_command = 'predict ' + p_f + ' -m ' + p_m + ' -o ' + p_o
        var.set(v_command)
        o_m = n_var.get()
        n_var.set(o_m + '\n' + v_command)

#Save operation process

def gui_memory():
    o_m = n_var.get()
    with open('.\\rpct\\bin\\History.txt','a',encoding = 'UTF8') as f:
        f.write('\n' + o_m)
        f.close()
    var.set('This operation process has been saved in History.txt !')

# create window ###############################################################

window = tk.Tk()
window.title('RPCT_v3.0')
window.geometry('640x260')
window.configure(bg='Snow') 
window.iconbitmap('.\\rpct\\bin\\logo.ico')

# create frame levels #########################################################

#level one

frame = tk.Frame(window,bg='snow')
frame.pack(fill='both') 

#level two

frame_1 = tk.Frame(frame,bg='snow')
frame_2 = tk.Frame(frame,bg='snow')
frame_1.pack(side='top',fill='x')
frame_2.pack(side='bottom',fill='x')

#level three

frame_1_1 = tk.Frame(frame_1,bg='DarkTurquoise',height=3)#title
frame_2_1 = tk.Frame(frame_2,bg='snow')#progress
frame_1_2 = tk.Frame(frame_1,bg='snow')#tab
frame_2_2 = tk.Frame(frame_2,bg='snow')#history
frame_1_1.pack(side='top',fill='x')
frame_2_1.pack(side='top',fill='x')
frame_1_2.pack(side='bottom',fill='x')
frame_2_2.pack(side='bottom',fill='x')

# title line(frame_1_1) #######################################################

tilogo = tk.PhotoImage(file='.\\rpct\\bin\\Title.gif')
tilogo_label = tk.Label(frame_1_1,image=tilogo,bg='DarkTurquoise',anchor='center')
tilogo_label.pack(side='left')
title = tk.Label(frame_1_1,text='RPCT: PSSM-RAAC-based Protein Analysis Tool',
                 bg='DarkTurquoise',font=('SimHei', 16), width=50, height=3,anchor='center').pack()

# create tab contral(frame_1_2) ###########################################################

tabControl = ttk.Notebook(frame_1_2)

#read
tab1 = ttk.Frame(tabControl)
tabControl.add(tab1, text='    read    ')
#blast
tab2 = ttk.Frame(tabControl)
tabControl.add(tab2, text='    blast   ')
#extract
tab3 = ttk.Frame(tabControl)
tabControl.add(tab3, text='   extract  ')
#search
tab4 = ttk.Frame(tabControl)
tabControl.add(tab4, text='    search  ')
#filter
tab5 = ttk.Frame(tabControl)
tabControl.add(tab5, text='    filter  ')
#fffs
tab6 = ttk.Frame(tabControl)
tabControl.add(tab6, text='    fffs    ')
#eval
tab7 = ttk.Frame(tabControl)
tabControl.add(tab7, text='    train   ')
#eval
tab8 = ttk.Frame(tabControl)
tabControl.add(tab8, text='    eval    ')
#roc
tab9 = ttk.Frame(tabControl)
tabControl.add(tab9, text='     roc    ')
#predict
tab10 = ttk.Frame(tabControl)
tabControl.add(tab10, text='  predict  ')

tabControl.pack(expand=1, fill="x")

# create container tabs(frame_1_2) ############################################

#read
container_read = ttk.LabelFrame(tab1, text='Read Fasta Files')
container_read.pack(fill='x',padx=8,pady=4)
######file
tk.Label(container_read,text='file name',width=8,anchor='w').pack(side='left')
e_r_f = tk.Entry(container_read,show=None,width=20,font=('SimHei', 11))
e_r_f.pack(side='left')
tk.Label(container_read,text='',width=2,anchor='w').pack(side='left')
######out
tk.Label(container_read,text='out name',width=8,anchor='w').pack(side='left')
e_r_o = tk.Entry(container_read,show=None,width=20,font=('SimHei', 11))
e_r_o.pack(side='left')
tk.Label(container_read,text='',width=2,anchor='w').pack(side='left')
######button
b_read = tk.Button(container_read,text='run',font=('SimHei', 11),width=5,height=1,command=gui_read)
b_read.pack(side='right')

#blast
container_blast = ttk.LabelFrame(tab2, text='PSI-Blast')
container_blast.pack(fill='x',padx=8,pady=4)
######frame
cbf_1 = tk.Frame(container_blast)
cbf_2 = tk.Frame(container_blast)
cbf_1.pack(side='top',fill='x')
cbf_2.pack(side='bottom',fill='x')
######file
tk.Label(cbf_1,text='folder',width=5,anchor='w').pack(side='left')
e_b_f = tk.Entry(cbf_1,show=None,width=20,font=('SimHei', 11))
e_b_f.pack(side='left')
tk.Label(cbf_1,text='',width=2,anchor='w').pack(side='left')
######database
tk.Label(cbf_1,text='database',width=8,anchor='w').pack(side='left')
e_b_db = tk.Entry(cbf_1,show=None,width=20,font=('SimHei', 11))
e_b_db.pack(side='left')
tk.Label(cbf_1,text='',width=2,anchor='w').pack(side='left')
######clcye number
tk.Label(cbf_2,text='number',width=8,anchor='w').pack(side='left')
e_b_n = tk.Entry(cbf_2,show=None,width=5,font=('SimHei', 11))
e_b_n.pack(side='left')
tk.Label(cbf_2,text='',width=2,anchor='w').pack(side='left')
######evaluate value
tk.Label(cbf_2,text='evaluate',width=8,anchor='w').pack(side='left')
e_b_ev = tk.Entry(cbf_2,show=None,width=5,font=('SimHei', 11))
e_b_ev.pack(side='left')
tk.Label(cbf_2,text='',width=2,anchor='w').pack(side='left')
######out
tk.Label(cbf_2,text='out',width=4,anchor='w').pack(side='left')
e_b_o = tk.Entry(cbf_2,show=None,width=20,font=('SimHei', 11))
e_b_o.pack(side='left')
tk.Label(cbf_2,text='',width=2,anchor='w').pack(side='left')
######button
b_blast = tk.Button(cbf_2,text='run',font=('SimHei', 11),width=5,height=1,command=gui_blast)
b_blast.pack(side='right')

#extract
container_extract = ttk.LabelFrame(tab3, text='Extract Features')
container_extract.pack(fill='x',padx=8,pady=4)
######frame
cef_1 = tk.Frame(container_extract)
cef_2 = tk.Frame(container_extract)
cef_1.pack(side='top',fill='x')
cef_2.pack(side='bottom',fill='x')
######folder 1
tk.Label(cef_1,text='positive folder',width=12,anchor='w').pack(side='left')
e_ex_f1 = tk.Entry(cef_1,show=None,width=18,font=('SimHei', 11))
e_ex_f1.pack(side='left')
tk.Label(cef_1,text='',width=2,anchor='w').pack(side='left')
######folder 2
tk.Label(cef_1,text='negative folder',width=13,anchor='w').pack(side='left')
e_ex_f2 = tk.Entry(cef_1,show=None,width=18,font=('SimHei', 11))
e_ex_f2.pack(side='left')
tk.Label(cef_1,text='',width=2,anchor='w').pack(side='left')
######lmda
tk.Label(cef_1,text='lmda',width=5,anchor='w').pack(side='left')
e_ex_l = tk.Entry(cef_1,show=None,width=5,font=('SimHei', 11))
e_ex_l.pack(side='left')
tk.Label(cef_1,text='',width=2,anchor='w').pack(side='left')
######reduce file
tk.Label(cef_2,text='raaCODE',width=12,anchor='w').pack(side='left')
e_ex_r = tk.Entry(cef_2,show=None,width=10,font=('SimHei', 11))
e_ex_r.pack(side='left')
tk.Label(cef_2,text='',width=2,anchor='w').pack(side='left')
######out
tk.Label(cef_2,text='out',width=4,anchor='w').pack(side='left')
e_ex_o = tk.Entry(cef_2,show=None,width=10,font=('SimHei', 11))
e_ex_o.pack(side='left')
tk.Label(cef_2,text='',width=2,anchor='w').pack(side='left')
######reduce file
tk.Label(cef_2,text='selfraac',width=12,anchor='w').pack(side='left')
e_ex_s = tk.Entry(cef_2,show=None,width=10,font=('SimHei', 11))
e_ex_s.pack(side='left')
tk.Label(cef_2,text='',width=2,anchor='w').pack(side='left')
######button
b_extract = tk.Button(cef_2,text='run',font=('SimHei', 11),width=5,height=1,command=gui_extract)
b_extract.pack(side='right')

#search
container_search = ttk.LabelFrame(tab4, text='Search Hyperparameters')
container_search.pack(fill='x',padx=8,pady=4)
######document
tk.Label(container_search,text='document name',width=14,anchor='w').pack(side='left')
e_s_d = tk.Entry(container_search,show=None,width=20,font=('SimHei', 11))
e_s_d.pack(side='left')
tk.Label(container_search,text='',width=2,anchor='w').pack(side='left')
######folder
tk.Label(container_search,text='folder name',width=10,anchor='w').pack(side='left')
e_s_f = tk.Entry(container_search,show=None,width=20,font=('SimHei', 11))
e_s_f.pack(side='left')
tk.Label(container_search,text='',width=2,anchor='w').pack(side='left')
######button
b_search = tk.Button(container_search,text='run',font=('SimHei', 11),width=5,height=1,command=gui_search)
b_search.pack(side='right')

#filter
container_filter = ttk.LabelFrame(tab5, text='Filter Features')
container_filter.pack(fill='x',padx=8,pady=4)
######frame
cff_1 = tk.Frame(container_filter)
cff_2 = tk.Frame(container_filter)
cff_1.pack(side='top',fill='x')
cff_2.pack(side='bottom',fill='x')
######file
tk.Label(cff_1,text='file name',width=8,anchor='w').pack(side='left')
e_fi_f = tk.Entry(cff_1,show=None,width=20,font=('SimHei', 11))
e_fi_f.pack(side='left')
tk.Label(cff_1,text='',width=2,anchor='w').pack(side='left')
######c number
tk.Label(cff_1,text='c',width=1,anchor='w').pack(side='left')
e_fi_c = tk.Entry(cff_1,show=None,width=4,font=('SimHei', 11))
e_fi_c.pack(side='left')
tk.Label(cff_1,text='',width=2,anchor='w').pack(side='left')
######gamma
tk.Label(cff_1,text='g',width=1,anchor='w').pack(side='left')
e_fi_g = tk.Entry(cff_1,show=None,width=4,font=('SimHei', 11))
e_fi_g.pack(side='left')
tk.Label(cff_1,text='',width=2,anchor='w').pack(side='left')
######crossV
tk.Label(cff_1,text='cv',width=2,anchor='w').pack(side='left')
e_fi_cv = tk.Entry(cff_1,show=None,width=3,font=('SimHei', 11))
e_fi_cv.pack(side='left')
tk.Label(cff_1,text='',width=2,anchor='w').pack(side='left')
######out
tk.Label(cff_2,text='out',width=5,anchor='w').pack(side='left')
e_fi_o = tk.Entry(cff_2,show=None,width=20,font=('SimHei', 11))
e_fi_o.pack(side='left')
tk.Label(cff_2,text='',width=2,anchor='w').pack(side='left')
######cycle
tk.Label(cff_2,text='r',width=2,anchor='e').pack(side='left')
e_fi_r = tk.Entry(cff_2,show=None,width=3,font=('SimHei', 11))
e_fi_r.pack(side='left')
tk.Label(cff_2,text='',width=2,anchor='w').pack(side='left')
######button
b_filter = tk.Button(cff_2,text='run',font=('SimHei', 11),width=5,height=1,command=gui_filter)
b_filter.pack(side='right')

#filter features file setting
container_fffs = ttk.LabelFrame(tab6, text='Filter Features File Setting')
container_fffs.pack(fill='x',padx=8,pady=4)
######frame
cfsf_1 = tk.Frame(container_fffs)
cfsf_2 = tk.Frame(container_fffs)
cfsf_1.pack(side='top',fill='x')
cfsf_2.pack(side='bottom',fill='x')
######file
tk.Label(cfsf_1,text='file name',width=8,anchor='w').pack(side='left')
e_fs_f = tk.Entry(cfsf_1,show=None,width=20,font=('SimHei', 11))
e_fs_f.pack(side='left')
tk.Label(cfsf_1,text='',width=2,anchor='w').pack(side='left')
######IFS file
tk.Label(cfsf_1,text='IFS file name',width=11,anchor='w').pack(side='left')
e_fs_i = tk.Entry(cfsf_1,show=None,width=20,font=('SimHei', 11))
e_fs_i.pack(side='left')
tk.Label(cfsf_1,text='',width=2,anchor='w').pack(side='left')
######end feature
tk.Label(cfsf_2,text='end',width=5,anchor='w').pack(side='left')
e_fs_n = tk.Entry(cfsf_2,show=None,width=4,font=('SimHei', 11))
e_fs_n.pack(side='left')
tk.Label(cfsf_2,text='',width=3,anchor='w').pack(side='left')
######out
tk.Label(cfsf_2,text='out',width=4,anchor='w').pack(side='left')
e_fs_o = tk.Entry(cfsf_2,show=None,width=20,font=('SimHei', 11))
e_fs_o.pack(side='left')
tk.Label(cfsf_2,text='',width=4,anchor='w').pack(side='left')
######button
b_fffs = tk.Button(cfsf_2,text='run',font=('SimHei', 11),width=5,height=1,command=gui_fffs)
b_fffs.pack(side='right')

#train
container_train = ttk.LabelFrame(tab7, text='Train Models')
container_train.pack(fill='x',padx=8,pady=4)
######frame
ctf_1 = tk.Frame(container_train)
ctf_2 = tk.Frame(container_train)
ctf_1.pack(side='top',fill='x')
ctf_2.pack(side='bottom',fill='x')
######file
tk.Label(ctf_1,text='file',width=3,anchor='w').pack(side='left')
e_t_d = tk.Entry(ctf_1,show=None,width=20,font=('SimHei', 11))
e_t_d.pack(side='left')
tk.Label(ctf_1,text='',width=2,anchor='w').pack(side='left')
######folder
tk.Label(ctf_1,text='folder',width=5,anchor='w').pack(side='left')
e_t_f = tk.Entry(ctf_1,show=None,width=20,font=('SimHei', 11))
e_t_f.pack(side='left')
tk.Label(ctf_1,text='',width=2,anchor='w').pack(side='left')
######c number
tk.Label(ctf_1,text='c',width=1,anchor='w').pack(side='left')
e_t_c = tk.Entry(ctf_1,show=None,width=4,font=('SimHei', 11))
e_t_c.pack(side='left')
tk.Label(ctf_1,text='',width=2,anchor='w').pack(side='left')
######gamma
tk.Label(ctf_2,text='g',width=2,anchor='w').pack(side='left')
e_t_g = tk.Entry(ctf_2,show=None,width=4,font=('SimHei', 11))
e_t_g.pack(side='left')
tk.Label(ctf_2,text='',width=2,anchor='w').pack(side='left')
######out
tk.Label(ctf_2,text='out',width=4,anchor='w').pack(side='left')
e_t_o = tk.Entry(ctf_2,show=None,width=20,font=('SimHei', 11))
e_t_o.pack(side='left')
tk.Label(ctf_2,text='',width=2,anchor='w').pack(side='left')
######cg file
tk.Label(ctf_2,text='cg',width=3,anchor='w').pack(side='left')
e_t_cg = tk.Entry(ctf_2,show=None,width=20,font=('SimHei', 11))
e_t_cg.pack(side='left')
tk.Label(ctf_2,text='',width=2,anchor='w').pack(side='left')
######button
b_train = tk.Button(ctf_2,text='run',font=('SimHei', 11),width=5,height=1,command=gui_train)
b_train.pack(side='right')

#eval
container_eval = ttk.LabelFrame(tab8, text='Evaluate Models')
container_eval.pack(fill='x',padx=8,pady=4)
######frame
cevf_1 = tk.Frame(container_eval)
cevf_2 = tk.Frame(container_eval)
cevf_1.pack(side='top',fill='x')
cevf_2.pack(side='bottom',fill='x')
######file
tk.Label(cevf_1,text='file',width=3,anchor='w').pack(side='left')
e_e_d = tk.Entry(cevf_1,show=None,width=20,font=('SimHei', 11))
e_e_d.pack(side='left')
tk.Label(cevf_1,text='',width=2,anchor='w').pack(side='left')
######folder
tk.Label(cevf_1,text='folder',width=6,anchor='w').pack(side='left')
e_e_f = tk.Entry(cevf_1,show=None,width=20,font=('SimHei', 11))
e_e_f.pack(side='left')
tk.Label(cevf_1,text='',width=2,anchor='w').pack(side='left')
######c number
tk.Label(cevf_1,text='c',width=2,anchor='w').pack(side='left')
e_e_c = tk.Entry(cevf_1,show=None,width=4,font=('SimHei', 11))
e_e_c.pack(side='left')
tk.Label(cevf_1,text='',width=2,anchor='w').pack(side='left')
######gamma
tk.Label(cevf_2,text='g',width=1,anchor='w').pack(side='left')
e_e_g = tk.Entry(cevf_2,show=None,width=4,font=('SimHei', 11))
e_e_g.pack(side='left')
tk.Label(cevf_2,text='',width=2,anchor='w').pack(side='left')
######crossV
tk.Label(cevf_2,text='cv',width=2,anchor='w').pack(side='left')
e_e_cv = tk.Entry(cevf_2,show=None,width=3,font=('SimHei', 11))
e_e_cv.pack(side='left')
tk.Label(cevf_2,text='',width=2,anchor='w').pack(side='left')
######out
tk.Label(cevf_2,text='out',width=3,anchor='w').pack(side='left')
e_e_o = tk.Entry(cevf_2,show=None,width=18,font=('SimHei', 11))
e_e_o.pack(side='left')
tk.Label(cevf_2,text='',width=2,anchor='w').pack(side='left')
######cg file
tk.Label(cevf_2,text='cg',width=2,anchor='w').pack(side='left')
e_e_cg = tk.Entry(cevf_2,show=None,width=18,font=('SimHei', 11))
e_e_cg.pack(side='left')
tk.Label(cevf_2,text='',width=2,anchor='w').pack(side='left')
######button
b_eval = tk.Button(cevf_2,text='run',font=('SimHei', 11),width=5,height=1,command=gui_eval)
b_eval.pack(side='right')

#roc
container_roc = ttk.LabelFrame(tab9, text='ROC Graph')
container_roc.pack(fill='x',padx=8,pady=4)
######frame
crof_1 = tk.Frame(container_roc)
crof_2 = tk.Frame(container_roc)
crof_1.pack(side='top',fill='x')
crof_2.pack(side='bottom',fill='x')
######file name
tk.Label(crof_1,text='file name',width=8,anchor='w').pack(side='left')
e_roc_f = tk.Entry(crof_1,show=None,width=20,font=('SimHei', 11))
e_roc_f.pack(side='left')
tk.Label(crof_1,text='',width=2,anchor='w').pack(side='left')
######feature number
tk.Label(crof_1,text='feature number',width=14,anchor='w').pack(side='left')
e_roc_n = tk.Entry(crof_1,show=None,width=4,font=('SimHei', 11))
e_roc_n.pack(side='left')
tk.Label(crof_1,text='',width=2,anchor='w').pack(side='left')
######out
tk.Label(crof_2,text='out',width=3,anchor='w').pack(side='left')
e_roc_o = tk.Entry(crof_2,show=None,width=14,font=('SimHei', 11))
e_roc_o.pack(side='left')
tk.Label(crof_2,text='',width=2,anchor='w').pack(side='left')
######c number
tk.Label(crof_2,text='c',width=2,anchor='w').pack(side='left')
e_roc_c = tk.Entry(crof_2,show=None,width=4,font=('SimHei', 11))
e_roc_c.pack(side='left')
tk.Label(crof_2,text='',width=2,anchor='w').pack(side='left')
######gamma
tk.Label(crof_2,text='g',width=2,anchor='w').pack(side='left')
e_roc_g = tk.Entry(crof_2,show=None,width=4,font=('SimHei', 11))
e_roc_g.pack(side='left')
tk.Label(crof_2,text='',width=2,anchor='w').pack(side='left')
######button
b_roc = tk.Button(crof_2,text='run',font=('SimHei', 11),width=5,height=1,command=gui_roc)
b_roc.pack(side='right')

#predict
container_predict = ttk.LabelFrame(tab10, text='Predict Models')
container_predict.pack(fill='x',padx=8,pady=4)
######file
tk.Label(container_predict,text='file name',width=8,anchor='w').pack(side='left')
e_p_f = tk.Entry(container_predict,show=None,width=15,font=('SimHei', 11))
e_p_f.pack(side='left')
tk.Label(container_predict,text='',width=2,anchor='w').pack(side='left')
######model
tk.Label(container_predict,text='model name',width=11,anchor='w').pack(side='left')
e_p_m = tk.Entry(container_predict,show=None,width=15,font=('SimHei', 11))
e_p_m.pack(side='left')
tk.Label(container_predict,text='',width=2,anchor='w').pack(side='left')
######out
tk.Label(container_predict,text='out',width=3,anchor='w').pack(side='left')
e_p_o = tk.Entry(container_predict,show=None,width=10,font=('SimHei', 11))
e_p_o.pack(side='left')
tk.Label(container_predict,text='',width=2,anchor='w').pack(side='left')
######button
b_predict = tk.Button(container_predict,text='run',font=('SimHei', 11),width=5,height=1,command=gui_predict)
b_predict.pack(side='right')

# Current Process line(frame_2_1) ########################################################

cpl = tk.Label(frame_2_1,text='Current Process: ',width=20,anchor='e',bg='LightBlue',font=('SimHei', 11))
cpl.pack(side='left',fill='y')
var = tk.StringVar()
comment = tk.Label(frame_2_1,textvariable=var,width=67,anchor='w',bg='LightBlue')
comment.pack(side='left')

# Historical process line(frame_2_2) #####################################################

tk.Label(frame_2_2,text='Creation time: ',width=20,anchor='e',bg='Snow',
         font=('SimHei', 11)).pack(side='left',padx=2)
n_var = tk.StringVar()
comment = tk.Label(frame_2_2,textvariable=n_var,width=60,height=1,anchor='nw',bg='Snow',wraplength=426)
comment.pack(side='left')

# Get current time ############################################################

new_memory = '═══════════ ' + time.asctime(time.localtime(time.time())) + ' ════════════'
n_var.set(new_memory)
var.set('Current working path>>> ' + now_path)

# create menu ##################################################################

#root menu
root_menu = tk.Menu(window)
#file edit menu
filemenu = tk.Menu(root_menu,tearoff=0)
root_menu.add_cascade(label='File', menu=filemenu)
filemenu.add_command(label='Edit RAAC database',command=messagebox_edit_raac)
filemenu.add_command(label='Make Blast Database',command=messagebox_makedb)
filemenu.add_command(label='Save Operation Process',command=gui_memory)
filemenu.add_command(label='Set Hyperparameters file',command=messagebox_sethys)
filemenu.add_command(label='Exit',command=window.quit)
#tools menu
toolmenu = tk.Menu(root_menu,tearoff=0)
root_menu.add_cascade(label='Tools', menu=toolmenu)
toolmenu.add_command(label='Integrated Learning',command=messagebox_intlen)
toolmenu.add_command(label='Multprocess',command=messagebox_help_Multprocess)
toolmenu.add_command(label='Principal Component Analysis',command=messagebox_pca)
toolmenu.add_command(label='Self Reduce Amino Acids Code',command=messagebox_raa)
#help menu
editmenu = tk.Menu(root_menu, tearoff=0)
root_menu.add_cascade(label='Help',menu=editmenu)
editmenu.add_command(label='AAindex',command=messagebox_help_aaindex)
editmenu.add_command(label='Author Information',command=messagebox_help_auther)
editmenu.add_command(label='Precautions',command=messagebox_help_precaution)
#view
window.config(menu=root_menu)

# Real-time refresh ###########################################################

window.mainloop()