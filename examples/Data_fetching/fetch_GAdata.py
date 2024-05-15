import MDSplus
from mygadata import gadata
# import matplotlib.pyplot as plt
import h5py
import pickle
import numpy as np
import time
from tqdm import tqdm
import sys
import subprocess

# To run the code
# module purge & module load defaults
# python2.7 fetch_data.py

# *******start of user block************
output_path = '/cscratch/curiem/Data_fetch_Basic'
ece_pcece = False
size_GB = 400
directory_path = "/cscratch/curiem"  # to check the total file sizes
shot_list = np.arange(170000, 200000)

interval = 1000
# shot_list = shot_list[:10]
# *******end of user block************


def size_limiter_sleep(directory_path="/cscratch/curiem", size_GB=450):
    try:
        size = subprocess.check_output(
            ['du', '-sh', directory_path]).split()[0].decode('utf-8')
    except subprocess.CalledProcessError as e:
        print("Error fetching directory size: "+str(e))
        sys.exit(1)

    print("Size of" + directory_path + ": "+str(size))

    if size[-1] == "G" and float(size[:-1]) > size_GB:
        print("Size exceeds "+str(size_GB)+"GB. Sleeping for 1hr...")
        # Sleep for 1 hour
        time.sleep(3600)  # 3600 seconds = 1 hour
        print("1 hour has passed. Checking size again...")
        try:
            size = subprocess.check_output(
                ['du', '-sh', directory_path]).split()[0].decode('utf-8')
        except subprocess.CalledProcessError as e:
            print("Error fetching directory size: "+str(e))
            sys.exit(1)

        if size[-1] == "G" and float(size[:-1]) > size_GB:
            print("Size still exceeds "+str(size_GB)+"GB. Stopping")
            sys.exit(1)


def data2dict(shotn, signame, hf, atlconn):
    dict_group = hf.create_group(str(signame))
    try:
        data = gadata(signame, shotn, connection=atlconn)
        dict_group['xdata'] = data.xdata
        dict_group['ydata'] = data.ydata
        dict_group['zdata'] = data.zdata
        dict_group['xunits'] = data.xunits
        dict_group['yunits'] = data.yunits
        dict_group['zunits'] = data.zunits
    except: 
        print('%s not available, filled with NULL!' % (signame))
        dict_group['xdata'] = []
        dict_group['ydata'] = []
        dict_group['zdata'] = []
        dict_group['xunits'] = []
        dict_group['yunits'] = []
        dict_group['zunits'] = []
        del atlconn
        # global atlconn
        atlconn = MDSplus.Connection('atlas.gat.com')
        pass
    return atlconn


atlconn = MDSplus.Connection('atlas.gat.com')
ech_gytname = ['lei', 'luk', 'r2d']

# shot_list = np.loadtxt('DIIID_BES_Shot_List_Fatima.txt',
#                        delimiter='\n', dtype=np.int32)
# shot_list = np.load('tm-control-shots.npy')
# shot_list = np.unique(shot_list).astype(np.int)
# shot_list = [np.int32(sys.argv[1])]
# shot_list=[193266,193273,193280]
cannot_find = (['triangularity_u', 'triangularity_l', 'pech', 'neutronsrate'] +
               ['fplastic', 'fzns', 'fncrate01', 'fncrate02', 'fncrate03',
                'fncrate04', 'plasticfx1', 'plasticfx2', 'plasticfx3',
                'plasticfx4', 'neutronsrate1', 'neutronsrate2',
                'neutronsrate3', 'neutronsrate4'])
            
# basic is fundamental measured quantities (in contrast of fitted quantities)

signal_list = {'profiles': ['betap', 'betan', 'pres', 'wmhd', 'li',
                            'q0', 'q95', 'qmin', 'qpsi', 'rhoqmin', 'r0',
                            'aminor', 'kappa', 'tritop', 'tribot', 'alpha',
                            'psirz', 'ssibry', 'ssimag', 'rmaxis', 'zmaxis', 
                            'volume', 'drsep', 'gapbot', 'gapin', 'gapout',
                            'gaptop', 'zxpt1', 'zxpt2', 'edensfit', 'etempfit',
                            'trotfit', 'itempfit', 'idensfit', 'n1rms','n2rms',
                            'n3rms'],
               'basic': ['ip', 'ipsip', 'iptipp', 'pcbcoil', 'bcoil', 'bt',
                         'vloop'] + ['plasticfix', 'fzns'] +
                        ['fs00', 'fs01', 'fs02', 'fs03', 'fs04', 'fs05'],
               'actu': ['pinjf_%dl' % k for k in [15, 21, 30, 33]] +
                       ['pinjf_%dr' % k for k in [15,21,30,33]] +
                       ['tinj_%dl' % k for k in [15,21,30,33]] +
                       ['tinj_%dr' % k for k in [15,21,30,33]] +
                       ['echpwrc','echpwr'] +
                       ['ec%sfpwrc' % (x) for x in ech_gytname] +
                       ['ec%sxmfrac' % (x) for x in ech_gytname] +
                       ['ec%spolang' % (x) for x in ech_gytname] +
                       ['gasa', 'gasb', 'gasc', 'gasd', 'gase'] +
                       ['c19', 'c79', 'c139', 'c199', 'c259', 'c319', 'iu30', 
                        'iu90', 'iu150', 'iu210', 'iu270', 'iu330', 'il30', 
                        'il90', 'il150', 'il210', 'il270', 'il330'] +
                       ['ecoila', 'ecoilb', 'e567up', 'e567dn', 'e89dn',
                        'e89up'] +
                       ['f1a','f2a','f3a','f4a','f5a','f6a','f7a','f8a','f9a',
                        'f1b','f2b','f3b','f4b','f5b','f6b','f7b','f8b','f9b']}

for i in tqdm(range(len(shot_list))):
    shotn = shot_list[i]
    t1 = time.time()
    
    for grpname, signals in signal_list.items():
        hf = h5py.File(output_path + '/' + str(shotn) + '_'+grpname+'.h5', 'w')
        for signame in signals:
            atlconn = data2dict(shotn, signame, hf, atlconn)
        hf.close()
    
    if ece_pcece:
        hf = h5py.File(output_path + '/' + str(shotn)+'_ece.h5', 'w')
        pece_group = hf.create_group('pcece')    
        ece_group = hf.create_group('ece')
        rtece_group = hf.create_group('rtece')
                
        for k in range(40):
            print('chn %i' % (k+1))
            pece_data = gadata('pcece%d' % (k+1), shotn, connection=atlconn)
            pece_group['pcece%02d' % (k+1)] = pece_data.zdata
            ece_data = gadata('tecef%02d' % (k+1), shotn, connection=atlconn)
            ece_group['tecef%02d' % (k+1)] = ece_data.zdata
                        
            rtece_data = gadata('ecsdata%d' % (k+97), shotn, connection=atlconn)
            rtece_group['ecsdata%d' % (k+97)] = rtece_data.zdata
                        
        pece_group['xdata'] = pece_data.xdata
        pece_group['ydata'] = pece_data.ydata
        pece_group['xunits'] = pece_data.xunits
        pece_group['yunits'] = pece_data.yunits
        pece_group['pceceunits'] = pece_data.zunits
                
        ece_group['xdata'] = ece_data.xdata
        ece_group['ydata'] = ece_data.ydata
        ece_group['xunits'] = ece_data.xunits
        ece_group['yunits'] = ece_data.yunits
        ece_group['eceunits'] = ece_data.zunits

        rtece_group['xdata'] = rtece_data.xdata
        rtece_group['ydata'] = rtece_data.ydata
        rtece_group['xunits'] = rtece_data.xunits
        rtece_group['yunits'] = rtece_data.yunits
        rtece_group['rteceunits'] = rtece_data.zunits
        hf.close()
    if i % interval == 0:
        size_limiter_sleep(size_GB=size_GB)
        print('Shot #%d'%(shotn,))
        print(i)
#    print('time per shot:%ds' % (time.time()-t1))
