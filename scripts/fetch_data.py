import MDSplus
from mygadata import gadata
#import matplotlib.pyplot as plt
import h5py
import pickle
import numpy as np
import time
# from tqdm import tqdm
import sys


def data2dict(shotn, signame, hf, atlconn) :
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
		#global atlconn
                atlconn = MDSplus.Connection('atlas.gat.com')
		pass
	return atlconn

atlconn = MDSplus.Connection('atlas.gat.com')
ech_gytname = ['lei','luk','r2d']

shot_list = np.loadtxt('rtcakenn-shots.txt',delimiter='\n',dtype=np.int32)
# shot_list = np.load('tm-control-shots.npy');shot_list=np.unique(shot_list).astype(np.int)
# shot_list = [np.int32(sys.argv[1])]
# shot_list=[200385]
# shot_list=np.arange(136219,136225)

signal_list= {
'profiles':['aminor','alpha','bcoil', 'betan', 'bmspinj', 'bmstinj', 'bt', 'dssdenest', 'edensfit', 'etempfit', 'fzns', 'ip', 'ipsip', 'iptipp', 'neutronsrate', 'pcbcoil', 'pinj', 'plasticfix', 'pnbi', 'pres', 'q', 'q95', 'tinj','n1rms','n2rms','r0','kappa','tritop','tribot','gapin','fs00','fs01','fs02','fs03','fs04','fs05','fs06','fs07','psirz']
,'pinj':['pinjf_%dl' % k for k in [15,21,30,33]]+['pinjf_%dr' % k for k in [15,21,30,33]]
,'tinj':['tinj_%dl' % k for k in [15,21,30,33]]+['tinj_%dr' % k for k in [15,21,30,33]]
,'ech':['pech','echpwrc','echpwr']+['ec%sfpwrc' % (x) for x in ech_gytname]+['ec%sxmfrac' % (x) for x in ech_gytname]+['ec%spolang' % (x) for x in ech_gytname]
,'shape':['kappa','triangularity_u','triangularity_l']
,'qmin':['qmin']
}

profl = True
ece_pcece=True
ts = True
co2= True
cer= True
mse= True
magnetics = True

for shotn in shot_list:

        t1=time.time()
        print('Shot #%d'%(shotn,))
        
        if profl:
            for grpname,signals in signal_list.items():
                    hf = h5py.File('/cscratch/jalalvanda/outputs/'+ str(shotn)+'_'+grpname+'.h5','w')
                    for signame in signals:
                            atlconn=data2dict(shotn,signame,hf,atlconn)
                    hf.close()


        if ece_pcece:
                hf = h5py.File('/cscratch/jalalvanda/outputs/'+ str(shotn)+'_ece.h5','w')
                pece_group = hf.create_group('pcece')	
#		ece_group = hf.create_group('ece')
#		rtece_group = hf.create_group('rtece')

                for k in range(40):
                        print('chn %i' % (k+1))
                        pece_data = gadata('pcece%d' % (k+1), shotn, connection=atlconn)
                        pece_group['pcece%02d' % (k+1)] = pece_data.zdata
#			ece_data = gadata('tecef%02d' % (k+1), shotn, connection=atlconn)
#			ece_group['tecef%02d' % (k+1)] = ece_data.zdata

#			rtece_data = gadata('ecsdata%d' % (k+97), shotn, connection=atlconn)
#			rtece_group['ecsdata%d' % (k+97)] = rtece_data.zdata

                pece_group['xdata'] = pece_data.xdata
                pece_group['ydata'] = pece_data.ydata
                pece_group['xunits'] = pece_data.xunits
                pece_group['yunits'] = pece_data.yunits
                pece_group['pceceunits'] = pece_data.zunits

#		ece_group['xdata'] = ece_data.xdata
#		ece_group['ydata'] = ece_data.ydata
#		ece_group['xunits'] = ece_data.xunits
#		ece_group['yunits'] = ece_data.yunits
#		ece_group['eceunits'] = ece_data.zunits

#		rtece_group['xdata'] = rtece_data.xdata
#		rtece_group['ydata'] = rtece_data.ydata
#		rtece_group['xunits'] = rtece_data.xunits
#		rtece_group['yunits'] = rtece_data.yunits
#		rtece_group['rteceunits'] = rtece_data.zunits
                hf.close()

        if ts:
                hf = h5py.File('/cscratch/jalalvanda/outputs/'+ str(shotn)+'_ts.h5','w')
                for corediv in ['core','div']:
                        for nete in ['ne','te']:
                                print('ts%s_%s' % (nete,corediv))
                                try: 
                                    ts_data = gadata('ts%s_%s' % (nete,corediv), shotn, connection=atlconn)

                                    ts_group = hf.create_group('ts%s_%s' % (nete,corediv))

                                    ts_group['ts%s_%s_xdata' % (nete,corediv)] = ts_data.xdata
                                    ts_group['ts%s_%s_ydata' % (nete,corediv)] = ts_data.ydata
                                    ts_group['ts%s_%s_zdata' % (nete,corediv)] = ts_data.zdata

                                    ts_group['ts%s_%s_xunits' % (nete,corediv)] = ts_data.xunits
                                    ts_group['ts%s_%s_yunits' % (nete,corediv)] = ts_data.yunits
                                    ts_group['ts%s_%s_zunits' % (nete,corediv)] = ts_data.zunits

                                    ts_data = gadata('ts%s_e_%s' % (nete,corediv), shotn, connection=atlconn)

                                    ts_group['ts%s_e_%s_xdata' % (nete,corediv)] = ts_data.xdata
                                    ts_group['ts%s_e_%s_ydata' % (nete,corediv)] = ts_data.ydata
                                    ts_group['ts%s_e_%s_zdata' % (nete,corediv)] = ts_data.zdata

                                    ts_group['ts%s_e_%s_xunits' % (nete,corediv)] = ts_data.xunits
                                    ts_group['ts%s_e_%s_yunits' % (nete,corediv)] = ts_data.yunits
                                    ts_group['ts%s_e_%s_zunits' % (nete,corediv)] = ts_data.zunits
                                except Exception as e:
                                    print('Bad shot %d - ts%s_%s\n%s' % (shotn,nete,corediv,e))
                                    with open('bad-shots.txt','a') as fid:
                                            fid.write('%d - ts%s_%s\n' % (shotn,nete,corediv))

                hf.close()


        if co2:
                hf = h5py.File('/cscratch/jalalvanda/outputs/'+ str(shotn)+'_co2.h5','w')
                for co2_chn in ['r0','v1','v2','v3']:
                        co2_group = hf.create_group('co2_%s' % (co2_chn))
                        for co2_idx in range(10):
                                print('co2%s_%i' % (co2_chn,co2_idx))

                                co2_data = gadata('den%s_uf_%i' % (co2_chn,co2_idx), shotn, connection=atlconn)

                                co2_group['co2_%s_%i_xdata' % (co2_chn,co2_idx)] = co2_data.xdata
                                co2_group['co2_%s_%i_ydata' % (co2_chn,co2_idx)] = co2_data.ydata
                                co2_group['co2_%s_%i_zdata' % (co2_chn,co2_idx)] = co2_data.zdata

                                co2_group['co2_%s_%i_xunits' % (co2_chn,co2_idx)] = co2_data.xunits
                                co2_group['co2_%s_%i_yunits' % (co2_chn,co2_idx)] = co2_data.yunits
                                co2_group['co2_%s_%i_zunits' % (co2_chn,co2_idx)] = co2_data.zunits

                hf.close()


        if cer:
                hf = h5py.File('/cscratch/jalalvanda/outputs/'+ str(shotn)+'_cer.h5','w')
                for pref in ['crstit', 'crsrott', 'crsampt']:
                        for chn in np.arange(5,25):
                                print('%s%d' % (pref,chn))
                                cer_group = hf.create_group('%s%d' % (pref,chn))
                                cer_data = gadata('%s%d' % (pref,chn), shotn, connection=atlconn)

                                cer_group['%s%d_xdata' % (pref,chn)] = cer_data.xdata
                                cer_group['%s%d_ydata' % (pref,chn)] = cer_data.ydata
                                cer_group['%s%d_zdata' % (pref,chn)] = cer_data.zdata

                                cer_group['%s%d_xunits' % (pref,chn)] = cer_data.xunits
                                cer_group['%s%d_yunits' % (pref,chn)] = cer_data.yunits
                                cer_group['%s%d_zunits' % (pref,chn)] = cer_data.zunits


        if mse:
                hf = h5py.File('/cscratch/jalalvanda/outputs/'+ str(shotn)+'_mse.h5','w')

                for k in range(69):
                        mse_group = hf.create_group('mse%d' % (k+1))
                        print('mse chn %i' % (k+1))
                        mse_data = gadata('mssgamma%d' % (k+1), shotn, connection=atlconn)

                        mse_group['xdata'] = mse_data.xdata
                        mse_group['ydata'] = mse_data.ydata
                        mse_group['zdata'] = mse_data.zdata
                        mse_group['xunits'] = mse_data.xunits
                        mse_group['yunits'] = mse_data.yunits
                        mse_group['zunits'] = mse_data.zunits

                hf.close()

        if magnetics:
                with open('magnetics_list.txt','r') as fid:
                        mag_list=fid.read().splitlines()
                hf = h5py.File('/cscratch/jalalvanda/outputs/'+ str(shotn)+'_magnetics.h5','w')

                for mag_name in mag_list:
                        mag_group = hf.create_group(mag_name)
                        print(mag_name)
                        mag_data = gadata(mag_name, shotn, connection=atlconn)

                        mag_group['xdata'] = mag_data.xdata
                        mag_group['ydata'] = mag_data.ydata
                        mag_group['zdata'] = mag_data.zdata
                        mag_group['xunits'] = mag_data.xunits
                        mag_group['yunits'] = mag_data.yunits
                        mag_group['zunits'] = mag_data.zunits

                hf.close()

#	print('time per shot:%ds' % (time.time()-t1))
