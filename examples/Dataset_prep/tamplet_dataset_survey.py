import data_prep_obj as data_prep
import pickle

discharge_min=170000
discharge_max=200000-1

discharge_search_list=range(discharge_min,discharge_max+1)
suffix_list=['ece_cali','co2_pl']
prep_obj_1=data_prep.DatasetPrep(discharge_search_list, suffix_list)
print(prep_obj_1.discharge_search_list)
print(prep_obj_1.suffix_list)

discharge_list=prep_obj_1.filter_discharges()
print(discharge_list)


with open('co2_ece_discharge.pkl', 'wb') as file:
    pickle.dump(discharge_list, file)

for key in discharge_list.keys():
    print(f'{key}:{len(discharge_list[key])}')

over_lap=set(discharge_list['co2_pl']) & set(discharge_list['ece_cali'])

print(f'overlap:{len(over_lap)}')