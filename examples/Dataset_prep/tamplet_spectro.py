import data_prep_obj as data_prep

discharge_obj = data_prep.DichargePerp()

suffix_list = ["co2_pl"]

# Example discharge number - replace with actual discharge
discharge = 12345

all_file_dict = discharge_obj.get_full_data(discharge, suffix_list, norm_mode="no")

# Get the file dictionary for the specific discharge
file_dict = all_file_dict

freq, time, amp_f_t = discharge_obj.spectro_calc(
    file_dict["co2_pl"]["r0"]["xdata"][:],
    file_dict["co2_pl"]["r0"]["zdata"][:],
    plot=True,
)

freq_enhanced, time_enhanced, amp_f_t_enhanced = discharge_obj.spec_filters(
    freq, time, amp_f_t
)

discharge_obj.spectro_plot(freq_enhanced, time_enhanced, amp_f_t_enhanced)
