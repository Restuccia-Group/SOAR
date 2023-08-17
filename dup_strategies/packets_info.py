import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

path_to_files = '/home/sharon/Documents/Research/Drones-Offloading/output/FPS/deadline/packets_info/packets_info/'
deadlines=[0.020, 0.015, 0.030, 0.010]
deadline=deadlines[0]
list_antenna=['2x2','3x3','4x4']
scenarios=['bb','aldrich']
path_save_figs='/home/sharon/Documents/Research/Drones-Offloading/output/FPS/deadline/packets_info/plots/'

scenarios_2x2_dup_1fps, axscenarios_2x2_dup_1fps = plt.subplots()
scenarios_2x2_dup_5fps, axscenarios_2x2_dup_5fps = plt.subplots()
scenarios_2x2_dup_15fps, axscenarios_2x2_dup_15fps = plt.subplots()
scenarios_2x2_dup_30fps, axscenarios_2x2_dup_30fps = plt.subplots()

scenarios_2x2_nodup_1fps, axscenarios_2x2_nodup_1fps = plt.subplots()
scenarios_2x2_nodup_5fps, axscenarios_2x2_nodup_5fps = plt.subplots()
scenarios_2x2_nodup_15fps, axscenarios_2x2_nodup_15fps = plt.subplots()
scenarios_2x2_nodup_30fps, axscenarios_2x2_nodup_30fps = plt.subplots()

scenarios_3x3_dup_1fps, axscenarios_3x3_dup_1fps = plt.subplots()
scenarios_3x3_dup_5fps, axscenarios_3x3_dup_5fps = plt.subplots()
scenarios_3x3_dup_15fps, axscenarios_3x3_dup_15fps = plt.subplots()
scenarios_3x3_dup_30fps, axscenarios_3x3_dup_30fps = plt.subplots()

scenarios_3x3_nodup_1fps, axscenarios_3x3_nodup_1fps = plt.subplots()
scenarios_3x3_nodup_5fps, axscenarios_3x3_nodup_5fps = plt.subplots()
scenarios_3x3_nodup_15fps, axscenarios_3x3_nodup_15fps = plt.subplots()
scenarios_3x3_nodup_30fps, axscenarios_3x3_nodup_30fps = plt.subplots()

scenarios_4x4_dup_1fps, axscenarios_4x4_dup_1fps = plt.subplots()
scenarios_4x4_dup_5fps, axscenarios_4x4_dup_5fps = plt.subplots()
scenarios_4x4_dup_15fps, axscenarios_4x4_dup_15fps = plt.subplots()
scenarios_4x4_dup_30fps, axscenarios_4x4_dup_30fps = plt.subplots()

scenarios_4x4_nodup_1fps, axscenarios_4x4_nodup_1fps = plt.subplots()
scenarios_4x4_nodup_5fps, axscenarios_4x4_nodup_5fps = plt.subplots()
scenarios_4x4_nodup_15fps, axscenarios_4x4_nodup_15fps = plt.subplots()
scenarios_4x4_nodup_30fps, axscenarios_4x4_nodup_30fps = plt.subplots()

def compare_scenarios(dup_x, dup_y,dup_std,nodup_x,nodup_y,nodup_std,scenario_sta, antenna_sta,fps_ap):

    if antenna_sta == '2x2' and fps_ap == '1fps':
        axscenarios_2x2_dup_1fps.errorbar(nodup_x, nodup_y, yerr=nodup_std, fmt='o', label=scenario_sta)
        axscenarios_2x2_nodup_1fps.errorbar(dup_x,dup_y,yerr=dup_std, fmt='o', label=scenario_sta)
        axscenarios_2x2_dup_1fps.set_title('Comparision 2x2 - With Duplication - 1fps')
        axscenarios_2x2_nodup_1fps.set_title('Comparision 2x2 - Without Duplication - 1fps')
        axscenarios_2x2_dup_1fps.set_xlabel('Time (s)')
        axscenarios_2x2_dup_1fps.set_ylabel('Percentage of Packets Lost (%)')
        axscenarios_2x2_nodup_1fps.set_xlabel('Time (s)')
        axscenarios_2x2_nodup_1fps.set_ylabel('Percentage of Packets Lost (%)')

        axscenarios_2x2_dup_1fps.legend()
        axscenarios_2x2_nodup_1fps.legend()

    if antenna_sta == '2x2' and fps_ap == '5fps':
        axscenarios_2x2_dup_5fps.errorbar(nodup_x, nodup_y, yerr=nodup_std, fmt='o', label=scenario_sta)
        axscenarios_2x2_nodup_5fps.errorbar(dup_x,dup_y,yerr=dup_std, fmt='o', label=scenario_sta)
        axscenarios_2x2_dup_5fps.set_title('Comparision 2x2 - With Duplication - 5fps')
        axscenarios_2x2_nodup_5fps.set_title('Comparision 2x2 - Without Duplication - 5fps')
        axscenarios_2x2_dup_5fps.set_xlabel('Time (s)')
        axscenarios_2x2_dup_5fps.set_ylabel('Percentage of Packets Lost (%)')
        axscenarios_2x2_nodup_5fps.set_xlabel('Time (s)')
        axscenarios_2x2_nodup_5fps.set_ylabel('Percentage of Packets Lost (%)')

        axscenarios_2x2_dup_5fps.legend()
        axscenarios_2x2_nodup_5fps.legend()

    if antenna_sta == '2x2' and fps_ap == '15fps':
        axscenarios_2x2_dup_15fps.errorbar(nodup_x, nodup_y, yerr=nodup_std, fmt='o', label=scenario_sta)
        axscenarios_2x2_nodup_15fps.errorbar(dup_x,dup_y,yerr=dup_std, fmt='o', label=scenario_sta)
        axscenarios_2x2_dup_15fps.set_title('Comparision 2x2 - With Duplication - 15fps')
        axscenarios_2x2_nodup_15fps.set_title('Comparision 2x2 - Without Duplication - 15fps')
        axscenarios_2x2_dup_15fps.set_xlabel('Time (s)')
        axscenarios_2x2_dup_15fps.set_ylabel('Percentage of Packets Lost (%)')
        axscenarios_2x2_nodup_15fps.set_xlabel('Time (s)')
        axscenarios_2x2_nodup_15fps.set_ylabel('Percentage of Packets Lost (%)')

        axscenarios_2x2_dup_15fps.legend()
        axscenarios_2x2_nodup_15fps.legend()

    if antenna_sta == '2x2' and fps_ap == '30fps':
        axscenarios_2x2_dup_30fps.errorbar(nodup_x, nodup_y, yerr=nodup_std, fmt='o', label=scenario_sta)
        axscenarios_2x2_nodup_30fps.errorbar(dup_x,dup_y,yerr=dup_std, fmt='o', label=scenario_sta)
        axscenarios_2x2_dup_30fps.set_title('Comparision 2x2 - With Duplication - 30fps')
        axscenarios_2x2_nodup_30fps.set_title('Comparision 2x2 - Without Duplication - 30fps')
        axscenarios_2x2_dup_30fps.set_xlabel('Time (s)')
        axscenarios_2x2_dup_30fps.set_ylabel('Percentage of Packets Lost (%)')
        axscenarios_2x2_nodup_30fps.set_xlabel('Time (s)')
        axscenarios_2x2_nodup_30fps.set_ylabel('Percentage of Packets Lost (%)')

        axscenarios_2x2_dup_30fps.legend()
        axscenarios_2x2_nodup_30fps.legend()

    if antenna_sta == '3x3' and fps_ap == '1fps':
        axscenarios_3x3_dup_1fps.errorbar(nodup_x, nodup_y, yerr=nodup_std, fmt='o', label=scenario_sta)
        axscenarios_3x3_nodup_1fps.errorbar(dup_x,dup_y,yerr=dup_std, fmt='o', label=scenario_sta)
        axscenarios_3x3_dup_1fps.set_title('Comparision 2x2 - With Duplication - 1fps')
        axscenarios_3x3_nodup_1fps.set_title('Comparision 2x2 - Without Duplication - 1fps')
        axscenarios_3x3_dup_1fps.set_xlabel('Time (s)')
        axscenarios_3x3_dup_1fps.set_ylabel('Percentage of Packets Lost (%)')
        axscenarios_3x3_nodup_1fps.set_xlabel('Time (s)')
        axscenarios_3x3_nodup_1fps.set_ylabel('Percentage of Packets Lost (%)')

        axscenarios_3x3_dup_1fps.legend()
        axscenarios_3x3_nodup_1fps.legend()

    if antenna_sta == '3x3' and fps_ap == '5fps':
        axscenarios_3x3_dup_5fps.errorbar(nodup_x, nodup_y, yerr=nodup_std, fmt='o', label=scenario_sta)
        axscenarios_3x3_nodup_5fps.errorbar(dup_x,dup_y,yerr=dup_std, fmt='o', label=scenario_sta)
        axscenarios_3x3_dup_5fps.set_title('Comparision 3x3 - With Duplication - 5fps')
        axscenarios_3x3_nodup_5fps.set_title('Comparision 3x3 - Without Duplication - 5fps')
        axscenarios_3x3_dup_5fps.set_xlabel('Time (s)')
        axscenarios_3x3_dup_5fps.set_ylabel('Percentage of Packets Lost (%)')
        axscenarios_3x3_nodup_5fps.set_xlabel('Time (s)')
        axscenarios_3x3_nodup_5fps.set_ylabel('Percentage of Packets Lost (%)')

        axscenarios_3x3_dup_5fps.legend()
        axscenarios_3x3_nodup_5fps.legend()

    if antenna_sta == '3x3' and fps_ap == '15fps':
        axscenarios_3x3_dup_15fps.errorbar(nodup_x, nodup_y, yerr=nodup_std, fmt='o', label=scenario_sta)
        axscenarios_3x3_nodup_15fps.errorbar(dup_x,dup_y,yerr=dup_std, fmt='o', label=scenario_sta)
        axscenarios_3x3_dup_15fps.set_title('Comparision 3x3 - With Duplication - 15fps')
        axscenarios_3x3_nodup_15fps.set_title('Comparision 3x3 - Without Duplication - 15fps')
        axscenarios_3x3_dup_15fps.set_xlabel('Time (s)')
        axscenarios_3x3_dup_15fps.set_ylabel('Percentage of Packets Lost (%)')
        axscenarios_3x3_nodup_15fps.set_xlabel('Time (s)')
        axscenarios_3x3_nodup_15fps.set_ylabel('Percentage of Packets Lost (%)')

        axscenarios_3x3_dup_15fps.legend()
        axscenarios_3x3_nodup_15fps.legend()

    if antenna_sta == '3x3' and fps_ap == '30fps':
        axscenarios_3x3_dup_30fps.errorbar(nodup_x, nodup_y, yerr=nodup_std, fmt='o', label=scenario_sta)
        axscenarios_3x3_nodup_30fps.errorbar(dup_x,dup_y,yerr=dup_std, fmt='o', label=scenario_sta)
        axscenarios_3x3_dup_30fps.set_title('Comparision 3x3 - With Duplication - 30fps')
        axscenarios_3x3_nodup_30fps.set_title('Comparision 3x3 - Without Duplication - 30fps')
        axscenarios_3x3_dup_30fps.set_xlabel('Time (s)')
        axscenarios_3x3_dup_30fps.set_ylabel('Percentage of Packets Lost (%)')
        axscenarios_3x3_nodup_30fps.set_xlabel('Time (s)')
        axscenarios_3x3_nodup_30fps.set_ylabel('Percentage of Packets Lost (%)')

        axscenarios_3x3_dup_30fps.legend()
        axscenarios_3x3_nodup_30fps.legend()

    if antenna_sta == '4x4' and fps_ap == '1fps':
        axscenarios_4x4_dup_1fps.errorbar(nodup_x, nodup_y, yerr=nodup_std, fmt='o', label=scenario_sta)
        axscenarios_4x4_nodup_1fps.errorbar(dup_x,dup_y,yerr=dup_std, fmt='o', label=scenario_sta)
        axscenarios_4x4_dup_1fps.set_title('Comparision 4x4 - With Duplication - 1fps')
        axscenarios_4x4_nodup_1fps.set_title('Comparision 4x4 - Without Duplication - 1fps')
        axscenarios_4x4_dup_1fps.set_xlabel('Time (s)')
        axscenarios_4x4_dup_1fps.set_ylabel('Percentage of Packets Lost (%)')
        axscenarios_4x4_nodup_1fps.set_xlabel('Time (s)')
        axscenarios_4x4_nodup_1fps.set_ylabel('Percentage of Packets Lost (%)')

        axscenarios_4x4_dup_1fps.legend()
        axscenarios_4x4_nodup_1fps.legend()

    if antenna_sta == '4x4' and fps_ap == '5fps':
        axscenarios_4x4_dup_5fps.errorbar(nodup_x, nodup_y, yerr=nodup_std, fmt='o', label=scenario_sta)
        axscenarios_4x4_nodup_5fps.errorbar(dup_x,dup_y,yerr=dup_std, fmt='o', label=scenario_sta)
        axscenarios_4x4_dup_5fps.set_title('Comparision 4x4 - With Duplication - 5fps')
        axscenarios_4x4_nodup_5fps.set_title('Comparision 4x4 - Without Duplication - 5fps')
        axscenarios_4x4_dup_5fps.set_xlabel('Time (s)')
        axscenarios_4x4_dup_5fps.set_ylabel('Percentage of Packets Lost (%)')
        axscenarios_4x4_nodup_5fps.set_xlabel('Time (s)')
        axscenarios_4x4_nodup_5fps.set_ylabel('Percentage of Packets Lost (%)')

        axscenarios_4x4_dup_5fps.legend()
        axscenarios_4x4_nodup_5fps.legend()

    if antenna_sta == '4x4' and fps_ap == '15fps':
        axscenarios_4x4_dup_15fps.errorbar(nodup_x, nodup_y, yerr=nodup_std, fmt='o', label=scenario_sta)
        axscenarios_4x4_nodup_15fps.errorbar(dup_x,dup_y,yerr=dup_std, fmt='o', label=scenario_sta)
        axscenarios_4x4_dup_15fps.set_title('Comparision 4x4 - With Duplication - 15fps')
        axscenarios_4x4_nodup_15fps.set_title('Comparision 4x4 - Without Duplication - 15fps')
        axscenarios_4x4_dup_15fps.set_xlabel('Time (s)')
        axscenarios_4x4_dup_15fps.set_ylabel('Percentage of Packets Lost (%)')
        axscenarios_4x4_nodup_15fps.set_xlabel('Time (s)')
        axscenarios_4x4_nodup_15fps.set_ylabel('Percentage of Packets Lost (%)')

        axscenarios_4x4_dup_15fps.legend()
        axscenarios_4x4_nodup_15fps.legend()

    if antenna_sta == '4x4' and fps_ap == '30fps':
        axscenarios_4x4_dup_30fps.errorbar(nodup_x, nodup_y, yerr=nodup_std, fmt='o', label=scenario_sta)
        axscenarios_4x4_nodup_30fps.errorbar(dup_x,dup_y,yerr=dup_std, fmt='o', label=scenario_sta)
        axscenarios_4x4_dup_30fps.set_title('Comparision 4x4 - With Duplication - 30fps')
        axscenarios_4x4_nodup_30fps.set_title('Comparision 4x4 - Without Duplication - 30fps')
        axscenarios_4x4_dup_30fps.set_xlabel('Time (s)')
        axscenarios_4x4_dup_30fps.set_ylabel('Percentage of Packets Lost (%)')
        axscenarios_4x4_nodup_30fps.set_xlabel('Time (s)')
        axscenarios_4x4_nodup_30fps.set_ylabel('Percentage of Packets Lost (%)')

        axscenarios_4x4_dup_30fps.legend()
        axscenarios_4x4_nodup_30fps.legend()

def clean_plt():
    scenarios_2x2_dup_1fps.clf(), scenarios_2x2_dup_5fps.clf()
    scenarios_2x2_dup_15fps.clf(), scenarios_2x2_dup_30fps.clf()

    scenarios_2x2_nodup_1fps.clf(), scenarios_2x2_nodup_5fps.clf()
    scenarios_2x2_nodup_15fps.clf(), scenarios_2x2_nodup_30fps.clf()

    scenarios_3x3_dup_1fps.clf(), scenarios_3x3_dup_5fps.clf()
    scenarios_3x3_dup_15fps.clf(), scenarios_3x3_dup_30fps.clf()

    scenarios_3x3_nodup_1fps.clf(), scenarios_3x3_nodup_5fps.clf()
    scenarios_3x3_nodup_15fps.clf(), scenarios_3x3_nodup_30fps.clf()

    scenarios_4x4_dup_1fps.clf(), scenarios_4x4_dup_5fps.clf()
    scenarios_4x4_dup_15fps.clf(), scenarios_4x4_dup_30fps.clf()

    scenarios_4x4_nodup_1fps.clf(), scenarios_4x4_nodup_5fps.clf()
    scenarios_4x4_nodup_15fps.clf(), scenarios_4x4_nodup_30fps.clf()

def compare_deadlines():
    print("")

def missing_packets_with_duplication(ap_packets, sta_packets, user_sta, config):
    """
    This function create sets of packets sent separeted by images and compares whether the same amount sent was received 
    """
    n_antennas=int(config[-1])
    sets_ap=[]; idx_of_set=[];new_set=set()
    for idx,element in enumerate(ap_packets):
        if element != 0:
            new_set.add(int(element))
        else:
            try:
                sets_ap.append(new_set)
                idx_of_set.append(idx)
                new_set=set()
                new_set.add(int(element))
            except:
                new_set=set()
                new_set.add(int(element))

    sets_sta=[]
    for element in sta_packets:
        if element != 0:
            new_set.add(int(element))
        else:
            try:
                sets_sta.append(new_set)
                new_set=set()
                new_set.add(int(element))
            except:
                new_set=set()
                new_set.add(int(element))

    sets_sta = [s for s in sets_sta if s]
    sets_ap = [s for s in sets_ap if s]

    image_index = []; tot_set_sta = len(sets_sta); to_check_ap=[];
    for idx_setAP, set_ap in enumerate(sets_ap):
        set_removed=False
        for set_sta in sets_sta:
            #if len(set_ap & set_sta) == len(set_ap) and len(set_ap & set_sta) == len(set_sta) :
            if len(set_ap & set_sta) ==  len(set_ap):
                set_removed = True
                if idx_setAP%n_antennas == 0:
                    sets_sta.remove(set_sta)
                break;
        
        #If the set from AP couldn't be found then store it
        if set_removed == False:
            image_index.append(idx_setAP)
            to_check_ap.append(set_ap)

    packets_lost_list=[]
    """
    for set_sta in sets_sta:
        list_stas = []
        for idx_ap,set_ap in enumerate(to_check_ap):
            if len(set_sta - set_ap) == 0:
                list_stas.append((len(set_ap - set_sta),image_index[idx_ap], idx_ap))

        if list_stas:
            packets_lost_list.append(min(list_stas, key=lambda x: x[0]))
            image_index.remove(packets_lost_list[-1][1])
            del to_check_ap[packets_lost_list[-1][2]]
    """
    missing_indexes=[idx for idx in range(len(sets_ap)) if idx in image_index]

    tot_lost_packet_sta=[]; other_indexes=[];
    for element in packets_lost_list:
        tot_lost_packet_sta.append(element[0])
        other_indexes.append(element[1])

    if user_sta == 'n2':
        "Completing information for 1 station when the ap sent information to the 2 stations"
        new_missing_indexes=[element for idx,element in enumerate(missing_indexes) if (idx % 2) == 0]
        sets_lost=[]; missing_indexes=[];
        for i in new_missing_indexes:
            missing_indexes += [idx_of_set[i]]
            sets_lost += [len(sets_ap[i])]
    else:
        new_missing_indexes=missing_indexes
        missing_indexes = []; sets_lost=[];
        for i in new_missing_indexes:
            missing_indexes += [idx_of_set[i]]
            sets_lost+=[len(sets_ap[i])]

    packets_lost_dup = sets_lost + tot_lost_packet_sta
    return packets_lost_dup, missing_indexes+other_indexes

def missing_packets_no_duplication(ap_packets,indexes_ap_packets,sta_packets,user_sta):
    index_losses=[]; idx_sta=0;
    num_images_sent=len([idx for idx,e in enumerate(ap_packets) if e == 0 ])
    image_loss=[0]*num_images_sent;

    for idx_ap,ap_packet in enumerate(ap_packets):
        if user_sta=='n2': idx_to_use=indexes_ap_packets[idx_ap]
        else: idx_to_use=idx_ap

        if ap_packet != sta_packets[idx_sta]:
            image_loss[(idx_ap % num_images_sent)]+=1
        else:
            idx_sta=min(idx_sta+1,len(sta_packets)-1)

        if ap_packet == 0: index_losses += [idx_to_use]

    return image_loss,index_losses

def choose_packets(ap_packets):
    """
    Choose the packets of only one user
    """

    skip_next_zero = True
    result=[]; indexes_skipped=[]; indexes=[]
    for idx,element in enumerate(ap_packets):
        if element == 0 and skip_next_zero:
            skip_next_zero = False
            indexes_skipped.append(idx)
            continue
        
        if element == 0:
            indexes.append(idx)
            skip_next_zero = True

    anchor=0; result_indexes=[]
    for element in indexes:
        try:
            tmp = ap_packets[element:indexes_skipped[anchor+1]].tolist()
            filled_numbers = list(range(element, indexes_skipped[anchor + 1]))
        except:
            tmp = ap_packets[element:len(ap_packets)-1].tolist()
            filled_numbers = list(range(element, len(ap_packets)-1))
        result+=tmp
        result_indexes+=filled_numbers
        anchor+=1

    return result, result_indexes

def adjust_to_deadline(df_sta, deadline):
    #ap_packets = df_ap['Packet'].values
    #sta_packets = df_sta['Packet'].values
    """
    Every time a new packet comes, just cut until the deadline in both dataframes
    """
    columns = ['Timestamp', 'Packet']
    """
    new_df_ap = pd.DataFrame(columns=columns)
    anchor=df_ap.iloc[0]['Timestamp']; waiting = False
    for row in df_ap.iterrows():
        if (row[1]['Timestamp'] - anchor) < deadline:
            new_row1 = {'Timestamp': row[1]['Timestamp'], 'Packet': row[1]['Packet']}
            new_df_ap = new_df_ap._append(new_row1, ignore_index=True)
        else:
            waiting = True
        
        if waiting == True and row[1]['Packet'] == 0:
            anchor= row[1]['Timestamp']
            new_row1 = {'Timestamp': row[1]['Timestamp'], 'Packet': row[1]['Packet']}
            new_df_ap = new_df_ap._append(new_row1, ignore_index=True)
            waiting = False
    """

    new_df_sta = pd.DataFrame(columns=columns)

    anchor=df_sta.iloc[0]['Timestamp']; waiting = False
    for row in df_sta.iterrows():
        if (row[1]['Timestamp'] - anchor) < deadline:
            new_row1 = {'Timestamp': row[1]['Timestamp'], 'Packet': int(row[1]['Packet'])}
            new_df_sta = new_df_sta._append(new_row1, ignore_index=True)
        else:
            waiting = True
        
        if waiting == True and row[1]['Packet'] == 0:
            anchor= row[1]['Timestamp']
            new_row1 = {'Timestamp': row[1]['Timestamp'], 'Packet': int(row[1]['Packet'])}
            new_df_sta = new_df_sta._append(new_row1, ignore_index=True)
            waiting = False

    return new_df_sta

deadline=deadlines[3]
for filename_ap in os.listdir(path_to_files):
    print("************************** NEW FILE BEING CHECKED ************************")
    if 'ap' in filename_ap:
        df_ap=pd.read_csv(path_to_files+filename_ap)
        #print("Filename ap: ",path_to_files+filename_ap)
        antenna_ap=filename_ap.split(".")[0].split("_")[2]
        user_ap=filename_ap.split(".")[0].split("_")[3]
        fps_ap=filename_ap.split(".")[0].split("_")[4]
        scenario_ap=filename_ap.split(".")[0].split("_")[6]

        for filename_rover in os.listdir(path_to_files):
            dup_stats=[]; no_sup_stats=[]

            if 's102' in filename_rover and antenna_ap in list_antenna:
                antenna_sta=filename_rover.split(".")[0].split("_")[2]
                user_sta=filename_rover.split(".")[0].split("_")[3]
                fps_sta=filename_rover.split(".")[0].split("_")[4]
                scenario_sta=filename_rover.split(".")[0].split("_")[6]

                if antenna_sta == antenna_ap and user_ap == user_sta and fps_ap == fps_sta and scenario_ap == scenario_sta:
                    df_sta=pd.read_csv(path_to_files+filename_rover)
                    #ap_packets = df_ap['Packet'].values
                    #sta_packets = df_sta['Packet'].values
                    #seq_num=np.zeros([41,])

                    df_sta=adjust_to_deadline(df_sta,deadline)

                    ap_packets = df_ap['Packet'].values
                    sta_packets = df_sta['Packet'].values
                    sta_packets = [int(item) for item in df_sta['Packet'].values]

                    packet_images_lost_dup, image_indexes_dup = missing_packets_with_duplication(ap_packets, sta_packets
                                                                                                    , user_sta,antenna_ap)

                    assert len(packet_images_lost_dup) == len(image_indexes_dup), "The total packets lost and the length of image " \
                                                                    "indexes are not equal in "+filename_rover+ ". Also, " \
                                                                    "tot_packets_lost is: "+str(len(packet_images_lost_dup))+" and " \
                                                                    "length of images is: "+str(len(image_indexes_dup))

                    indexes_ap_packets=[] #Choose the packets number of the
                    if user_sta == 'n2':
                        ap_packets, indexes_ap_packets = choose_packets(ap_packets)
                        x_ap_packets=df_ap.iloc[indexes_ap_packets]['Timestamp'] - df_ap.iloc[0]['Timestamp']
                        assert len(ap_packets) == len(indexes_ap_packets), "Size of indexes in ap_packets doesn't matches " \
                                                                        "the size of ap_packets"
                    else:
                        x_ap_packets=df_ap.iloc[:]['Timestamp'] - df_ap.iloc[0]['Timestamp']
                    packet_images_lost_nodup, image_indexes_nodup = missing_packets_no_duplication(ap_packets,indexes_ap_packets,sta_packets,user_sta)

                    # Bins resolution to get the standard deviations
                    bins = range(0,int(df_ap.iloc[len(df_ap)-1]['Timestamp'] - df_ap.iloc[0]['Timestamp']),5)
                    ap_packets_df = pd.DataFrame({'Seconds':x_ap_packets.values.tolist(), 'Total':[1]*len(ap_packets)})

                    x_dup = df_ap.iloc[image_indexes_dup]['Timestamp'] - df_ap.iloc[0]['Timestamp']
                    y_dup = [element for element in packet_images_lost_dup]
                    dup_df = pd.DataFrame({'Seconds':x_dup.values.tolist(), 'Losses':y_dup})

                    if len(x_dup)>0:
                        flag=True
                        min_seconds = dup_df.iloc[0]['Seconds']; sum_packets = 0;
                        for row_dup in dup_df.iterrows():
                            for row_ap in ap_packets_df.iterrows():
                                if row_dup[1]['Seconds'] > row_ap[1]['Seconds'] and min_seconds < row_ap[1]['Seconds']:
                                    sum_packets += row_ap[1]['Total']
                                else:
                                    min_seconds = row_dup[1]['Seconds']
                                    sum_packets = row_ap[1]['Total']
                                    break

                            row_dup[1]['Losses'] = row_dup[1]['Losses']/sum_packets

                        dup_df['bin'] = pd.cut(dup_df['Seconds'], bins)
                        dup_grouped = dup_df.groupby('bin')

                        dup_aggregated_result = dup_grouped['Losses'].agg(['mean', 'std'])
                        dup_bin_values = dup_aggregated_result.index._values

                        dup_bin_values_list=[]
                        for element in dup_bin_values: dup_bin_values_list.append(element.mid)

                        dup_aggregated_result['mean'] = dup_aggregated_result['mean'].fillna(0.0)
                        dup_aggregated_result['std'] = dup_aggregated_result['std'].fillna(0.0)
                    else:
                        flag=False
                    x_nodup = df_ap.iloc[image_indexes_nodup]['Timestamp'] - df_ap.iloc[0]['Timestamp']
                    y_nodup = [ element for element in packet_images_lost_nodup]
                    nodup_df = pd.DataFrame({'Seconds':x_nodup.values.tolist(), 'Losses':y_nodup})
                    nodup_df['bin'] = pd.cut(nodup_df['Seconds'], bins)

                    min_seconds=nodup_df.iloc[0]['Seconds']; sum_packets=0
                    for row_nodup in nodup_df.iterrows():
                        for row_ap in ap_packets_df.iterrows():
                            if row_nodup[1]['Seconds'] > row_ap[1]['Seconds'] and min_seconds < row_ap[1]['Seconds']:
                                sum_packets += row_ap[1]['Total']
                            else:
                                min_seconds = row_nodup[1]['Seconds']
                                sum_packets = row_ap[1]['Total']
                                break

                        row_nodup[1]['Losses'] = row_nodup[1]['Losses']/sum_packets

                    nodup_grouped = nodup_df.groupby('bin')
                    nodup_aggregated_result = nodup_grouped['Losses'].agg(['mean', 'std'])
                    nodup_bin_values = nodup_aggregated_result.index._values
                    nodup_bin_values_list=[]
                    for element in nodup_bin_values:nodup_bin_values_list.append(element.mid)

                    nodup_aggregated_result['mean'] = nodup_aggregated_result['mean'].fillna(0.0)
                    nodup_aggregated_result['std'] = nodup_aggregated_result['std'].fillna(0.0)

                    #plt.title("antenna: "+antenna_sta+" scenario:"+scenario_sta+" user:"+user_sta+" deadline: "+str(deadline))
                    #plt.errorbar(nodup_bin_values_list, nodup_aggregated_result['mean'], yerr=nodup_aggregated_result['std'], fmt='o', label='Without Duplication')
                    if flag:
                        #plt.errorbar(dup_bin_values_list, dup_aggregated_result['mean'], yerr=dup_aggregated_result['std'], fmt='o', label='With Duplication')
                        if user_sta =='n2':
                            compare_scenarios(dup_bin_values_list, dup_aggregated_result['mean'], dup_aggregated_result['std'], nodup_bin_values_list, nodup_aggregated_result['mean'], nodup_aggregated_result['std'], scenario_sta, antenna_sta,fps_ap)
                    else:
                        #plt.errorbar(nodup_bin_values_list, [0] *len(nodup_aggregated_result['mean']),
                        #             yerr=[0]*len(nodup_aggregated_result['std']), fmt='o', label='Without Duplication')
                        if user_sta == 'n2':
                            compare_scenarios(nodup_bin_values_list,  [0] *len(nodup_aggregated_result['mean']), [0]*len(nodup_aggregated_result['std']), nodup_bin_values_list, nodup_aggregated_result['mean'], nodup_aggregated_result['std'], scenario_sta, antenna_sta,fps_ap)

                    #plt.legend()
                    #plt.ylabel("Percentage of Packets Lost (%)")
                    #plt.xlabel("Time (s)")

                    #compare_deadlines()
                    #plt.show()
                    #plt.savefig(path_save_figs+"antenna_"+antenna_sta+"_scenario_"+scenario_sta+"_user_"+user_sta+"_deadline_"+str(deadline)+".jpg")
                    #plt.clf()

print("Scenarios are saving ...")
scenarios_4x4_nodup_1fps.savefig(path_save_figs+"1fps_4x4_scenarios_comparision_nodup"+"_user_"+user_sta+"_deadline_"+str(deadline)+".jpg")
scenarios_4x4_nodup_5fps.savefig(path_save_figs+"5fps_4x4_scenarios_comparision_nodup"+"_user_"+user_sta+"_deadline_"+str(deadline)+".jpg")
scenarios_4x4_nodup_15fps.savefig(path_save_figs+"15fps_4x4_scenarios_comparision_nodup"+"_user_"+user_sta+"_deadline_"+str(deadline)+".jpg")
scenarios_4x4_nodup_30fps.savefig(path_save_figs+"30fps_4x4_scenarios_comparision_nodup"+"_user_"+user_sta+"_deadline_"+str(deadline)+".jpg")

scenarios_4x4_dup_1fps.savefig(path_save_figs+"1fps_4x4_scenarios_comparision_dup"+"_user_"+user_sta+"_deadline_"+str(deadline)+".jpg")
scenarios_4x4_dup_5fps.savefig(path_save_figs+"5fps_4x4_scenarios_comparision_dup"+"_user_"+user_sta+"_deadline_"+str(deadline)+".jpg")
scenarios_4x4_dup_15fps.savefig(path_save_figs+"15fps_4x4_scenarios_comparision_dup"+"_user_"+user_sta+"_deadline_"+str(deadline)+".jpg")
scenarios_4x4_dup_30fps.savefig(path_save_figs+"30fps_4x4_scenarios_comparision_dup"+"_user_"+user_sta+"_deadline_"+str(deadline)+".jpg")

scenarios_3x3_dup_1fps.savefig(path_save_figs+"1fps_3x3_scenarios_comparision_dup"+"_user_"+user_sta+"_deadline_"+str(deadline)+".jpg")
scenarios_3x3_dup_5fps.savefig(path_save_figs+"5fps_3x3_scenarios_comparision_dup"+"_user_"+user_sta+"_deadline_"+str(deadline)+".jpg")
scenarios_3x3_dup_15fps.savefig(path_save_figs+"15fps_3x3_scenarios_comparision_dup"+"_user_"+user_sta+"_deadline_"+str(deadline)+".jpg")
scenarios_3x3_dup_30fps.savefig(path_save_figs+"30fps_3x3_scenarios_comparision_dup"+"_user_"+user_sta+"_deadline_"+str(deadline)+".jpg")

scenarios_3x3_nodup_1fps.savefig(path_save_figs+"1fps_3x3_scenarios_comparision_nodup"+"_user_"+user_sta+"_deadline_"+str(deadline)+".jpg")
scenarios_3x3_nodup_5fps.savefig(path_save_figs+"5fps_3x3_scenarios_comparision_nodup"+"_user_"+user_sta+"_deadline_"+str(deadline)+".jpg")
scenarios_3x3_nodup_15fps.savefig(path_save_figs+"15fps_3x3_scenarios_comparision_nodup"+"_user_"+user_sta+"_deadline_"+str(deadline)+".jpg")
scenarios_3x3_nodup_30fps.savefig(path_save_figs+"30fps_3x3_scenarios_comparision_nodup"+"_user_"+user_sta+"_deadline_"+str(deadline)+".jpg")

scenarios_2x2_dup_1fps.savefig(path_save_figs+"1fps_2x2_scenarios_comparision_dup"+"_user_"+user_sta+"_deadline_"+str(deadline)+".jpg")
scenarios_2x2_dup_5fps.savefig(path_save_figs+"5fps_2x2_scenarios_comparision_dup"+"_user_"+user_sta+"_deadline_"+str(deadline)+".jpg")
scenarios_2x2_dup_15fps.savefig(path_save_figs+"15fps_2x2_scenarios_comparision_dup"+"_user_"+user_sta+"_deadline_"+str(deadline)+".jpg")
scenarios_2x2_dup_30fps.savefig(path_save_figs+"30fps_2x2_scenarios_comparision_dup"+"_user_"+user_sta+"_deadline_"+str(deadline)+".jpg")

scenarios_2x2_nodup_1fps.savefig(path_save_figs+"1fps_2x2_scenarios_comparision_nodup"+"_user_"+user_sta+"_deadline_"+str(deadline)+".jpg")
scenarios_2x2_nodup_5fps.savefig(path_save_figs+"5fps_2x2_scenarios_comparision_nodup"+"_user_"+user_sta+"_deadline_"+str(deadline)+".jpg")
scenarios_2x2_nodup_15fps.savefig(path_save_figs+"15fps_2x2_scenarios_comparision_nodup"+"_user_"+user_sta+"_deadline_"+str(deadline)+".jpg")
scenarios_2x2_nodup_30fps.savefig(path_save_figs+"30fps_2x2_scenarios_comparision_nodup"+"_user_"+user_sta+"_deadline_"+str(deadline)+".jpg")
#clean_plt()

