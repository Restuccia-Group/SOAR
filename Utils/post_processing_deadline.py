import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import codecs
import argparse  
import os 
import seaborn as sns

def preprocess_telemetry(telemetry):
    """
    Preprocess telemetry data
    Input: telemetry dataframe
    Output: telemetry dataframe with distance and acceleration
    """

    df = telemetry.drop_duplicates()

    df['North_m'].replace('None', np.nan, inplace=True)
    df['East_m'].replace('None', np.nan, inplace=True)

    df.dropna(subset=['North_m', 'East_m'], how='all', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['color']='blue'
    df['size']=10    
    change_color_idx=df.index[df['Mode']=='VehicleMode:GUIDED']
    change_size_idx=df.index[df['Mode']=='VehicleMode:GUIDED']
    df.loc[change_color_idx,'color']='red'
    df.loc[change_size_idx,'size']=90
    print(change_color_idx)
    
    for idx,element in enumerate([['-7.5','-11.5'],['-3','-9.5'],['2','-8']]):
        df['North_m_sta_'+str(idx+1)] = float(element[0])
        df['East_m_sta_'+str(idx+1)] = float(element[1])
        df['Down_m_sta_'+str(idx+1)] = float(0)
        
        row = df.iloc[-1].copy()  
        row['North_m']=element[0] ; row['East_m']=element[1]; row['color']='red'; row['size']=90
        new_df = pd.DataFrame([row])
        df = pd.concat([df, new_df], axis=0, ignore_index=True)

    df['distance_1']=(df['North_m_sta_1'].sub(df['North_m'].astype(float), axis=0))**2 +  \
                        (df['East_m_sta_1'].sub(df['East_m'].astype(float), axis=0))**2  + \
                        (df['Down_m_sta_1'].sub(df['Down_m'].astype(float).abs(), axis=0))**2
    
    df['distance_1']=df['distance_1'].apply(np.sqrt)

    df['distance_2']=(df['North_m_sta_2'].sub(df['North_m'].astype(float), axis=0))**2 +  \
                (df['East_m_sta_2'].sub(df['East_m'].astype(float), axis=0))**2  + \
                (df['Down_m_sta_2'].sub(df['Down_m'].astype(float).abs(), axis=0))**2

    df['distance_2']=df['distance_2'].apply(np.sqrt)

    df['distance_3']=(df['North_m_sta_3'].sub(df['North_m'].astype(float), axis=0))**2 +  \
                        (df['East_m_sta_3'].sub(df['East_m'].astype(float), axis=0))**2  + \
                        (df['Down_m_sta_3'].sub(df['Down_m'].astype(float).abs(), axis=0))**2
    
    df['distance_3']=df['distance_3'].apply(np.sqrt)

    #Convert float to strings
    df['distance_1'].astype(str); df['distance_2'].astype(str); df['distance_3'].astype(str) ; 
    df['North_m_sta_1'].astype(str); df['North_m_sta_2'].astype(str); df['North_m_sta_3'].astype(str)
    df['East_m_sta_1'].astype(str); df['East_m_sta_2'].astype(str); df['East_m_sta_3'].astype(str)
    df['North_m'].astype(str); df['East_m'].astype(str)

    return df
        
def conversations_and_telemetry(conversations,plots,telemetry,config,save_config):
    """
    Save Delays/Data Rate - Distance/Acceleration plots
    Input: Conversations, Telemetry files, Output Plots, Output Folder, and Plot Titles
    Output: Visualization of Delays/Data Rate - Distance/Acceleration plots
    """
    f_conv = open(conversations)
    data = pd.read_csv(f_conv)
    data.dropna(subset=['Duration', 'Bits/s A → B'], how='all', inplace=True)

    f_tel = open(telemetry)
    telem = pd.read_csv(f_tel,on_bad_lines='skip')
    print("length of data ",len(data))
    data_filtered=data[(data['Duration'].astype('float')>0)  & (data['Bits/s A → B'].astype('float')>0) &
                       (data['Duration'].astype('float')<0.08)  & (data['Bits/s A → B'].astype('float')>0)]
    data_filtered['Duration']=data_filtered['Duration']*1000
    print("length of filtered data ",len(data_filtered))
    data_filtered['Bits/s A → B']= data_filtered['Bits/s A → B']/1000000

    telem=preprocess_telemetry(telem)

    y_dis_1=np.zeros((len(data_filtered['Duration'],)))
    y_dis_2=np.zeros((len(data_filtered['Duration'],)))
    y_dis_3=np.zeros((len(data_filtered['Duration'],)))

    y_acc_x=np.zeros((len(data_filtered['Duration'],)))

    telem['Xaccel'] = telem['Xaccel'].fillna(0)

    for idx,delay in enumerate(data_filtered['Duration']):
        n_slots=round(delay/10)
        if n_slots==0: n_slots=1
        y_dis_1[idx]=int(sum(telem[idx:idx+n_slots]['distance_1'])/n_slots)
        y_dis_2[idx]=int(sum(telem[idx:idx+n_slots]['distance_2'])/n_slots)
        y_dis_3[idx]=int(sum(telem[idx:idx+n_slots]['distance_3'])/n_slots)
        y_acc_x[idx]=round(sum(telem[idx:idx+n_slots]['Xaccel'].abs()*0.01)/n_slots,1)

    data_filtered['distance_1']=y_dis_1
    data_filtered['distance_2']=y_dis_2
    data_filtered['distance_3']=y_dis_3
    data_filtered['x_accel'] = y_acc_x 
    
    
    fig, ax = plt.subplots()
    sns.boxplot(x="distance_1", y="Duration", data=data_filtered, ax=ax)
    sns.regplot(x="distance_1", y="Duration", data=data_filtered, ax=ax, scatter=False, truncate=False, ci=None, color='red',line_kws={"linestyle":'--'})

    plt.ylim(0,85)
    plt.grid()
    plt.title("Distance vs Delay - {}".format(config))
    plt.xlabel("Distance (m)")
    plt.ylabel("Delay (ms)")
    plt.savefig(plots+"/"+save_config+"_distance_delay.png")
 
    fig, ax = plt.subplots()
    sns.boxplot(x="x_accel", y="Duration", data=data_filtered, ax=ax)
    sns.regplot(x="x_accel", y="Duration", data=data_filtered, ax=ax, scatter=False, truncate=False, ci=None, color='red',line_kws={"linestyle":'--'})

    plt.ylim(0,85)
    plt.grid()
    plt.title("Acceleration in X-axis vs Delay - {}".format(config))
    plt.xlabel("Acceleration (m/s^2)")
    plt.ylabel("Delay (ms)")
    plt.savefig(plots+"/"+save_config+"_accel_delay.png")

    fig, ax = plt.subplots()
    sns.boxplot(x="distance_1", y="Bits/s A → B", data=data_filtered, ax=ax)
    sns.regplot(x="distance_1", y="Bits/s A → B", data=data_filtered, ax=ax, scatter=False, truncate=False, ci=None, color='red',line_kws={"linestyle":'--'})

    plt.ylim(0,250)
    plt.grid()
    plt.title("Distance vs Data Rate - {}".format(config))
    plt.xlabel("Distance (m)")
    plt.ylabel("Data Rate (Mb/s)")
    plt.savefig(plots+"/"+save_config+"_distance_datarate.png")

    fig, ax = plt.subplots()
    sns.boxplot(x="x_accel", y="Bits/s A → B", data=data_filtered, ax=ax)
    sns.regplot(x="x_accel", y="Bits/s A → B", data=data_filtered, ax=ax, scatter=False, truncate=False, ci=None, color='red',line_kws={"linestyle":'--'})

    plt.ylim(0,250)
    plt.grid()
    plt.title("Acceleration in X-axis vs Data Rate - {}".format(config))
    plt.xlabel("Acceleration (m/s^2)")
    plt.ylabel("Data Rate (Mb/s)")
    plt.savefig(plots+"/"+save_config+"_accel_datarate.png")

    plt.show()
 
def deadline_pcaps(json_sta,plots_folder,deadline,output, json_ap):
    """
    Delay based on a deadline in seconds 
    Input: JSON files of base station and access point, Deadline, Output Folder, Plots Folder
    Output: Visualization of Delay based on the deadline restriction
    """

    f_sta = open(json_sta)
    data_sta = json.load(f_sta)
    print(" FINISH OPENING THE JSON FILE ")
    print(len(data_sta))
    df_sta = pd.json_normalize(data_sta[0:len(data_sta)])
    
    print(" NORMALIZE THE JSON FILE ")
    df_sta = df_sta.replace(np.nan, 0, regex=True)

    # Clean the data for stream number and length of the packet
    df_sta['_source.layers.udp.udp.stream']=pd.to_numeric(df_sta['_source.layers.udp.udp.stream'], errors='coerce')
    df_sta['_source.layers.data.data.len']=pd.to_numeric(df_sta['_source.layers.data.data.len'], errors='coerce')
    df_sta['_source.layers.frame.frame.number']=pd.to_numeric(df_sta['_source.layers.frame.frame.number'], errors='coerce')

 
    stream_start_sta=[df_sta.iloc[idx+1]['_source.layers.frame.frame.number'] for idx,e in enumerate(df_sta['_source.layers.data.data.len']) if e==11 and idx<len(df_sta['_source.layers.data.data.len'])-1]
    num_streams=len(stream_start_sta)

    print("READING AP")
    f_ap = open(json_ap)
    data_ap = json.load(f_ap)
    df_ap = pd.json_normalize(data_ap[0:len(data_ap)])
    df_ap = df_ap.replace(np.nan, 0, regex=True)

    df_ap['_source.layers.udp.udp.stream']=pd.to_numeric(df_ap['_source.layers.udp.udp.stream'], errors='coerce')
    df_ap['_source.layers.data.data.len']=pd.to_numeric(df_ap['_source.layers.data.data.len'], errors='coerce')
    df_ap['_source.layers.frame.frame.number']=pd.to_numeric(df_ap['_source.layers.frame.frame.number'], errors='coerce')

    stream_start_ap=[df_ap.iloc[idx+1]['_source.layers.frame.frame.number'] for idx,e in enumerate(df_ap['_source.layers.data.data.len']) if e==11 and idx<len(df_ap['_source.layers.data.data.len'])-1]
    num_streams_ap=len(stream_start_ap)
    print("NUMBER OF STREAMS ACCORDING TO THE AP: ",num_streams_ap)

    print("NUMBER OF STREAMS: ",num_streams)
    losses=np.zeros((num_streams,))
    losses_avg=np.zeros((num_streams,))
    #Read each image and save it
    i=0; deadline_float=float(deadline)

    while i< (min([(num_streams-1),(num_streams_ap-1)])):
        num_packets_ap = stream_start_ap[i+1] - stream_start_ap[i] - 2
        num_packets_stream = stream_start_sta[i+1] - stream_start_sta[i] -2
        print("NUMBER OF PACKETS ACCORDING TO THE AP: ",num_packets_ap)
        print("NUMBER OF PACKETS ACCORDING TO THE STA: ",num_packets_stream)
        #print(num_packets," PACKETS IN IMAGE ", i)
        time_0=float(df_sta[df_sta['_source.layers.frame.frame.number']==stream_start_sta[i]]['_source.layers.frame.frame.time_epoch'].iloc[0])
        time_delta=0; image=""
        #print(df_sta[df_sta['_source.layers.frame.frame.number']==stream_start_sta[i]])
        for jdx in range(num_packets_stream):
            try:
                df_udp=df_sta[df_sta['_source.layers.frame.frame.number']==stream_start_sta[i]+jdx]
                if time_delta < deadline_float:
                    time_delta=float(df_udp['_source.layers.frame.frame.time_epoch'].iloc[0])-time_0
                    #print("TIME DELTA: ", time_delta)

                    if isinstance(df_udp['_source.layers.data.data.data'].iloc[0],str):
                        image=image+df_udp['_source.layers.data.data.data'].iloc[0]

                else:
                    print("TIME DELTA WHEN LOSS: ", time_delta)
                    losses[i] = num_packets_stream -jdx
                    print("Losses of {} is {}".format(i,losses[i]))
                    break
            except:
                pass
                    #print("----------- EXCEPTION -------------- ")
        
        losses[i]=max(losses[i] + (num_packets_ap - num_packets_stream),0)

        losses_avg[i]=losses[i]/num_packets_ap
        
        print(" START WRITING IMAGE ")
        with open(output+'/image_'+str(i)+'.jpg','wb') as f:
            f.write(codecs.decode(image.replace(":",""),'hex'))
        i+=1
    
    with open(plots_folder+'/deadline_'+str(deadline_float*1000)+'ms_lossess.csv','a') as f:
        f.write("image_name,losses, losses_avg"+"\n") 
        for idx,element in enumerate(losses):
            f.write(str(idx)+","+str(element)+","+str(losses_avg[idx])+"\n")           

def info_from_pcaps_ap(folder_ap):
    for folders in os.listdir(folder_ap):
        for file in os.listdir(folder_ap+"/"+folders):
            if file.endswith(".json"):
                print("READING AP")
                f_ap = open(folder_ap+"/"+folders+"/"+file)
                data_ap = json.load(f_ap)
                df_ap = pd.json_normalize(data_ap[0:len(data_ap)])
                df_ap = df_ap.replace(np.nan, 0, regex=True)

                #Get info from AP
                df_ap['_source.layers.data.data.len']=pd.to_numeric(df_ap['_source.layers.data.data.len'], errors='coerce')
                df_ap['_source.layers.frame.frame.number']=pd.to_numeric(df_ap['_source.layers.frame.frame.number'], errors='coerce')

                stream_start_ap=[df_ap.iloc[idx]['_source.layers.frame.frame.number'] for idx,e in enumerate(df_ap['_source.layers.data.data.len']) if e==11]
                num_streams_ap=len(stream_start_ap)
                number_last_stream = stream_start_ap[num_streams_ap-1] -stream_start_ap[num_streams_ap-2] - 1
                print("NUMBER OF STREAMS: ",num_streams_ap)
                print("START OF STREAMS: ",stream_start_ap)
                print("NUMBER OF LAST STREAM",number_last_stream)
                print("LAST STREAM: ",stream_start_ap[num_streams_ap-2])
                #print("MAX: ",max(df_ap['_source.layers.frame.frame.number']))

                stream_start_time=df_ap[df_ap['_source.layers.frame.frame.number']==stream_start_ap[0]]
                print("START TIME: ",stream_start_time['_source.layers.frame.frame.time_epoch'].iloc[0])

                try:
                    stream_end_time=df_ap[df_ap['_source.layers.frame.frame.number']==(stream_start_ap[num_streams_ap-2]+number_last_stream)]
                    print("END TIME: ",stream_end_time['_source.layers.frame.frame.time_epoch'].iloc[0])
                except:
                    max_index=max(df_ap['_source.layers.frame.frame.number'])
                    stream_end_time=df_ap[df_ap['_source.layers.frame.frame.number']==max_index]
                    print("END TIME: ",stream_end_time['_source.layers.frame.frame.time_epoch'].iloc[0])
                    print("EXCEPTION WAS RAISED")
                        
                total_time=float(stream_end_time['_source.layers.frame.frame.time_epoch'].iloc[0])-float(stream_start_time['_source.layers.frame.frame.time_epoch'].iloc[0])
                print("TOTAL TIME: ",total_time)

                with open(folder_ap+'/times_ap.csv','a') as f:
                    f.write(file+","+str(num_streams_ap)+","+str(total_time)+","+str(stream_start_time['_source.layers.frame.frame.time_epoch'].iloc[0])+","+str(stream_end_time['_source.layers.frame.frame.time_epoch'].iloc[0])+"\n")
            
def info_from_pcaps(json_sta, json_ap,plots_folder,image_folder):
    f_sta = open(json_sta)
    data_sta = json.load(f_sta)
    print(" FINISH OPENING THE JSON FILE ")
    print(len(data_sta))
    df_sta = pd.json_normalize(data_sta[0:len(data_sta)])
    print(" NORMALIZE THE JSON FILE ")
    df_sta = df_sta.replace(np.nan, 0, regex=True)

    # Clean the data for stream number and length of the packet
    df_sta['_source.layers.udp.udp.stream']=pd.to_numeric(df_sta['_source.layers.udp.udp.stream'], errors='coerce')
    df_sta['_source.layers.data.data.len']=pd.to_numeric(df_sta['_source.layers.data.data.len'], errors='coerce')
    df_sta['_source.layers.frame.frame.number']=pd.to_numeric(df_sta['_source.layers.frame.frame.number'], errors='coerce')
    #print(df_sta['_source.layers.data.data.len'])
    print("STOP")
    stream_start_sta=[df_sta.iloc[idx]['_source.layers.frame.frame.number']+1 for idx,e in enumerate(df_sta['_source.layers.data.data.len']) if e==11]
    num_streams=len(stream_start_sta)
    print("NUMBER OF STREAMS: ",num_streams)
    losses=np.zeros((num_streams,))
    losses_avg=np.zeros((num_streams,))
    #Read each image and save it
    i=0
    while i< (num_streams-1):
        num_packets=stream_start_sta[i+1] - stream_start_sta[i] -2
        image = ""
        print(num_packets," PACKETS IN IMAGE ", i)
        for jdx in range(num_packets):
            try:
                df_udp=df_sta[df_sta['_source.layers.frame.frame.number']==stream_start_sta[i]+jdx]
                if isinstance(df_udp['_source.layers.data.data.data'].iloc[0],str):
                    image=image+df_udp['_source.layers.data.data.data'].iloc[0]
                else:
                    print("FRAME NUMBER ", stream_start_sta[i]+jdx, "COULDN'T BE ADDED IN THE IMAGE")
                    losses[i]+=1
            except:
                pass
                losses[i]+=1
                print("PASSED STREAM ",i,"FRAME ", stream_start_sta[i]+jdx)
        losses[i]=losses[i]/num_packets
        print(" START WRITING IMAGE ")
        with open(image_folder+'/image_'+str(i)+'.jpg','wb') as f:
            f.write(codecs.decode(image.replace(":",""),'hex'))
        i+=1
    
    print("READING AP")
    f_ap = open(json_ap)
    data_ap = json.load(f_ap)
    df_ap = pd.json_normalize(data_ap[0:len(data_ap)])
    df_ap = df_ap.replace(np.nan, 0, regex=True)

    #Compare AP and STA file
    df_ap['_source.layers.data.data.len']=pd.to_numeric(df_ap['_source.layers.data.data.len'], errors='coerce')
    df_ap['_source.layers.frame.frame.number']=pd.to_numeric(df_ap['_source.layers.frame.frame.number'], errors='coerce')
    #print(df_sta['_source.layers.data.data.len'])
    print("STOP")
    stream_start_ap=[df_ap.iloc[idx]['_source.layers.frame.frame.number']+1 for idx,e in enumerate(df_ap['_source.layers.data.data.len']) if e==11]
    num_streams_ap=len(stream_start_ap)
    print("NUMBER OF STREAMS: ",num_streams_ap)

    for i in range(num_streams_ap-1):
        num_packets_ap = stream_start_ap[i + 1] - stream_start_ap[i] - 2
        sta_idx=0
        for jdx in range(num_packets_ap):
            try:
                df_udp_ap=df_ap[df_ap['_source.layers.frame.frame.number']==stream_start_ap[i]+jdx]
                df_udp_sta = df_sta[df_sta['_source.layers.frame.frame.number'] == stream_start_sta[i] + sta_idx]
                if jdx==0:
                    for k in df_udp_ap.keys():
                        print(k)
                #print(df_udp)
                if isinstance(df_udp['_source.layers.data.data.data'].iloc[0],str):
                   if df_udp_ap['_source.layers.data.data.data'].iloc[0] == df_udp_sta['_source.layers.data.data.data'].iloc[0]:
                    pass
                    sta_idx += 1
                else:
                    print("lost packet ", jdx, " of stream: ", i)
            except:
                pass
                print("PASSED STREAM ",i,"FRAME ", stream_start_ap[i]+jdx)

    
    plt.scatter(range(len(losses)),losses)
    plt.title("Estimated Packets Lost per Image")
    plt.xlabel("Image #")
    plt.ylabel("% Packets lost")
    plt.grid()
    plt.savefig(plots_folder+"prop_delay.pdf")

def conversations_and_deadline(conversations,plots_folder,save_image):
    print("READING CONVERSATIONS")
    f_conv = open(conversations)
    data = pd.read_csv(f_conv)
    data.dropna(subset=['Duration', 'Packets'], how='all', inplace=True)
    plt.clf()
    plt.scatter(range(len(data)), data['Packets'])
    plt.title("Packets per Image")
    plt.xlabel("Image #")
    plt.ylabel("# Packets")
    plt.grid()
    plt.savefig(plots_folder +"/"+ save_image+"_num_packets.pdf")

def info_from_conversations(conversations,plots_folder):
    #Read and plot Transmission delay and datarate
    plt.clf()
    f = open(conversations)
    data = pd.read_csv(f)
    data=data[(data['Duration']>0) & (data['Bits/s A → B']>0)]
    data=data[(data['Duration']<1) & (data['Bits/s A → B']>0)]
    data['Duration']=data['Duration']*1000
    plt.scatter(range(len(data['Duration'])), data['Duration'])
    plt.title("Estimated Transmission Delay per Image")
    plt.xlabel("Image #")
    plt.ylabel("Transmission Delay (ms)")
    plt.grid()
    plt.savefig(plots_folder + "trans_delay.pdf")

    plt.clf()
    plt.hist(data['Duration'])
    plt.title("Estimated Transmission Delay per Image")
    plt.xlabel("Image #")
    plt.ylabel("Transmission Delay (ms)")
    plt.grid()
    plt.savefig(plots_folder + "hist_trans_delay.pdf")

    plt.clf()
    data['Bits/s A → B']= data['Bits/s A → B']/1000000
    plt.scatter(range(len(data)), data['Bits/s A → B'])
    plt.title("Data Rate per Image")
    plt.xlabel("Image #")
    plt.ylabel("Transmission Delay (MB/s)")
    plt.grid()
    plt.savefig(plots_folder + "datarate.pdf")

    plt.clf()
    data['Bytes']= data['Bytes']/1000
    plt.scatter(range(len(data)), data['Bytes'])
    plt.title("Size of Image")
    plt.xlabel("Image #")
    plt.ylabel("Size (KB)")
    plt.grid()
    plt.savefig(plots_folder + "sizes.pdf")

    plt.clf()
    plt.scatter(range(len(data)), data['Packets'])
    plt.title("Packets per Image")
    plt.xlabel("Image #")
    plt.ylabel("# Packets")
    plt.grid()
    plt.savefig(plots_folder + "num_packets.pdf")

    plt.clf()
    plt.scatter(data['Packets'], data['Bits/s A → B'])
    plt.title("Packets per Data Rate")
    plt.xlabel("# Packets")
    plt.ylabel("Data Rate (MB/s)")
    plt.grid()
    plt.savefig(plots_folder + "packs_datarate.pdf")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-json_sta",required=False)
    parser.add_argument("-json_ap",required=False)
    parser.add_argument("-plots",required=False)
    parser.add_argument("-conversations",required=False)
    parser.add_argument("-output",required=False)
    parser.add_argument("-ap_folder",required=False)
    parser.add_argument("-telemetry",required=False)
    parser.add_argument("-config",required=False)
    parser.add_argument("-save_config",required=False)
    parser.add_argument("-deadline",required=False)

    args = parser.parse_args()
    json_sta= '/home/sharon/Documents/Research/Drones-Offloading/data_collected/MU_MIMO_Drone_Offloading_UCI_Dataset/Througput_Latency_PacketLoss/9C/1_STA/sta_9C_1x1_80MHz_packets.json' #'/home/sharon/Documents/Research/Drones-Offloading/data/pcaps/010-sta-packets.json'
    json_ap = '/home/sharon/Documents/Research/Drones-Offloading/data_collected/MU_MIMO_Drone_Offloading_UCI_Dataset/Througput_Latency_PacketLoss/AP/1_STA/ap_1x1_80MHz_packets.json' #'/home/sharon/Documents/Research/Drones-Offloading/data/pcaps/010-ap-packets.json'
    delay_filename='/home/sharon/Documents/Research/Drones-Offloading/results/'
    image_folder='/home/sharon/Documents/Research/Drones-Offloading/output/'
    conversations='/home/sharon/Documents/Research/Drones-Offloading/data/txts/conversations.txt'
    ap_folder='/home/sharon/Documents/Research/Drones-Offloading/data_collected/MU_MIMO_Drone_Offloading_UCI_Dataset/Througput_Latency_PacketLoss/AP/'
    
    #main(args.json_sta,args.json_ap,args.plots,args.output,args.conversations)
    #info_from_pcaps(args.json_sta,args.json_ap,args.plots,args.output)
    #info_from_pcaps_ap(args.ap_folder)
    #info_from_conversations(args.conversations,args.plots)
    #conversations_and_telemetry(args.conversations,args.plots,args.telemetry,args.config,args.save_config)
    deadline_pcaps(args.json_sta,args.plots,args.deadline,args.output,args.json_ap)