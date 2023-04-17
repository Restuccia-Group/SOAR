import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
import codecs
import argparse

def main(json_sta, json_ap,plots_folder,image_folder,conversations):
    f_ap = open(json_ap)
    data_ap = json.load(f_ap)
    df_ap = pd.json_normalize(data_ap[0:len(data_ap)])
    df_ap = df_ap.replace(np.nan, 0, regex=True)

    f_sta = open(json_sta)
    data_sta = json.load(f_sta)

    # Clean the data for stream number and length of the packet
    df_sta = pd.json_normalize(data_sta[0:len(data_sta)])
    df_sta = df_sta.replace(np.nan, 0, regex=True)
    df_tcp = df_sta
    df_tcp['_source.layers.tcp.tcp.analysis.tcp.analysis.ack_rtt']=pd.to_numeric(df_tcp['_source.layers.tcp.tcp.analysis.tcp.analysis.ack_rtt'], errors='coerce')
    delay=[(element[1]*1000)/2 for element in df_tcp[df_tcp['_source.layers.tcp.tcp.analysis.tcp.analysis.ack_rtt']>0]['_source.layers.tcp.tcp.analysis.tcp.analysis.ack_rtt'].items() if ((element[1]*1000)/2)<500]
    df_sta['_source.layers.udp.udp.stream']=pd.to_numeric(df_sta['_source.layers.udp.udp.stream'], errors='coerce')
    num_streams=max(df_sta['_source.layers.udp.udp.stream'])
    for i in range(num_streams):
        df_udp=df_sta[df_sta['_source.layers.udp.udp.stream']== i]
        try:
            image=df_udp['_source.layers.udp.udp.payload'].iloc[0]
        except:
            pass
            print("PASS ",i)
        skip_first=True
        for element in df_udp.iterrows():
            if skip_first==True:
                skip_first=False
            else:
                if not isinstance(element[1]['_source.layers.udp.udp.payload'],int):
                    image=image+element[1]['_source.layers.udp.udp.payload']
        with open(image_folder+'image_'+str(i)+'.jpg','wb') as f:
            f.write(base64.b64decode(codecs.decode(image.replace(":",""),'hex_codec').decode("ascii")))

    #Compare AP and STA file
    df_ap['_source.layers.udp.udp.stream'] = pd.to_numeric(df_ap['_source.layers.udp.udp.stream'], errors='coerce')
    num_streams_ap = max(df_ap['_source.layers.udp.udp.stream'])
    for i in range(num_streams_ap):
        df_udp_ap = df_ap[df_ap['_source.layers.udp.udp.stream'] == i]
        df_udp_sta = df_sta[df_sta['_source.layers.udp.udp.stream'] == i]
        skip_first = True
        for jdx,element in enumerate(df_udp_ap.iterrows()):
            if skip_first == True:
                skip_first = False
            else:
                if not isinstance(element[1]['_source.layers.udp.udp.payload'], int):
                    if (element[1]['_source.layers.udp.udp.payload'] == df_udp_sta['_source.layers.udp.udp.payload'].iloc[jdx]):
                        pass
                    else:
                        print("lost packet ", jdx, " of stream: ", i)

    print("# of images that had a delay < 500 ms: ", len(delay))
    print("Average delay per 100 images: ", sum(delay)/len(delay))
    plt.scatter(range(len(delay)),delay)
    plt.title("Estimated Propagation Delay per Image")
    plt.xlabel("Image #")
    plt.ylabel("Propagation Delay (ms)")
    plt.grid()
    plt.savefig(plots_folder+"prop_delay.pdf")

    #Read and plot Transmission delay and datarate
    plt.clf()
    f = open(conversations)
    data = pd.read_csv(f)
    data['Duration']=data['Duration']*1000
    plt.scatter(range(len(data)), data['Duration'])
    plt.title("Estimated Transmission Delay per Image")
    plt.xlabel("Image #")
    plt.ylabel("Transmission Delay (ms)")
    plt.grid()
    plt.savefig(plots_folder + "trans_delay.pdf")

    plt.clf()
    #f = open(conversations)
    #data = pd.read_csv(f)
    data['Bits/s A → B']= data['Bits/s A → B']/1000000
    plt.scatter(range(len(data)), data['Bits/s A → B'])
    plt.title("Data Rate per Image")
    plt.xlabel("Image #")
    plt.ylabel("Transmission Delay (MB/s)")
    plt.grid()
    plt.savefig(plots_folder + "datarate.pdf")

    plt.clf()
    #f = open(conversations)
    #data = pd.read_csv(f)
    data['Bytes']= data['Bytes']/1000
    plt.scatter(range(len(data)), data['Bytes'])
    plt.title("Size of Image")
    plt.xlabel("Image #")
    plt.ylabel("Size (KB)")
    plt.grid()
    plt.savefig(plots_folder + "sizes.pdf")

    plt.clf()
    #f = open(conversations)
    #data = pd.read_csv(f)
    plt.scatter(range(len(data)), data['Packets'])
    plt.title("Packets per Image")
    plt.xlabel("Image #")
    plt.ylabel("# Packets")
    plt.grid()
    plt.savefig(plots_folder + "num_packets.pdf")

    plt.clf()
    #f = open(conversations)
    #data = pd.read_csv(f)
    plt.scatter(data['Packets'], data['Bits/s A → B'])
    plt.title("Packets per Data Rate")
    plt.xlabel("# Packets")
    plt.ylabel("Data Rate (MB/s)")
    plt.grid()
    plt.savefig(plots_folder + "packs_datarate.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-json_sta")
    parser.add_argument("-json_ap")
    parser.add_argument("-plots")
    parser.add_argument("-conversations")
    parser.add_argument("-output")

    args = parser.parse_args()
    json_sta='/home/sharon/Documents/Research/Drones-Offloading/data/pcaps/010-sta-packets.json'
    json_ap = '/home/sharon/Documents/Research/Drones-Offloading/data/pcaps/010-ap-packets.json'
    delay_filename='/home/sharon/Documents/Research/Drones-Offloading/results/'
    image_folder='/home/sharon/Documents/Research/Drones-Offloading/output/'
    conversations = '/home/sharon/Documents/Research/Drones-Offloading/data/txts/conversations.txt'
    main(args.json_sta,args.json_ap,args.plots,args.output,args.conversations)
