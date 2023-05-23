import argparse
import pandas as pd
import os
import pickle
import subprocess

class ExpsReader:
    def __init__(self, path, filename):
        self.path = path
        self.filename = filename

    def get_udpstats(self):
        df=pd.read_csv(os.path.join(self.path,self.filename))
        arr=df.to_numpy()
        for idx,e in enumerate(arr):
            if e[0][0]=='-':
                break
        idx= idx+3
        drate=arr[idx][0].split()[6] # Datarate - Mbps
        jitt=arr[idx][0].split()[8] # Jitter - ms
        ratioLost=arr[idx][0].split()[10].split("/") # Lost/Total Datagrams
        lost=int(ratioLost[0]) / int(ratioLost[1])
        err_dict ={'data_rate':drate,'jitter':jitt,'datagram_lost':lost}
        with open(os.path.join(self.path,self.filename)[:-4]+'.pickle', 'wb') as handle:
            pickle.dump(err_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        os.remove(os.path.join(self.path,self.filename))

    def connect_ssh(self,user,host,cmd):
        print("connect ssh ", user, "host ", host, "command ", cmd)
        sshprocess = subprocess.Popen(["ssh","{}@{}".format(user,host)],stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         )
        out, err = sshprocess.communicate(cmd.encode())
        print(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read files to save stats')
    parser.add_argument("-p", "--path")
    parser.add_argument("-f", "--filename")
    parser.add_argument("-lip", "--lipadd")
    parser.add_argument("-rip","--ripadd")
    args=parser.parse_args()

    reader=ExpsReader(args.path,args.filename+'.txt')
    print("passed reader")

    reader.connect_ssh('root', args.lipadd,
                       'cd /mnt/sda1/stats ; iperf3 -c '+
                       args.ripadd+' -u -b '+args.filename+' -p 5201 -t 10  -i 2 > '+args.filename+'.txt \n')
    reader.get_udpstats()