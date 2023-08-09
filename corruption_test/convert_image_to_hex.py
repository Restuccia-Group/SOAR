
import base64
import os 
import binascii
import codecs
import time
import random
from PIL import Image
import argparse

"""
split_image():
Takes all the images of type .jpg from the 'original_images' variable, 
It convert those images to hexadecimal files chopped in packets,
It stores one image in packets in a folder within the directory 'hex_images' 
"""
def split_image(original, hex_images):
    for folder in os.listdir(original):
        print(folder)
        for image in os.listdir(original+"/"+folder):
            print(image)
            initial_index=0; packet_size=1024; index=0;
            image_folder=hex_images+"/"+image[:-4]
            os.mkdir(image_folder)
            
            img = Image.open(original+"/"+folder+"/"+image)
            rgb_img = img.convert('RGB')
            rgb_img.save(original+"/"+folder+"/"+image[:-4]+".jpg")

            with open(original+"/"+folder+"/"+image[:-4]+".jpg","rb") as image2string:
                byteString = binascii.hexlify(bytearray(image2string.read()))
                tot_packets=len(byteString)/packet_size
                rem_packets=len(byteString)%packet_size
                print("Total packets: ", tot_packets)

                while index < tot_packets -1 :
                    with open(image_folder+"/encode_packet_"+str(index)+"_image_"+image[:-4]+".bin", "wb") as file:
                        file.write(byteString[initial_index:packet_size+initial_index])
                    initial_index+=packet_size
                    index+=1
                
                if rem_packets >0:
                    with open(image_folder+"/encode_packet_"+str(index)+"_image_"+image[:-4]+".bin", "wb") as file:
                            file.write(byteString[initial_index:initial_index+rem_packets])
            os.remove(original+"/"+folder+"/"+image[:-4]+".jpg")



def main():
    parser = argparse.ArgumentParser(description="Input Arguments")
    parser.add_argument('original_images', help='original image directory')
    parser.add_argument('hex_images', help='hex image directory')

    args = parser.parse_args()

    original_images = args.original_images
    hex_images = args.hex_images
    split_image(original_images, hex_images)


if __name__ == "__main__":
    main()
