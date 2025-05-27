import os
import codecs
import argparse


def create_folder(directory, folder_name):
    folder_path = os.path.join(directory, folder_name)
    os.makedirs(folder_path, exist_ok=True)


def create_loss_images(loss_fraction, hex_images, loss_image_dir):
    for image_folder in os.listdir(hex_images):
        packet_bytes = b''
        tot_packets = max([int(packet.split("_")[2]) for packet in os.listdir(hex_images+"/"+image_folder)])
        list_packets = [i for i in range(tot_packets)]

        print("Total packets:", tot_packets)
        num_packets = int(tot_packets*loss_fraction)
        to_remove = list_packets[len(list_packets)-num_packets:]

        for packet_num in list_packets:
            if packet_num in to_remove:
                list_packets.remove(packet_num) 

        for i in list_packets:
            for packet in os.listdir(hex_images+"/"+image_folder):
                if int(packet.split("_")[2]) == i:
                    with open(hex_images+"/"+image_folder+"/"+packet, "rb") as image2string:

                        packet_bytes += image2string.read()
                elif int(packet.split("_")[2]) in to_remove:
                    continue

        create_folder(loss_image_dir, str(int(loss_fraction*100)))

        with open(loss_image_dir + "/" + str(int(loss_fraction*100)) + '/' + image_folder+".jpg", 'wb') as file:
            file.write(codecs.decode(packet_bytes, 'hex'))


def main():
    parser = argparse.ArgumentParser(description="Input Arguments")
    parser.add_argument('loss_fraction', help='loss fraction of the images')
    parser.add_argument('hex_images', help='hex images directory')
    parser.add_argument('loss_image_dir', help='loss images directory')

    args = parser.parse_args()

    loss_fraction = float(args.loss_fraction)
    hex_images = args.hex_images
    loss_image_dir = args.loss_image_dir

    create_loss_images(loss_fraction, hex_images, loss_image_dir)


if __name__ == "__main__":
    main()