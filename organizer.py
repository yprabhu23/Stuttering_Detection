import os
import pathlib
import shutil

import pandas as pd

DATASET_PATH = '/media/yashprabhu/Extreme SSD/Sep28k/clips'#/home/yashprabhu/Documents/Sep28k/clips/HVSA/
csvreader = pd.read_csv('/media/yashprabhu/Extreme SSD/Sep28k/SEP-28k_labels.csv')
destination1 = '/media/yashprabhu/Extreme SSD/SortedBy'
destination2 =  '/media/yashprabhu/Extreme SSD/SortedBy'
prolonged = []
nonProlonged = []
arrayThing = ['Block', 'SoundRep','WordRep','Interjection','NoStutteredWords']
for thing in arrayThing:
    destination1 = '/media/yashprabhu/Extreme SSD/SortedBy'
    destination2 = '/media/yashprabhu/Extreme SSD/SortedBy'
    for i in range (0,28176):
        if i<7629 or i>9939:
            destination1 = '/media/yashprabhu/Extreme SSD/SortedBy'
            destination2 = '/media/yashprabhu/Extreme SSD/SortedBy'
            destination1+= thing +'/'+thing
            destination2+= thing +'/No' +thing
            show = csvreader["Show"][i]
            clipID = csvreader["ClipId"][i]
            epID = csvreader["EpId"][i]

            if csvreader[thing][i] >0:
                origin = DATASET_PATH+"/"+ show +"/" + str(epID) + "/"+ show+"_" + str(epID) + "_" + str(clipID) + ".wav"
                print(origin)
                shutil.copy(origin,
                            destination1)
            else:
                origin = DATASET_PATH+"/"+ show +"/" + str(epID) + "/"+ show+"_" + str(epID) + "_" + str(clipID) + ".wav"
                print(origin)
                shutil.copy(origin,
                            destination2)


