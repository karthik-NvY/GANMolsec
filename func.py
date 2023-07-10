# For entropy calculation
def shannon(data):
    LOG_BASE = 2
   # We determine the frequency of each byte
   # in the dataset and if this frequency is not null we use it for the
   # entropy calculation
    dataSize = len(data)
    ent = 0.0
    freq={} 
    for c in data:
        if c in freq:
            freq[c] += 1
        else:
            freq[c] = 1
   # to determine if each possible value of a byte is in the list
    for key in freq.keys():
        f = float(freq[key])/dataSize
        if f > 0: # to avoid an error for log(0)
            ent = ent + f * math.log(f, LOG_BASE)
    return -ent

def pre_entropy(payload):
    
    characters=[]
    for i in payload:
            characters.append(i)
    return shannon(characters)


# For classifying ports. Total of 14 classes.

def port_class(port):
    port_list=[0,53,67,68,80,123,443,1900,5353,49153]
    if port in port_list:
        return port_list.index(port)+1
    elif 0 <= port <= 1023:
        return 11
    elif  1024 <= port <= 49151 :
        return 12
    elif 49152 <=port <= 65535 :
        return 13
    else:
        return 0
    
    
def port_1023(port):
    if 0 <= port <= 1023:
        return port
    elif  1024 <= port <= 49151 :
        return 2
    elif 49152 <=port <= 65535 :
        return 3
    else:
        return 0
