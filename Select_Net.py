import os

def select_net(group_id,channel_n):
    group_List = ['Ki1', 'Ki2', 'Ki3', 'Ki4', 'Ki5']
    if group_id == group_List[0]:
        from Net_List.Ki1_510_53 import ConvNet_2
        model = ConvNet_2(channel_n)

    elif group_id == group_List[1]:
        from Net_List.Ki2_284_99 import ConvNet_2
        model = ConvNet_2(channel_n)

    elif group_id == group_List[2]:
        from Net_List.Ki3_512_80 import ConvNet_2
        model = ConvNet_2(channel_n)
    
    elif group_id == group_List[2]:
        from Net_List.Ki4_356_56 import ConvNet_2
        model = ConvNet_2(channel_n)

    elif group_id == group_List[2]:
        from Net_List.Ki5_280_44 import ConvNet_2
        model = ConvNet_2(channel_n)

    #print(f" Have selected Net for {group_id}")

    return model

if __name__ == "__main__":
    select_net('Ki1',5)




            