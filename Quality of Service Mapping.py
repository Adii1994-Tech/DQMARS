import time

class QoSProfile:
    def __init__(self, identifier, mapping_time):
        self.identifier = identifier
        self.mapping_time = mapping_time

def find_closest_bw(target_bw, bw_list):
    return min(bw_list, key=lambda x: abs(x - target_bw))

def mapping_to_dc(app_id, app_bw):
    start_time = time.time()

    dc_bw = [100, 200, 300, 400, 500]
    DC_GBR = [
        [1, 100],
        [2, 200],
        [3, 300],
        [4, 400],
        [5, 500]
    ]

    qos_bw = find_closest_bw(app_bw, dc_bw)

    for k in range(5):
        if DC_GBR[k][1] == qos_bw:
            mapping_time = time.time() - start_time
            return QoSProfile(DC_GBR[k][0], mapping_time)

    mapping_time = time.time() - start_time
    return QoSProfile(500, mapping_time)

def mapping_to_gbr(app_id, app_pdb, app_per):
    start_time = time.time()

    gbr_pdb = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]

    GBR = [
        [101, 10, 0.0001],
        [102, 20, 0.0002],
        [103, 30, 0.0003],
        [104, 40, 0.0004],
        [105, 50, 0.0005],
        [106, 60, 0.0006],
        [107, 70, 0.0007],
        [108, 80, 0.0008],
        [109, 90, 0.0009],
        [110, 100, 0.001],
        [111, 110, 0.0011],
        [112, 120, 0.0012]
    ]

    qos_pdb = find_closest_bw(app_pdb, gbr_pdb)

    for k in range(12):
        temp = 0
        if GBR[k + temp][2] == qos_pdb and GBR[k + temp][3] >= app_per:
            mapping_time = time.time() - start_time
            return QoSProfile(GBR[k + temp][0], mapping_time)

    mapping_time = time.time() - start_time
    return QoSProfile(400, mapping_time)

def mapping_to_nongbr(app_id, app_per):
    start_time = time.time()

    NonGBR = [
        [201, 0],
        [202, 1],
        [203, 2],
        [204, 3],
        [205, 4],
        [206, 5],
        [207, 6],
        [208, 7],
        [209, 8]
    ]

    if app_per == 0:
        mapping_time = time.time() - start_time
        return QoSProfile(NonGBR[4][0], mapping_time)
    elif app_per == 1:
        mapping_time = time.time() - start_time
        return QoSProfile(NonGBR[6][0], mapping_time)
    elif app_per > 1:
        temp = 0
        for k in range(8):
            if app_per <= NonGBR[k + temp][2]:
                mapping_time = time.time() - start_time
                return QoSProfile(NonGBR[k + temp][0], mapping_time)
            temp += 1
            if NonGBR[k + temp][2] in [2, 3]:
                temp += 1
            if temp > 8:
                temp = 0
                break
        mapping_time = time.time() - start_time
        return QoSProfile(NonGBR[temp][0], mapping_time) if temp <= 8 else QoSProfile(0, mapping_time)

# Main algorithm
def main_algorithm(app_id, app_bw, app_pdb, app_per, app_deadline, app_jitter):
    if app_bw > 0 and app_deadline and app_jitter:
        qos_profile = mapping_to_dc(app_id, app_bw)
        print("QoS Profile for DelayCriticalGBR:", qos_profile.identifier, "Mapping Time:", qos_profile.mapping_time)
    elif app_bw == 0 and app_deadline and app_jitter:
        qos_profile = mapping_to_gbr(app_id, app_pdb, app_per)
        print("QoS Profile for GBR:", qos_profile.identifier, "Mapping Time:", qos_profile.mapping_time)
    else:
        qos_profile = mapping_to_nongbr(app_id, app_per)
        print("QoS Profile for NonGBR:", qos_profile.identifier, "Mapping Time:", qos_profile.mapping_time)

# Example usage with applications having different requirements
app_1 = {'id': 1, 'bw': 250, 'pdb': 35, 'per': 0.0005, 'deadline': True, 'jitter': True}
app_2 = {'id': 2, 'bw': 0, 'pdb': 50, 'per': 0.0005, 'deadline': True, 'jitter': True}
app_3 = {'id': 3, 'bw': 0, 'pdb': 0, 'per': 0, 'deadline': False, 'jitter': False}

main_algorithm(app_1['id'], app_1['bw'], app_1['pdb'], app_1['per'], app_1['deadline'], app_1['jitter'])
main_algorithm(app_2['id'], app_2['bw'], app_2['pdb'], app_2['per'], app_2['deadline'], app_2['jitter'])
main_algorithm(app_3['id'], app_3['bw'], app_3['pdb'], app_3['per'], app_3['deadline'], app_3['jitter'])

