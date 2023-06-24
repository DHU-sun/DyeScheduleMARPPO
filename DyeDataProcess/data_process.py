import json
import time

import numpy as np
import pandas as pd


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def dict_to_json(file_name, dict):
    # 转换成json
    j = json.dumps(dict, cls=NpEncoder)
    # 写到json文件
    fileObject = open(file_name, 'w')
    fileObject.write(j)
    fileObject.close()


time_start = time.time()

dye_job_dict = {}

path = '染缸调度数据/'

Dyelots_file = 'Dyelots.csv'
V_jh_dye_Project_file = 'V_jh_dye_Project.csv'
yx_ColorOrderDetail = 'yx_ColorOrderDetail.csv'
da_MachineNo = 'da_MachineNo.csv'
da_MachineType = 'da_MachineType.csv'

# 读取csv文件
data_dic = {}
data_dic[Dyelots_file] = pd.read_csv(path + Dyelots_file, header=0)
data_dic[V_jh_dye_Project_file] = pd.read_csv(path + V_jh_dye_Project_file, header=0)
data_dic[yx_ColorOrderDetail] = pd.read_csv(path + yx_ColorOrderDetail, header=0)
data_dic[da_MachineNo] = pd.read_csv(path + da_MachineNo, header=0)
data_dic[da_MachineType] = pd.read_csv(path + da_MachineType, header=0)

# print(data_dic[Dyelots_file])


# 从Dyelots表抓取某些数据
dye_Project_columnNames = ['BillDate', 'ColorID', 'ColorCode', 'ColorName', 'ColorClass', 'dyeKG',
                           'dyeCnt', 'RequestCnt', 'RequestKG', 'FixTime', 'dyeTime', 'ProduceNo']

for col_name in dye_Project_columnNames:
    dye_job_dict[col_name] = []

ColorOrder_columnNames = ['RequestTime']
for col_name in ColorOrder_columnNames:
    dye_job_dict[col_name] = []

exist_columnNames = ['dyelot', 'batch', 'standardtime', 'redye', 'totalweight']
for col_name in exist_columnNames:
    dye_job_dict[col_name] = []

# 查找属性并赋值

index_record = []
j = 0
for index, row in data_dic[Dyelots_file].iterrows():
    if index < 13818:
        pass
    else:
        row_index = data_dic[V_jh_dye_Project_file]['ProjectNo'] == str(row['dyelot'])
        if sum(row_index) > 0:
            row_index = list(row_index).index(True)
            index_record.append(row_index)

            for col_name in exist_columnNames:
                dye_job_dict[col_name].append(row[col_name])

            for col_name in dye_Project_columnNames:
                dye_job_dict[col_name].append(data_dic[V_jh_dye_Project_file].loc[row_index, col_name])

            # 查yx_ColorOrderDetail表
            row_index = data_dic[yx_ColorOrderDetail]['ColorBillNO'] == data_dic[V_jh_dye_Project_file].loc[
                row_index, 'ProduceNo']
            if sum(row_index) > 0:
                row_index = list(row_index).index(True)
                for col_name in ColorOrder_columnNames:
                    dye_job_dict[col_name].append(data_dic[yx_ColorOrderDetail].loc[row_index, col_name])
                # print(index)
                j += 1
                if j > 5000:
                    break
            else:
                for col_name in ColorOrder_columnNames:
                    dye_job_dict[col_name].append(None)


# 把能查找到的数据另存出来
print(dye_job_dict)
dict_to_json(file_name=path + 'job_file.json', dict=dye_job_dict)

time_end = time.time()
print('totally job num: %s' % (len(index_record)))
print('totally time cost %s s' % (time_end - time_start))

# 整理设备数据

machine_dict = {}
machine_dict['ID'] = []
machine_dict['KindID'] = []

machine_columnNames = ['MachineKG']
for col_name in machine_columnNames:
    machine_dict[col_name] = []

dye_mach_type_list = [1005001001,
                      1005001002,
                      1005001003,
                      1005001004,
                      1005001005,
                      1005001006,
                      1005001007,
                      1005001008,
                      1005001009,
                      1005001010,
                      1005001011,
                      1005001012,
                      1005001013,
                      1005001014,
                      1005001015,
                      1005001016,
                      1005001017,
                      1005001018,
                      ]

index_record = []
j = 0
for index, row in data_dic[da_MachineNo].iterrows():
    if row['KindID'] in dye_mach_type_list:
        j += 1
        machine_dict['ID'].append(row['ID'])
        machine_dict['KindID'].append(row['KindID'])
        row_index = data_dic[da_MachineType]['ID'] == row['KindID']
        row_index = list(row_index).index(True)
        index_record.append(index)
        for col_name in machine_columnNames:
            machine_dict[col_name].append(data_dic[da_MachineType].loc[row_index, col_name])


print(machine_dict)
dict_to_json(file_name=path + 'machine_file.json', dict=machine_dict)

time_end = time.time()
print('totally machine num: %s' % (len(index_record)))
print('totally time cost %s s' % (time_end - time_start))

# 读文件然后创建对象

