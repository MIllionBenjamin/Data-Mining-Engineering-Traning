import pandas as pd
from sklearn import preprocessing
import numpy as np
from scipy import stats

# 读取各个csv文件中的数据
train_data = pd.read_csv("./train/entprise_info.csv")
base_info = pd.read_csv("./train/base_info.csv")
annual_report = pd.read_csv("./train/annual_report_info.csv")
tax_info = pd.read_csv("./train/tax_info.csv")
change_info = pd.read_csv("./train/change_info.csv")
# 获得所有公司ID
ALL_ID = set(base_info["id"].to_list())

# base_info处理，筛选有意义的列，类别数据用label encode编码，并进行缺失值填补
# 阅读数据说明得到有意义的列。排除掉地址等说明列；排除掉缺失值过多且无法合理填补的列。
new_base_info = base_info.copy()
meaningful_column = ["id", 
                     "oplocdistrict", 
                     "industryphy", 
                     "industryco", 
                     "enttype", 
                     "state", 
                     "adbusign", 
                     "townsign", 
                     "regtype", 
                     "empnum", 
                     "compform", 
                     "parnum", 
                     "exenum", 
                     "venind", 
                     "regcap", 
                     "enttypegb"]
# oplocdistrict 用label-encoder编码
######
# 注意：树模型适用label-encoder编码，见https://zhuanlan.zhihu.com/p/36804348
######
le = preprocessing.LabelEncoder()
new_base_info["oplocdistrict"] = le.fit_transform(new_base_info["oplocdistrict"])
# industryphy 用label-encoder编码
le = preprocessing.LabelEncoder()
new_base_info["industryphy"] = le.fit_transform(new_base_info["industryphy"])
# industryco 用label-encoder编码
le = preprocessing.LabelEncoder()
new_base_info["industryco"] = le.fit_transform(new_base_info["industryco"])
# enttype 用label-encoder编码
le = preprocessing.LabelEncoder()
new_base_info["enttype"] = le.fit_transform(new_base_info["enttype"])
# state无需处理
# adbusign无需处理
# townsign无需处理
# regtype无需处理
# empnum用平均数补全缺失值
new_base_info["empnum"].fillna(np.mean(new_base_info["empnum"]), inplace=True)
# compform用0补全缺失值
new_base_info["compform"].fillna(0, inplace=True)
# parnum用0补全缺失值
new_base_info["parnum"].fillna(0, inplace=True)
# exenum用0补全缺失值
new_base_info["exenum"].fillna(0, inplace=True)
# venind用0补全缺失值
new_base_info["venind"].fillna(0, inplace=True)
# regcap用众数补全缺失值
print(stats.mode(new_base_info["regcap"])[0][0])
new_base_info["regcap"].fillna(stats.mode(new_base_info["regcap"])[0][0], inplace=True)
# enttypegb 用label-encoder编码
le = preprocessing.LabelEncoder()
new_base_info["enttypegb"] = le.fit_transform(new_base_info["enttypegb"])

#查看是否还有缺失值
print(new_base_info[meaningful_column].isnull().sum())
#将新base_info数据写入文件
new_base_info[meaningful_column].to_csv("./train/base_info_new.csv", index = False)


#print(pd.merge(train_data, base_info, on = ["id"]))


# annual_report_info处理，只保留最近年份的数据条目
######
# 注意：处理后发现总共24865个公司，只有8937家公司有年报数据。而年报数据难以可靠填充，故训练时不使用annual_report_info数据。
######
# 取annual_report_info表头
annual_report_head = list(annual_report.columns)
# print(annual_report_head)
# 对于每个id，取最近年份的条目放入annual_report_only_recent中
annual_report_only_recent = {}
for index, annual_report_row in annual_report.iterrows():
    #print(annual_report_row)
    if annual_report_row["id"] not in annual_report_only_recent:
        annual_report_only_recent[annual_report_row["id"]] = annual_report_row
    elif annual_report_only_recent[annual_report_row["id"]]["ANCHEYEAR"] < annual_report_row["ANCHEYEAR"]:
        annual_report_only_recent[annual_report_row["id"]] = annual_report_row
    else:
        continue
# 用annual_report_only_recent中的数据生成新的dataframe
only_recent_df_dic = {}
for item in annual_report_head:
    only_recent_df_dic[item] = []
for id_key in annual_report_only_recent:
    for item_key in only_recent_df_dic:
        only_recent_df_dic[item_key].append(annual_report_only_recent[id_key][item_key])
only_recent_df = pd.DataFrame(only_recent_df_dic)
# 储存只保留最近年份的数据到文件
only_recent_df.to_csv("./train/annual_report_info_only_recent.csv", index=False)


# tax_info数据处理，对每个公司，计算 TAX_AMOUNT税额和 与 TAXATION_BASIS计税依据和 的比率
# 处理后数据的表头
tax_basis_amount_sum_head = ["id", "AMOUNT_TO_BASIS"]
id_tax_basis_amount_sum = {}
for index, tax_info_row in tax_info.iterrows():
    # TAXATION_BASIS若为空，用TAX_AMOUNT填补
    tax_basis = tax_info_row["TAX_AMOUNT"] if pd.isnull(tax_info_row["TAXATION_BASIS"]) else tax_info_row["TAXATION_BASIS"]
    if tax_info_row["id"] not in id_tax_basis_amount_sum:
        id_tax_basis_amount_sum[tax_info_row["id"]] = {"TAXATION_BASIS_SUM": tax_basis, "TAX_AMOUNT_SUM": tax_info_row["TAX_AMOUNT"]}
    else:
        id_tax_basis_amount_sum[tax_info_row["id"]]["TAXATION_BASIS_SUM"] += tax_basis
        id_tax_basis_amount_sum[tax_info_row["id"]]["TAX_AMOUNT_SUM"] += tax_info_row["TAX_AMOUNT"]
# 用id_tax_basis_amount_sum中的数据生成新的dataframe
tax_basis_amount_sum_df_dic = {}
for item in tax_basis_amount_sum_head:
    tax_basis_amount_sum_df_dic[item] = []
for id_key in id_tax_basis_amount_sum:
    tax_basis_amount_sum_df_dic["id"].append(id_key)
    if id_tax_basis_amount_sum[id_key]["TAXATION_BASIS_SUM"] != 0:
        tax_basis_amount_sum_df_dic["AMOUNT_TO_BASIS"].append(id_tax_basis_amount_sum[id_key]["TAX_AMOUNT_SUM"] / id_tax_basis_amount_sum[id_key]["TAXATION_BASIS_SUM"])
    else:
        tax_basis_amount_sum_df_dic["AMOUNT_TO_BASIS"].append(0)
# 若公司在tax_info中没有记录，设其 TAX_AMOUNT税额和 与 TAXATION_BASIS计税依据和 的比率 为0
for no_tax_info_id in ALL_ID - set(tax_basis_amount_sum_df_dic["id"]):
    tax_basis_amount_sum_df_dic["id"].append(no_tax_info_id)
    tax_basis_amount_sum_df_dic["AMOUNT_TO_BASIS"].append(0)
tax_basis_amount_sum_df = pd.DataFrame(tax_basis_amount_sum_df_dic)
# 储存 TAXATION_BASIS计税依据和 与 TAX_AMOUNT税额和 的数据到文件
tax_basis_amount_sum_df.to_csv("./train/tax_info_basis_amount_sum.csv", index=False)


# change_info处理，统计每个公司对应的change数量
# 处理后数据的表头
change_info_amount_head = ["id", "CHANGE_AMOUNT"]
id_change_amount = {}
for index, change_info_row in change_info.iterrows():
    if change_info_row["id"] not in id_change_amount:
        id_change_amount[change_info_row["id"]] = 1
    else:
        id_change_amount[change_info_row["id"]] += 1
# 用id_change_amount中的数据生成新的dataframe
change_info_amount_df_dic = {}
for item in change_info_amount_head:
    change_info_amount_df_dic[item] = []
for id_key in id_change_amount:
    change_info_amount_df_dic["id"].append(id_key)
    change_info_amount_df_dic["CHANGE_AMOUNT"].append(id_change_amount[id_key])
# 若change_info中没有某公司的记录，则该公司change数量为0
for no_change_info_id in ALL_ID - set(change_info_amount_df_dic["id"]):
    change_info_amount_df_dic["id"].append(no_change_info_id)
    change_info_amount_df_dic["CHANGE_AMOUNT"].append(0)
change_info_amount_df = pd.DataFrame(change_info_amount_df_dic)
# 储存change次数数据到文件
change_info_amount_df.to_csv("./train/change_info_change_amount.csv", index=False)







    

