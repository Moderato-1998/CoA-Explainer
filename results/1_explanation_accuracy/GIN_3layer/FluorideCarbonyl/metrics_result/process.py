import numpy as np
import os

# 获取脚本当前目录
script_dir = os.path.dirname(__file__)
exp_name = ["gnnex", "pgex", "subx", "rcex", "coaex"]  # gnnex, pg, subx, our
# 文件名
edge_gea = [os.path.join(script_dir, f"{f}_gea.npy") for f in exp_name]
edge_gef = [os.path.join(script_dir, f"{f}_gef.npy") for f in exp_name]
pos_fid = [os.path.join(script_dir, f"{f}_fid+.npy") for f in exp_name]
neg_fid = [os.path.join(script_dir, f"{f}_fid-.npy") for f in exp_name]
time = [os.path.join(script_dir, f"{f}_time.npy") for f in exp_name]


def format_output(file_list):
    results = []
    for f in file_list:
        data = np.load(f)
        mean = np.mean(data)
        std_err = np.std(data) / np.sqrt(len(data))
        results.append(f"{mean:.3f} ± {std_err:.3f}")
    return results


def to_ms(file_list):
    results = []
    for f in file_list:
        data = np.load(f)
        data = np.array(data) * 1000  # convert to ms
        mean = np.mean(data)
        std_err = np.std(data) / np.sqrt(len(data))
        results.append(f"{mean:.3f} ± {std_err:.3f}")
    return results


edge_gea_results = format_output(edge_gea)
edge_gef_results = format_output(edge_gef)
pos_fid_results = format_output(pos_fid)
neg_fid_results = format_output(neg_fid)
time_results = to_ms(time)

print(f"edge_gea: {edge_gea_results}")
print(f"edge_gef: {edge_gef_results}")
print(f"fid+: {pos_fid_results}")
print(f"fid-: {neg_fid_results}")
print(f"time: {time_results}")
