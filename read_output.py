import numpy as np
import os

def save_iq(input_file, output_file, trim_or_not=True, dtype=np.complex64):
    # 读取数据
    x = np.fromfile(input_file, dtype=dtype)
    y = np.fromfile(output_file, dtype=dtype)

    # 获取原路径和文件名（不含扩展名）
    input_dir, input_name = os.path.split(input_file)
    output_dir, output_name = os.path.split(output_file)

    input_basename = os.path.splitext(input_name)[0]
    output_basename = os.path.splitext(output_name)[0]

    # 构造输出路径（同目录、换扩展名为 .csv）
    input_csv_path = os.path.join(input_dir, input_basename + '.csv')
    output_csv_path = os.path.join(output_dir, output_basename + '.csv')

    if trim_or_not:
        x_trimmed = x[100:]
        y_trimmed = y[100:]
        np.savetxt(input_csv_path, np.c_[x_trimmed.real, x_trimmed.imag],
                   delimiter=',', fmt='%f', header='I,Q', comments='')
        np.savetxt(output_csv_path, np.c_[y_trimmed.real, y_trimmed.imag],
                   delimiter=',', fmt='%f', header='I,Q', comments='')
    else:
        np.savetxt(input_csv_path, np.c_[x.real, x.imag],
                   delimiter=',', fmt='%f', header='I,Q', comments='')
        np.savetxt(output_csv_path, np.c_[y.real, y.imag],
                   delimiter=',', fmt='%f', header='I,Q', comments='')

    print(f"Saved input CSV to: {input_csv_path}")
    print(f"Saved output CSV to: {output_csv_path}")

# 示例调用
x_path = './GMP_PA/GMP_PA_input'
y_path = './GMP_PA/GMP_PA_output'
# save_iq(x_path, y_path, trim_or_not=True, dtype=np.complex64)
source_path = 'source_output'
f = np.fromfile(open(source_path), dtype=np.int8)
print(f)
np.savetxt('source_output.txt', f, fmt='%.10f')
