import pandas as pd
import matplotlib.pyplot as plt

def plot_times(user_file, kernel_file):
    user_data = pd.read_csv(user_file, header=None, names=["Input Size", "Ctime", "Ttime"])
    kernel_data = pd.read_csv(kernel_file, header=None, names=["Input Size", "Ctime", "Ttime"])

    user_data["Ctime"] = pd.to_numeric(user_data["Ctime"], errors='coerce')
    user_data["Ttime"] = pd.to_numeric(user_data["Ttime"], errors='coerce')
    
    kernel_data["Ctime"] = pd.to_numeric(kernel_data["Ctime"], errors='coerce')
    kernel_data["Ttime"] = pd.to_numeric(kernel_data["Ttime"], errors='coerce')

    input_size_user = user_data["Input Size"].to_numpy()
    ctime_user = user_data["Ctime"].to_numpy()
    ttime_user = user_data["Ttime"].to_numpy()

    input_size_kernel = kernel_data["Input Size"].to_numpy()
    ctime_kernel = kernel_data["Ctime"].to_numpy()
    ttime_kernel = kernel_data["Ttime"].to_numpy()

    plt.figure(figsize=(10, 6))

    plt.plot(input_size_user, ctime_user, label="User Space Ctime", marker='o')
    plt.plot(input_size_user, ttime_user, label="User Space Ttime", marker='x')

    plt.plot(input_size_kernel, ctime_kernel, label="Kernel Space Ctime", marker='o', linestyle='--')
    plt.plot(input_size_kernel, ttime_kernel, label="Kernel Space Ttime", marker='x', linestyle='--')

    plt.xlabel('Input Size')
    plt.ylabel('Time (ms)')
    plt.title('Execution Time: User Space vs Kernel Space')
    plt.legend()

    plt.yscale('log')

    plt.grid(True)
    plt.tight_layout()
    plt.savefig('execution_time_plot.pdf')

plot_times('file_uspace.csv', 'file_kspace.csv')
