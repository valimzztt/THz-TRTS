import os
import matplotlib.pyplot as plt

def process_folders(folder1, folder2, folder3, output_folder):
    # Function to process files in each folder

    def get_files(folder):
        # Get all files in the folder ending with 'Ave.d24' and sort them
        files = [f for f in os.listdir(folder) if f.endswith('Ave.d24')]
        files.sort()  # Ensure the files are in order of pump delay
        return files

    # Function to read and process data from a file
    def read_and_extract_columns(file_path):
        extracted_data = []
        with open(file_path, 'r') as file:
            for line in file:
                if line.strip():  # Skip empty lines
                    columns = line.split()
                    # Extract the first, second, and third columns
                    extracted_data.append((float(columns[0]), float(columns[1]), float(columns[2])))
        return extracted_data

    # Function to average three datasets
    def average_datasets(data1, data2, data3):
        if len(data1) != len(data2) or len(data1) != len(data3):
            print("Warning: Datasets have different lengths.")
            return []
        return [(x1, (y1 + y2 + y3) / 3, (z1 + z2 + z3) / 3) 
                for (x1, y1, z1), (x2, y2, z2), (x3, y3, z3) in zip(data1, data2, data3)]

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Retrieve and process files from all three folders
    files1 = get_files(folder1)
    files2 = get_files(folder2)
    files3 = get_files(folder3)

    if len(files1) != len(files2) or len(files1) != len(files3):
        print("Warning: The number of files in the folders is not the same.")

    # Process corresponding files
    for i, (file1, file2, file3) in enumerate(zip(files1, files2, files3)):
        file1_path = os.path.join(folder1, file1)
        file2_path = os.path.join(folder2, file2)
        file3_path = os.path.join(folder3, file3)

        print(f"Processing pump delay {i}: {file1_path}, {file2_path}, {file3_path}")

        # Extract data from all three files
        data1 = read_and_extract_columns(file1_path)
        data2 = read_and_extract_columns(file2_path)
        data3 = read_and_extract_columns(file3_path)

        # Average the datasets
        averaged_data = average_datasets(data1, data2, data3)

        # Save the averaged data to a .d24 file
        output_file = os.path.join(output_folder, f"averaged_pump_delay_{i}.d24")
        with open(output_file, 'w') as file:
            for row in averaged_data:
                file.write(f"{row[0]:.6f}\t{row[1]:.6f}\t{row[2]:.6f}\n")

        print(f"Saved averaged data for pump delay {i} to {output_file}")

        # Plot the averaged data
        plot_file = os.path.join(output_folder, f"plot_pump_delay_{i}.png")
        x_vals = [row[0] for row in averaged_data]
        y_vals = [row[1] for row in averaged_data]
        z_vals = [row[2] for row in averaged_data]

        plt.figure()
        plt.plot(x_vals, y_vals, label="Y values")
        plt.plot(x_vals, z_vals, label="Z values")
        plt.xlabel("X values")
        plt.ylabel("Averaged Values")
        plt.title(f"Averaged Data for Pump Delay {i}")
        plt.legend()
        plt.savefig(plot_file)
        plt.close()

        print(f"Saved plot for pump delay {i} to {plot_file}")

if __name__ == "__main__":
    folder1 = "Dec12_2D_Temp200_Avg1"
    folder2 = "Dec12_2D_Temp200_Avg2"
    folder3 = "Dec12_2D_Temp200_Avg3"
    output_folder = "Dec12_2D_Temp200_AveragedData"
    process_folders(folder1, folder2, folder3, output_folder)
