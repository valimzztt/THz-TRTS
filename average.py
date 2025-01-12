import os
from datetime import datetime

def process_folders(folder1, folder2, folder3, output_folder):
    def read_and_extract_columns(file_path):
        extracted_data = []
        header = None
        with open(file_path, 'r') as file:
            for line in file:
                if line.strip():
                    if header is None:
                        header = line.strip()
                    else:
                        columns = line.split()
                        extracted_data.append((float(columns[0]), float(columns[1]), float(columns[2])))
        return header, extracted_data

    # Function to average three datasets
    def average_datasets(data1, data2, data3):
        if len(data1) != len(data2) or len(data1) != len(data3):
            raise ValueError("Datasets have different lengths.")
            return []
        return [(x1, (y1 + y2 + y3) / 3, (z1 + z2 + z3) / 3) 
                for (x1, y1, z1), (x2, y2, z2), (x3, y3, z3) in zip(data1, data2, data3)]

    os.makedirs(output_folder, exist_ok=True)
    
    def get_files_sorted_by_date(folder):
        files = [f for f in os.listdir(folder) if f.endswith('Average.d24')]
        def extract_date_from_filename(filename):
            try:
                date_str = ''.join(filename.split(' ')[1:3])
                return datetime.strptime(date_str, '%b%d%H%M')
            except (IndexError, ValueError) as e:
                print(f"Error processing file {filename}: {e}")
                return None

        files.sort(key=lambda f: extract_date_from_filename(f) or datetime.min)
        return files

    files1 = get_files_sorted_by_date(folder1)
    files2 = get_files_sorted_by_date(folder2)
    files3 = get_files_sorted_by_date(folder3)

    if len(files1) != len(files2) or len(files1) != len(files3):
        raise ValueError("The number of files in the folders is not the same.")

    for i, (file1, file2, file3) in enumerate(zip(files1, files2, files3)):
        if(i==0):
            file1_path = os.path.join(folder1, file1)
            file2_path = os.path.join(folder2, file2)
            file3_path = os.path.join(folder3, file3)

            header1, data1 = read_and_extract_columns(file1_path)
            _, data2 = read_and_extract_columns(file2_path)
            _, data3 = read_and_extract_columns(file3_path)
            print(data1[0:10], data2[0:10], data3[0:10])

            averaged_data = average_datasets(data1, data2, data3)
            print(averaged_data[0:10])
            output_file = os.path.join(output_folder, f"averaged_pump_delay_{i}.d24")
            with open(output_file, 'w') as file:
                file.write(header1 + '\n')
                lines = [f"{row[0]:.6f}\t{row[1]:.6f}\t{row[2]:.6f}\n" for row in averaged_data]
                file.writelines(lines)

if __name__ == "__main__":
    folder1 = "Dec12_2D_Temp200_Avg1"
    folder2 = "Dec12_2D_Temp200_Avg2"
    folder3 = "Dec12_2D_Temp200_Avg3"
    output_folder = "Dec12_2D_Temp200_AveragedDataNew"
    process_folders(folder1, folder2, folder3, output_folder)
