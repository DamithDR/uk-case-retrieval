import csv
import os


def check_header(results_file_name, header):
    write_header = True
    if os.path.exists(results_file_name):
        with open(results_file_name, "r", newline="") as f:
            reader = csv.reader(f)
            first_row = next(reader, None)
            if first_row == header:
                write_header = False
    if write_header:
        with open(results_file_name, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)


def write_results(MAP, f1, p, r, model, k_values):
    results_file_name = "results/results.csv"
    header = ["Model", "Metric"] + [f"k_{k}" for k in k_values]
    check_header(results_file_name, header)
    with open(results_file_name, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([model, "F1", *f1])
        writer.writerow([model, "Precision", *p])
        writer.writerow([model, "Recall", *r])

    map_results_file = 'results/map_results.csv'
    map_header = ["Model", "MAP"]
    check_header(map_results_file, map_header)
    with open(map_results_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([model, MAP])
