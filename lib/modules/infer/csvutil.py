

import csv

# praatEXE = join('.',os.path.abspath(os.getcwd()) + r"\Praat.exe")


def CSVutil(file, rw, type, *args):
    if type == "formanting":
        if rw == "r":
            with open(file) as fileCSVread:
                csv_reader = list(csv.reader(fileCSVread))
                return (
                    (csv_reader[0][0], csv_reader[0][1], csv_reader[0][2])
                    if csv_reader is not None
                    else (lambda: exec('raise ValueError("No data")'))()
                )
        else:
            if args:
                doformnt = args[0]
            else:
                doformnt = False
            qfr = args[1] if len(args) > 1 else 1.0
            tmb = args[2] if len(args) > 2 else 1.0
            with open(file, rw, newline="") as fileCSVwrite:
                csv_writer = csv.writer(fileCSVwrite, delimiter=",")
                csv_writer.writerow([doformnt, qfr, tmb])
    elif type == "stop":
        stop = args[0] if args else False
        with open(file, rw, newline="") as fileCSVwrite:
            csv_writer = csv.writer(fileCSVwrite, delimiter=",")
            csv_writer.writerow([stop])

