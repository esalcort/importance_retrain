import pandas as pd
import os
import argparse

def Namespace(**kwargs):
    return locals()['kwargs']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('destination_csv')
    parser.add_argument('log_files', nargs='+')
    parser.add_argument('--print_file_done_list')
    args = parser.parse_args()
    
    if os.path.exists(args.destination_csv):
        df = pd.read_csv(args.destination_csv)
    else:
        df = pd.DataFrame()
    
    for logfile in args.log_files:
        state = 0
        with open(logfile) as f:
            for line in f:
                if state == 0:
                    if 'Configuration:' in line:
                        state = 1
                elif state == 1:
                    if 'experiment' in line:
                        config = eval(line)
                        state = 2
                    else:
                        state = 0
                elif state == 2:
                    if 'Results:' in line:
                        state = 3
                elif state == 3:
                    if 'train' in line and ('acc' in line or 'time' in line):
                        results = eval(line)
                        state = 0
                        merge_dict = {**config, **results}
                        df = df.append(merge_dict, ignore_index=True)
                        if args.print_file_done_list:
                            with open(args.print_file_done_list, 'a') as f:
                                f.write(logfile+'\n')
    print(df)
    df.to_csv(args.destination_csv, index=False)