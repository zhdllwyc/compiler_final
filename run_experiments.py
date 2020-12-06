import json
import argparse as ap
import os
import subprocess
import resource
# Collect benchmark data for PageRank and BFS
# directory with graphs on Tuxedo
graph_dir = "/net/ohm/export/iss/inputs"

# map alias to filename
graphs = {"road-usa": "road/USA-road-d.USA.gr", "road-fl": "road/USA-road-d.FLA.gr", # road
        "wikipedia": "unweighted/wikipedia-20061104.gr", "google": "unweighted/web-Google-clean.gr", # web
        "rmat21": "scalefree/rmat16-2e21-a=0.57-b=0.19-c=0.19-d=.05.gr", # rmat
        "rmat24": "scalefree/rmat16-2e24-a=0.57-b=0.19-c=0.19-d=.05.gr",
        "livejournal": "stanford/communities/LiveJournal/com-lj.gr", #social
        "orkut": "unweighted/orkut-component.gr"}

dir_path = os.path.dirname(os.path.realpath(__file__))
galois_path = os.getenv("HOME") + "/Galois/build/lonestar/analytics/gpu"

executables = {"pagerank":{
    "scalar": [dir_path+"/power_iteration_pr",  "s"],
    "adaptive": [dir_path+"/power_iteration_pr",  "a"],
    "worklist": dir_path+"/worklist_pr",
    "serial": dir_path+"/test_serial_algos",
    "galois": galois_path+"/pagerank/pagerank-gpu"},

    "bfs":{
    "worklist": dir_path+"/bfs_worklist",
    "op": dir_path+"/bfs_op",
    # TODO - add CPU implementation of BFS
    #"serial": [dir_path+"/test_serial_algos", "bfs"],
    # default to 0 as the starting node for Galois
    "galois": [galois_path+"/bfs/bfs-gpu", "-s0"]},
}

def extract_galois_info(performance, stdout):
    print(stdout)
    for l in stdout.split("\n"):
        if l.startswith("Total time") and "ms" in l:
            performance[args.problem] = float(l.strip().split()[2]) / 1000
        if l.startswith("PR took"): performance['iterations'] = int(l.strip().split()[2])


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("algo", choices=["scalar", "adaptive", "worklist", "serial", "galois"], default="worklist")
    parser.add_argument("--problem", choices=["pagerank", "bfs"], default="pagerank")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()
    print(args)

    try:
        executable = executables[args.problem][args.algo]
        if type(executable) is str: executable = [executable]
    except KeyError:
        print(f"There is no algo {args.algo} for {args.problem}.")
        raise

    output_pref = dir_path+"/results/"+args.problem+"_"+args.algo
    if args.algo == "galois":
        # resource.getrusage computes a running total of the runtimes of all children
        # consequently, we need to compute differences to get the runtime for a single child
        last_utime = last_stime = 0.0

    for i in range(args.runs):
        for alias, filename in graphs.items():
            full_path = graph_dir+"/" + filename
            output_file = output_pref+"_"+alias+str(i)+".json"
            if os.path.exists(output_file): continue
            print(f"Processing {alias}, round {i}")
            try:
                if args.algo == "galois":
                    if alias == "rmat24": continue
                    res = subprocess.run(args=executable+[full_path], check=True,
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="ascii")
                    info = resource.getrusage(resource.RUSAGE_CHILDREN)
                    performance = {"total": info.ru_stime+info.ru_utime-last_stime-last_utime}
                    last_stime, last_utime = info.ru_stime, info.ru_utime
                    extract_galois_info(performance, res.stdout+res.stderr)
                    with open(output_file, "w") as fp:
                        json.dump(performance, fp)
                else:
                    process_args = [executable[0]]+[full_path]+executable[1:]+[output_file]
                    res = subprocess.run(args=process_args, check=True)
            except subprocess.CalledProcessError as e:
                print(e)
                print(res)
                print(f"Error processing {alias}!")

