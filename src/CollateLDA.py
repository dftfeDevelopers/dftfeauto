import sys
import os
import copy
import json
import glob
import numpy as np

def getSysData(sysJSON, start=None, end=None, includeChargedSys = True, chargeTol = 1e-6):
    data = None
    sysNames = []
    systems = {}
    tmp = None
    with open(sysJSON) as f:
        tmp = json.load(f)

    sysKeys = list(tmp.keys())
    if start is None:
        start = 0

    if end is None:
        end = len(tmp.keys())

    else:
        end = min(end, len(sysKeys))

    for sysId in range(start,end):
        s  = sysKeys[sysId]
        systems[s] = tmp[s]

    return systems

def run(inp, stdoutF = sys.stdout):
    sysJSON = inp["sysJSON"]
    rootDir = os.path.abspath(inp["rootDir"])
    sysFilePostfix = inp["sysFilePostfix"]
    outJSON = inp["outJSON"]
    dmFname = "DM" + sysFilePostfix
    # get systems based on start and end IDs
    systems = getSysData(sysJSON)
    stdoutF.flush()
    nsys = len(systems)
    data = {}
    for s in systems:
        print(s)
        sysData = systems[s]
        words = s.split(":")
        subset = words[0]
        sysName = words[1]
        sysDir = os.path.join(rootDir, subset, sysName)

        data[s] = {"symbols": sysData["symbols"],\
                   "coords": sysData["coords"],\
                   "charge": sysData["charge"],
                   "mult": sysData["mult"]
                   }

        dmfiles = glob.glob(sysDir +"/"+dmFname + "*")
        DM = None
        if len(dmfiles) == 1:
            DM = np.loadtxt(dmfiles[0])

        elif len(dmfiles)==2:
            if sysData["mult"] == 1:
                raise RuntimeError(f'''Multiple density matrices found for '''\
                                   f'''spin-unpolaried system {s}''')

            DM0 = np.loadtxt(dmfiles[0])
            DM1 = np.loadtxt(dmfiles[1])
            DM = np.zeros((2, DM[0].shape[0], DM[0].shape[1]))
            DM[0] = DM0
            DM[1] = DM1

        else:
            raise RuntimeError('''Invalid number of density matrices found for '''\
                               f'''system {s}''')

        kseigs = np.loadtxt(os.path.join(sysDir, 'kseigs'))
        ksocc = np.loadtxt(os.path.join(sysDir, 'ksocc'))
        S = np.loadtxt(os.path.join(sysDir, "SMatrix"))
        data[s]["DM"] = DM.tolist()
        data[s]["SM"] = S.tolist()
        data[s]["kseigs"] = kseigs.tolist()
        data[s]["ksocc"] = ksocc.tolist()

    with open(outJSON, "w") as ff:
        json.dump(data, ff, indent=2)


if __name__ == "__main__":
    correctUse = True
    usageMsg = """Usage: `python CollateLDA.py inp.json [stdout_file]`,"""\
               """ where inp.json contains various input parameters,"""\
               """ and stdout_file is an optional filename to redirect the stdout."""\

    if len(sys.argv) not in [2,3,4,5]:
        correctUse = False

    else:
        if sys.argv[1] == "--help" or sys.argv[1] == "-h":
            print(usageMsg)
            sys.exit(0)

        else:
            if ".json" not in sys.argv[1]:
                correctUse  = False

    if not correctUse:
        print("Incorrect arguments. """ + usageMsg)
        raise RuntimeError

    inpJSON = str(sys.argv[1])

    inp = {}
    with open(inpJSON) as f:
        inp = json.load(f)

    # modify things if start and end ids are given
    if len(sys.argv) > 3:
        inp["sysRange"]["start"] = int(sys.argv[2])
        inp["sysRange"]["end"] = int(sys.argv[3])

    stdOutF = sys.stdout
    if inp["stdOut"] is not None:
        stdOutF = open(inp["stdOut"], "w")

    run(inp, stdOutF)

    stdOutF.close()
