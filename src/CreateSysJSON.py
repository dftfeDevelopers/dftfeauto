import os
import sys
import json
import numpy as np
import glob

def getGeom(fpath, nskiplines):
    symbols = []
    coords = []
    with open(fpath, "r") as ff:
        lines = ff.readlines()
        nlines = len(lines)
        for i in range(nskiplines, nlines):
            ll = lines[i].strip()
            if ll:
                words = ll.split()
                symbols.append(words[0])
                c = [float(x) for x in words[1:]]
                coords.append(c)

    return {"symbols": symbols, "coords": coords}

def processCSV(fpath, nskiplines):
    data = {}
    with open(fpath, 'r') as ff:
        lines = ff.readlines()
        nlines = len(lines)
        for i in range(nskiplines, nlines):
            ll = lines[i].strip()
            if ll:
                words = ll.split(",")
                sysName = words[0]
                energy = float(words[2])
                IP = float(words[4])
                data[sysName] = {"energy": energy, "IP": IP}

    return data

def getBasis(basisDir, symbols):
    data = {}
    for sym in symbols:
        fname = glob.glob(basisDir+"/" + "*" + sym + "*nwchem")
        if len(fname) > 1:
            raise Exception("More than one basis file found in " + basisDir + " for symbol" + sym)
        fpath = os.path.join(basisDir, fname[0])
        with open(fpath, "r") as ff:
            data[sym] = ff.read()
    return data

def run(rootDir, basisname, basisDir, outfilename):
    subDirs = [f.path for f in os.scandir(rootDir) if f.is_dir()]
    data = {}
    for subDir in subDirs:
        print(subDir)
        # read csv file
        csvFile = glob.glob(subDir+"/*.csv")[0]
        csvPath = os.path.join(csvFile)
        energyAndIPData = processCSV(csvPath, nskiplines=1)
        subSubDirs = [f.path for f in os.scandir(subDir) if f.is_dir()]
        subsetName = os.path.basename(subDir)
        for subSubDir in subSubDirs:
            print(subSubDir)
            sysname = os.path.basename(subSubDir)
            geomPath = os.path.join(subSubDir, "GEOM")
            dmPath = os.path.join(subSubDir, "run1/", "Pao_ci")
            geom = getGeom(geomPath, nskiplines=1)
            DM = np.loadtxt(dmPath, skiprows=1)
            symbols = geom["symbols"]
            coords = geom["coords"]
            basis = getBasis(basisDir, symbols)
            energy = energyAndIPData[sysname]["energy"]
            IP = energyAndIPData[sysname]["IP"]
            sysfullname = subsetName + ":" + sysname
            data[sysfullname] = {"symbols": symbols,\
                             "coords": coords,\
                             "energy": energy,\
                             "IP": IP,\
                             "DM": DM.tolist(),
                             "basisname": basisname,
                             "basis": basis,\
                             "charge": 0,\
                             "mult": 1}

    with open(outfilename, "w") as ff:
        json.dump(data, ff, indent=2)


if __name__=="__main__":
    inp = None
    with open(sys.argv[1]) as ff:
        inp = json.load(ff)

    rootDir = inp["rootDir"]
    basisDir = inp["basisDir"]
    basisname = inp["basisname"]
    outfilename = inp["outfilename"]
    run(rootDir, basisname, basisDir, outfilename)
