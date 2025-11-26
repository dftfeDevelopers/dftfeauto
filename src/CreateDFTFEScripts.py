import numpy as np
import os
import sys
import json
import math
import copy
import glob
import ModifyDFTFEParams as modprm

__ANGS2BOHR__ = 1.8897259886

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


def areSame(systems1, systems2):
    same = True
    syslist = []
    keysFloat = ["coords", "charge", "mult"]
    keysStr = ["symbols"]
    for sys in systems1:
        if sys not in systems2:
            same = False
            syslist.append(sys)

        for k in keysFloat:
            tmp = np.array(systems1[sys][k]) - np.array(systems2[sys][k])
            if np.linalg.norm(tmp) > 1e-10:
                same = False
                syslist.append(sys)

        for k in keysStr:
            if systems1[sys][k] != systems2[sys][k]:
                same = False
                syslist.append(sys)

    ret = {"same": same, "sys": syslist}
    return ret


def run(inp):
    sysJSONPrimary = inp["sysJSON"]["primary"]
    sysJSONSecondary = inp["sysJSON"]["secondary"]
    rootDir = os.path.abspath(inp["rootDir"])
    inpLengthUnit = inp["inpLengthUnit"]
    outFname = inp["outFname"]
    basisDir = inp["basisDir"]
    basisName = inp["basisName"]
    startID = inp["sysRange"]["start"]
    endID = inp["sysRange"]["end"]
    slurm = inp["slurm"]
    modulesAndEnv = inp["modulesAndEnv"]

    jobscript = os.path.abspath(inp["jobscript"]+ "_" + str(startID) + "_" + str(endID))
    symToZ = inp["symToZ"]
    domain = inp["domain"]
    inpParamsJSON = inp["inpParamsJSON"]
    gsParamsFname = inp["gsParamsFname"]
    invParamsFname = inp["invParamsFname"]
    execPath = os.path.abspath(inp["execPath"])
    systemsPrimary = getSysData(sysJSONPrimary, startID, endID)
    systemsSecondary = getSysData(sysJSONSecondary, startID, endID)
    checkData = areSame(systemsPrimary, systemsSecondary)
    if checkData["same"] == False:
        raise RuntimeError('''Incompatible data for the following systems:''', checkData['sys'])

    gsParams = None
    invParams = None
    with open(inpParamsJSON) as ff:
        inpParams = json.load(ff)
        gsParams = inpParams["gs"]
        invParams = inpParams["inv"]

    # loop over each system and create separate folders
    # with necessary input files
    dirs = []
    for sys in systemsPrimary:
        sysSubset = None
        sysName = sys
        sysData = systemsPrimary[sys]
        print('sys', sys)
        if ":" in sys:
            words = sys.split(":")
            sysSubset = words[0]
            sysName = words[1]

        if sysSubset is not None:
            sysDir = os.path.join(rootDir, sysSubset, sysName)
        else:
            sysDir = os.path.join(rootDir, sysName)

        if not os.path.exists(sysDir):
            os.makedirs(sysDir)

        symbols = sysData["symbols"]
        coords = sysData["coords"]
        natoms = len(symbols)

        ## modify gsParams and invParams as necessary
        gsParams["natoms"] = natoms
        gsParams["ntypes"] = len(set(symbols))
        gsParamsFpath = os.path.join(sysDir, gsParamsFname)
        invParamsFpath = os.path.join(sysDir, invParamsFname)

        # write gs and inverse parameters files
        modprm.modifyParamsFile(gsParams, gsParamsFpath, mode="gs")
        modprm.modifyParamsFile(invParams, invParamsFpath, mode="inv")

        factor = 1.0
        if inpLengthUnit[0].lower() == "a":
            factor = __ANGS2BOHR__

        coordsBohr = (np.array(coords)*factor).tolist()
        coordsAngs = (np.array(coordsBohr)/__ANGS2BOHR__).tolist()

        # copy basis file to sys directory
        symBasisFile = {}
        for sym in set(symbols):
            tmpstr = basisDir + "/*" + sym + "*" + basisName + "*"
            matchFiles = glob.glob(tmpstr)
            nmatch = len(matchFiles)
            basisFile = None
            if nmatch == 0:
                raise RuntimeError(f'''No {basisName} basis file found for '''\
                                   f'''chemical species {sym} in folder: {basisDir}''')

            elif nmatch == 1:
                basisFile = matchFiles[0]

            elif nmatch == 2:
                for bb in matchFiles:
                    if "nwchem" not in bb.lower():
                        basisFile = bb

            else:
                raise RuntimeError(f'''Multiple {basisName} file available for '''\
                                   f'''chemical species {sym} in folder: {basisDir}''')

            os.system("cp " + basisFile + " " + sysDir)
            symBasisFile[sym] = os.path.basename(basisFile)

        dftfeCoordsFname = gsParams["coordsFile"]
        aoCoordsFname = invParams["atomCoords"]
        dftfeCoordsFpath = os.path.join(sysDir, dftfeCoordsFname)
        aoCoordsFpath = os.path.join(sysDir, aoCoordsFname)

        # NOTE: DFTFE coords need to be in bohr
        with open(dftfeCoordsFpath, "w") as ff:
            for k in range(natoms):
                sym = symbols[k]
                Z = symToZ[sym]["Z"]
                ZValence = symToZ[sym]["ZValence"]
                print(Z, ZValence, coordsBohr[k][0], coordsBohr[k][1], coordsBohr[k][2], file = ff)

        # NOTE: AO coords need to be in angstrom
        with open(aoCoordsFpath, "w") as ff:
            for k in range(natoms):
                sym = symbols[k]
                print(sym, coordsAngs[k][0], coordsAngs[k][1], coordsAngs[k][2], symBasisFile[sym], file = ff)


        domainFname = gsParams["domainFile"]
        domainFpath = os.path.join(sysDir, domainFname)
        np.savetxt(domainFpath, np.array(domain))

        # copy density matrix to sys directory
        DMPrimaryFname = invParams.get("aoDMPrimary", None)
        DMSecondaryFname = invParams.get("aoDMSecondary", None)
        if DMPrimaryFname is not None:
            DMAlpha = 0.5*np.array(sysData["DM"])
            DMPath = os.path.join(sysDir, DMPrimaryFname)
            np.savetxt(DMPath, DMAlpha)

        if DMSecondaryFname is not None:
            DMAlpha = 0.5*np.array(systemsSecondary[sys]["DM"])
            DMPath = os.path.join(sysDir, DMSecondaryFname)
            np.savetxt(DMPath, DMAlpha)

        # copy S matrix
        SMFname = invParams.get("aoS", None)
        if SMFname is not None:
            SM = np.array(systemsSecondary[sys]["SM"])
            SMPath = os.path.join(sysDir, SMFname)
            np.savetxt(SMPath, SM)


        dirs.append(sysDir)

    ### write slurm script
    device = inp["device"]
    jobscript = inp["jobscript"]
    modulesAndEnv = inp["modulesAndEnv"]
    n = int(inp["nodesPerSys"])
    slurm = inp["slurm"]
    nodes = int(math.ceil(len(dirs)*n))
    gpusPerTask = ""
    gpusBind = ""
    if device.lower() == "gpu":
        gpusPerTask =  "--gpus-per-task=1"
        gpusBind = "--gpu-bind=closest"

    nodesPerSys = "--nodes=" + str(n)
    tasksPerNode = "--ntasks-per-node=" + str(slurm["tasksPerNode"])
    cpusPerTask = "--cpus-per-task=" + str(slurm["cpusPerTask"])

    # if jobscript exists, created a new one with name jobscript_<N>,
    #where <N> is the lowest positive integer such that jobscript_<N> does not exist
    if os.path.exists(jobscript):
        created = False
        jobscriptOrig = copy.deepcopy(jobscript)
        count = 1
        while not created:
            jobscript = jobscriptOrig +"_" + str(count)
            if not os.path.exists(jobscript):
                created = True

            else:
                count += 1


    f = open(jobscript, "w")
    # print top level SBATCH directives to the jobscript file
    print("#!/bin/sh", file = f)
    print("#SBATCH -A " + str(slurm["account"]), file = f)
    print("#SBATCH -J " + str(slurm["jobname"]), file = f)
    print("#SBATCH -o " + str(slurm["jobname"])+".out.%j", file = f)
    print("#SBATCH -e " + str(slurm["jobname"])+".err.%j", file = f)
    if slurm["queue"] is not None and str(slurm["queue"]).strip() != "":
        print("#SBATCH -q " + str(slurm["queue"]), file = f)

    if slurm["partition"] is not None and str(slurm["partition"]).strip() != "":
        print("#SBATCH -p " + str(slurm["partition"]), file = f)

    if slurm["extra"] is not None and str(slurm["extra"]).strip() != "":
        for ex in slurm["extra"]:
            print("#SBATCH " + str(ex), file = f)

    print("#SBATCH -t " + str(slurm["t"]), file = f)
    print("#SBATCH --nodes="+str(nodes), file = f)
    print("#SBATCH --mail-type=BEGIN,END", file = f)
    print("#SBATCH --mail-user=" + str(slurm["email"]), file = f)
    print("\n", file = f)

    # print modules and environment to the jobscript file
    if modulesAndEnv is not None and str(modulesAndEnv).strip() != "":
        with open(modulesAndEnv) as f2:
            lines = f2.readlines()
            for line in lines:
                print(line.strip(), file = f)

    print("\n", file = f)
    dirsf = open("dirs", "w")
    for dirname in dirs:
        print("cd " +  dirname, file = f)
        srunLine = "srun " + nodesPerSys + " " + tasksPerNode + " " + \
                    cpusPerTask + " " + gpusPerTask + " " + gpusBind + " " +\
                    execPath + " " + gsParamsFname + " " + invParamsFname + " &> " + outFname + " &"
        print(srunLine, file = f)
        print(file = f)
        print(dirname, file=dirsf)

    dirsf.close()

    print("wait", file = f)
    print("""echo "Successfully ran all the calculations." """, file = f)
    f.close()
    os.system("sbatch " + jobscript)


if __name__ == "__main__":
    inp = None
    with open(sys.argv[1]) as ff:
        inp = json.load(ff)

    run(inp)
