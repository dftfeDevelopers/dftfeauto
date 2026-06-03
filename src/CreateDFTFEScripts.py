import numpy as np
import os
import sys
import json
import math
import copy
import glob
import ModifyDFTFEParams as modprm
import PeriodicTableDict as ptd
import ParseUPF as pupf

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
    rootDir = os.path.abspath(inp["rootDir"])
    inpLengthUnit = inp["inpLengthUnit"]
    outFname = inp["outFname"]
    modulesAndEnv = inp.get("modulesAndEnv")
    inpParamsJSON = inp["inpParamsJSON"]
    gsParamsFname = inp["gsParamsFname"]
    execPath = os.path.abspath(inp["execPath"])
    device = inp["device"]
    slurm = inp["slurm"]
    jobscript = inp["jobscript"]

    # sysComputeInfo: list of {sysRange, nodesPerSys} dicts.
    # sysRange: {"start": null, "end": null} covers all systems and must be the only entry.
    # Multiple entries allow different nodesPerSys for different explicit ranges.
    sysComputeInfo = inp["sysComputeInfo"]
    if len(sysComputeInfo) > 1:
        for entry in sysComputeInfo:
            if entry["sysRange"]["start"] is None and entry["sysRange"]["end"] is None:
                raise RuntimeError("sysComputeInfo: sysRange {start: null, end: null} covers all systems "
                                   "and cannot be combined with other entries. Use explicit ranges.")

    pspUPFPath = inp.get("pspUPFPath", None)

    invdft = inp.get("invdft", None)

    # tddft = inp.get("tddft", None)  # future extension

    # Load all unique primary systems across all sysComputeInfo entries
    systemsPrimary = {}
    for entry in sysComputeInfo:
        s = entry["sysRange"]["start"]
        e = entry["sysRange"]["end"]
        systemsPrimary.update(getSysData(sysJSONPrimary, s, e))

    # if pseudopotentials are provided, parse each UPF once, validate symbols, cache ZValence
    symZValence = {}
    if pspUPFPath is not None:
        allSymbols = set()
        for sysData in systemsPrimary.values():
            allSymbols.update(sysData["symbols"])
        missingSyms = allSymbols - set(pspUPFPath.keys())
        if missingSyms:
            raise RuntimeError(f"No pseudopotential provided in pspUPFPath for symbols: {missingSyms}")

        for sym, upfFile in pspUPFPath.items():
            upf = pupf.ParseUPF(upfFile)
            upfSym = upf.getSymbol().strip()
            if upfSym != sym:
                raise RuntimeError(f"Symbol mismatch: pspUPFPath key '{sym}' but UPF file reports element '{upfSym}'")
            symZValence[sym] = int(upf.getZValence())

    if invdft is not None:
        systemsSecondary = {}
        for entry in sysComputeInfo:
            s = entry["sysRange"]["start"]
            e = entry["sysRange"]["end"]
            systemsSecondary.update(getSysData(invdft["secondary"], s, e))
        checkData = areSame(systemsPrimary, systemsSecondary)
        if checkData["same"] == False:
            raise RuntimeError('''Incompatible data for the following systems:''', checkData['sys'])

    gsParamsBase = None
    invParams = None
    with open(inpParamsJSON) as ff:
        inpParams = json.load(ff)
        gsParamsBase = inpParams["gs"]
        if invdft is not None:
            invParams = inpParams["inv"]

    # loop over each system and create separate folders with necessary input files
    sysDirs = {}
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

        # fresh copy of defaults each system; override with any per-system fields present in sysData
        gsParams = copy.deepcopy(gsParamsBase)
        for k, v in sysData.items():
            if k in gsParams:
                gsParams[k] = v

        if "domain" in sysData:
            domain = sysData["domain"]
        elif inp.get("domain", None) is not None:
            domain = inp["domain"]
        else:
            raise RuntimeError(f"No domain found for system '{sys}' and no fallback domain in input file.")

        factor = 1.0
        if inpLengthUnit[0].lower() == "a":
            factor = __ANGS2BOHR__

        coordsBohr = (np.array(coords)*factor).tolist()

        ## GS setup
        gsParams["natoms"] = natoms
        gsParams["ntypes"] = len(set(symbols))
        gsParamsFpath = os.path.join(sysDir, gsParamsFname)
        modprm.modifyParamsFile(gsParams, gsParamsFpath, mode="gs")

        # NOTE: DFTFE coords need to be in bohr
        dftfeCoordsFname = gsParams["coordsFile"]
        dftfeCoordsFpath = os.path.join(sysDir, dftfeCoordsFname)
        with open(dftfeCoordsFpath, "w") as ff:
            for k in range(natoms):
                sym = symbols[k]
                Z = ptd.__periodicTableSymbolDict_[sym]["AtomicNumber"]
                ZValence = symZValence[sym] if pspUPFPath is not None else ptd.__periodicTableSymbolDict_[sym]["NumberofValenceElectrons"]
                print(Z, ZValence, coordsBohr[k][0], coordsBohr[k][1], coordsBohr[k][2], file = ff)

        domainFname = gsParams["domainFile"]
        domainFpath = os.path.join(sysDir, domainFname)
        np.savetxt(domainFpath, np.array(domain))

        # write pseudo.inp (pseudopotential run); omitted for all-electron
        if pspUPFPath is not None:
            uniqueSyms = sorted(set(symbols),
                                key=lambda s: ptd.__periodicTableSymbolDict_[s]["AtomicNumber"])
            pseudoFpath = os.path.join(sysDir, "pseudo.inp")
            with open(pseudoFpath, "w") as ff:
                for sym in uniqueSyms:
                    Z = ptd.__periodicTableSymbolDict_[sym]["AtomicNumber"]
                    upfFile = pspUPFPath[sym]
                    os.system("cp " + upfFile + " " + sysDir)
                    print(Z, os.path.basename(upfFile), file=ff)

        ## invdft setup
        if invdft is not None:
            coordsAngs = (np.array(coordsBohr)/__ANGS2BOHR__).tolist()

            invParamsFname = invdft["invParamsFname"]
            invParamsFpath = os.path.join(sysDir, invParamsFname)
            modprm.modifyParamsFile(invParams, invParamsFpath, mode="inv")

            # copy basis files to sys directory
            basisDir = invdft["basisDir"]
            basisName = invdft["basisName"]
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

            # NOTE: AO coords need to be in angstrom
            aoCoordsFname = invParams["atomCoords"]
            aoCoordsFpath = os.path.join(sysDir, aoCoordsFname)
            with open(aoCoordsFpath, "w") as ff:
                for k in range(natoms):
                    sym = symbols[k]
                    print(sym, coordsAngs[k][0], coordsAngs[k][1], coordsAngs[k][2], symBasisFile[sym], file = ff)

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

        ## tddft setup
        # if tddft is not None:
        #     ...

        sysDirs[sys] = sysDir

    ### write slurm script
    # total nodes = sum over entries of (nodesPerSys * number of systems in that entry's range)
    totalNodes = 0
    for entry in sysComputeInfo:
        s = entry["sysRange"]["start"]
        e = entry["sysRange"]["end"]
        nSys = len(getSysData(sysJSONPrimary, s, e))
        totalNodes += int(math.ceil(nSys * int(entry["nodesPerSys"])))

    gpusPerTask = ""
    gpusBind = ""
    if device.lower() == "gpu":
        gpusPerTask = "--gpus-per-task=1"
        gpusBind = "--gpu-bind=closest"

    tasksPerNode = "--ntasks-per-node=" + str(slurm["tasksPerNode"])
    cpusPerTask = "--cpus-per-task=" + str(slurm["cpusPerTask"])

    # if jobscript exists, create a new one with name jobscript_<N>
    if os.path.exists(jobscript):
        created = False
        jobscriptOrig = copy.deepcopy(jobscript)
        count = 1
        while not created:
            jobscript = jobscriptOrig + "_" + str(count)
            if not os.path.exists(jobscript):
                created = True
            else:
                count += 1

    f = open(jobscript, "w")
    # print top level SBATCH directives to the jobscript file
    print("#!/bin/sh", file = f)
    print("#SBATCH -A " + str(slurm["account"]), file = f)
    print("#SBATCH -J " + str(slurm["jobname"]), file = f)
    print("#SBATCH -o " + str(slurm["jobname"]) + ".out.%j", file = f)
    print("#SBATCH -e " + str(slurm["jobname"]) + ".err.%j", file = f)
    if slurm["queue"] is not None and str(slurm["queue"]).strip() != "":
        print("#SBATCH -q " + str(slurm["queue"]), file = f)

    if slurm["partition"] is not None and str(slurm["partition"]).strip() != "":
        print("#SBATCH -p " + str(slurm["partition"]), file = f)

    if slurm["extra"] is not None and str(slurm["extra"]).strip() != "":
        for ex in slurm["extra"]:
            print("#SBATCH " + str(ex), file = f)

    print("#SBATCH -t " + str(slurm["t"]), file = f)
    print("#SBATCH --nodes=" + str(totalNodes), file = f)
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

    # srun lines grouped by sysComputeInfo entry (each with its own nodesPerSys)
    execArgs = gsParamsFname
    if invdft is not None:
        execArgs += " " + invdft["invParamsFname"]
    # if tddft is not None:
    #     execArgs += " " + tddft["tddftParamsFname"]

    for entry in sysComputeInfo:
        s = entry["sysRange"]["start"]
        e = entry["sysRange"]["end"]
        entrySystems = getSysData(sysJSONPrimary, s, e)
        n = int(entry["nodesPerSys"])
        nodesPerSysArg = "--nodes=" + str(n)
        for sys in entrySystems:
            dirname = sysDirs[sys]
            print("cd " + dirname, file = f)
            srunLine = "srun " + nodesPerSysArg + " " + tasksPerNode + " " + \
                        cpusPerTask + " " + gpusPerTask + " " + gpusBind + " " + \
                        execPath + " " + execArgs + " &> " + outFname + " &"
            print(srunLine, file = f)
            print(file = f)
            print(dirname, file=dirsf)

    dirsf.close()
    print("wait", file = f)
    print("""echo "Successfully ran all the calculations." """, file = f)
    f.close()
    print("Jobscript written: " + jobscript)


if __name__ == "__main__":
    inp = None
    with open(sys.argv[1]) as ff:
        inp = json.load(ff)

    run(inp)
