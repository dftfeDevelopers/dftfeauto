import sys
import os
from os.path import exists
import filecmp
import copy
import json
import ntpath
_KB_ = 3.1668090406215046e-06
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

def getMolBuildFile(sysData, basisStr, outFname, lengthUnit = "angs", gridLevel=4):
    with open(outFname, "w") as f:
        print('''mol = gto.Mole()''', file = f)
        print('''mol.verbose = 4''', file = f)
        print('''mol.atom="""''', file = f)
        symbols = sysData["symbols"]
        coords = sysData["coords"]
        charge = sysData["charge"]
        spin = sysData["mult"] - 1
        for i,sym in enumerate(symbols):
            line = sym + " "
            for idim in range(3):
                line += str(coords[i][idim]) + " "

            line = line.strip()
            print(line, file = f)

        print('''"""''', file = f)
        print('''mol.unit = ''' + "'"+ lengthUnit + "'", file = f)
        print("mol.charge = " + str(charge), file = f)
        print("mol.spin  = " + str(spin), file = f)
        if isinstance(basisStr, str):
            print("mol.basis = " + "'" + basisStr + "'", file = f)
        else:
            tmpStr = "mol.basis = {"
            for elem in basisStr:
                bfile = basisStr[elem]
                tmpStr += "'" + elem + "'" + ":" + "'" + bfile + "'" + ","

            tmpStr = tmpStr[0:-1] + "}"
            print(tmpStr, file = f)

        print("mol.build()", file = f)
        ## add grid
        print("grid = dft.gen_grid.Grids(mol)", file = f)
        print(f"grid.level = {gridLevel}", file = f)
        print("grid.build()", file = f)
        print("qpts = grid.coords", file = f)
        print("qwts = grid.weights", file = f)
        if spin == 0:
            print("mfl = dft.RKS(mol)", file = f)
        else:
            print("mfl = dft.UKS(mol)", file = f)

        print("\n", file = f)


def getPreSCFFile(sysData, xcStr = "PBE", method = 'diis', convTol=1e-6, outFname="prescfTmp"):
    with open(outFname, "w") as f:
        spin = sysData["mult"] - 1
        if xcStr in ["PBE", "NNGGA", "SCAN", "R2SCAN"]:
            print("mfl.xc = 'PBE,PBE'", file = f)

        if xcStr in ["PW92", "NNLDA"]:
            print("mfl.xc = 'SPW92'", file = f)

        if method.lower() not in ['diis', 'newton']:
            raise ValueError(f'''Invalid method {method} passed. Valid methods: 'diis', 'newton' ''')

        if method.lower() == 'newton':
            print("mfl = mfl.newton()", file = f)

        print("mfl.max_cycle = 100", file = f)
        print(f"mfl.conv_tol = {convTol}", file = f)
        print("mfl.kernel()", file = f)
        print("dm = mfl.make_rdm1()", file = f)

        print("\n", file = f)
        if spin == 0:
            print("mfl = dft.RKS(mol)", file = f)
        else:
            print("mfl = dft.UKS(mol)", file = f)

        print("mfl.init_guess = dm", file = f)


def getSCFFile(sysData, xc, addons, postfix="tmp", outFname="scfTmp"):
    xcStr = xc["type"]
    with open(outFname, "w") as f:
        spin = sysData["mult"] - 1
        if xcStr == "PBE":
            print("mfl.xc = 'PBE,PBE'", file = f)

        elif xcStr == "NNLDA":
            ptcFile = xc["ptcPath"]
            tol = xc["tol"]
            print("tol = " + str(tol), file = f)
            print('''nnlda = NNLDA("''' + ptcFile + '''", tol)''', file = f)
            print('''mfl = mfl.define_xc_(eval_xc_lda, 'LDA')''', file = f)

        elif xcStr == "NNGGA":
            ptcFile = xc["ptcPath"]
            tol = xc["tol"]
            sthres = xc["sthres"]
            print("tol = " + str(tol), file = f)
            print("sthres = " + str(sthres), file = f)
            print('''nngga = NNGGA("''' + ptcFile + '''", tol, sthres)''', file = f)
            print('''mfl = mfl.define_xc_(eval_xc_gga, 'GGA')''', file = f)

        elif xcStr == "PW92":
            print("mfl.xc = 'SPW92'", file = f)

        elif xcStr == "R2SCAN":
            print("mfl.xc = 'R2SCAN'", file = f)

        elif xcStr.lower() == "wb97x":
            print("mfl.xc = 'WB97X'", file = f)

        else:
            raise Exception("Invalid xcStr")

        convTol = addons["convTol"]
        print('''mfl.max_cycle = 100''', file = f)
        print(f'''mfl.conv_tol = {convTol}''', file = f)
        occupancy = addons["occupancy"]
        if occupancy:
            if occupancy["type"] is not None:
                otype = occupancy["type"].lower()
                if otype not in ["frac", "fermi"]:
                    raise ValueError(f"""Invalid occupancy type {otype} provide. Valid types: "frac" and "fermi" """)

                if otype == "frac":
                    print('''mfl = scf.addons.frac_occ(mfl)''', file = f)

                if otype == "fermi":
                    temp = float(occupancy["T"])
                    sigma = _KB_*temp
                    print(f'''mfl = scf.addons.smearing_(mfl, sigma={sigma}, method='fermi')''', file = f)

        print('''mfl.kernel()''', file = f)
        print('''np.savetxt('kseigs', mfl.mo_energy)''', file = f)
        print('''np.savetxt('ksocc', mfl.mo_occ)''', file = f)
        printOverlap = addons["printOverlap"]
        if printOverlap:
            print('''S = mol.intor("int1e_ovlp")''', file = f)
            print('''np.savetxt('SMatrix', S)''', file = f)

        printDM = addons["printDM"]
        if printDM:
            print('''dm = mfl.make_rdm1()''', file = f)
            if spin == 0:
                dmFname = "DM" + postfix
                printStr = f'''"{dmFname}"'''
                print(f'''np.savetxt({printStr}, dm)''', file = f)

            else:
                dmFname = "DM0" + postfix
                printStr = f'''"{dmFname}"'''
                print(f'''np.savetxt({printStr}, dm[0])''', file = f)
                dmFname = "DM1" + postfix
                printStr = f'''"{dmFname}"'''
                print(f'''np.savetxt({printStr}, dm[1])''', file = f)

        evalForce = addons["evalForce"]
        qbatchSize = addons["qbatch"]
        if evalForce:
            print('''g = mfl.Gradients()''', file = f)
            print('''gradnuc = g.kernel()''', file = f)
            forceFname = "forces" + postfix
            printStr = f'''"{forceFname}"'''
            print(f'''np.savetxt({printStr}, -gradnuc)''', file = f)

        evalRhoVH = addons["evalRhoVH"]
        if evalRhoVH:
            print("aovals = dft.numint.eval_ao(mol, qpts)", file = f)
            print(f"nq = qpts.shape[0]", file = f)
            if spin != 0:
                print("dmtotal = dm[0] + dm[1]", file = f)
            else:
                print("dmtotal = dm", file = f)

            print("rho = dft.numint.eval_rho(mol, aovals, dmtotal, xctype='LDA')", file = f)
            print(f"qbatch = {qbatchSize}", file = f)
            print("vh = np.zeros(nq)", file = f)
            print("for q in range(0, nq, qbatch):", file = f)
            print("\tqstart = q", file = f)
            print("\tqend = min(q+qbatch, nq)", file = f)
            print("\tvh[qstart:qend] = np.einsum('pij,ij->p', mol.intor('int1e_grids', grids=qpts[qstart:qend]), dmtotal)", file = f)
            print("A = np.hstack((qpts, qwts.reshape(nq,1), rho.reshape(nq,1), vh.reshape(nq,1)))", file = f)
            print("np.savetxt('rhovh', A)", file = f)

        print("\n", file = f)


def createScripts(systems, sysFilePostfix, header, rootDir, basisStr, xc, addons):
    for s in systems:
        words = s.split(":")
        subset = words[0]
        sysName = words[1]
        sysDir = os.path.join(rootDir, subset, sysName)
        if not os.path.exists(sysDir):
            os.makedirs(sysDir)


        scriptFilename = sysName + sysFilePostfix + ".py"
        scriptPath = os.path.join(rootDir, subset, sysName, scriptFilename)
        scriptF = open(scriptPath, "w")
        # add the header to script
        os.system("cat " + header + " > " + scriptPath)
        molTmpFname = "molTmp"
        preSCFTmpFname = "preSCFTmp"
        scfTmpFname = "scfTmp"
        getMolBuildFile(systems[s], basisStr, molTmpFname, addons["inpLengthUnit"], addons["gridLevel"])
        os.system("cat " + molTmpFname + " >> " + scriptPath)
        useGuess = addons["useGuess"]
        if useGuess["flag"]:
            getPreSCFFile(systems[s], useGuess["xcStr"], useGuess["method"], useGuess["convTol"], preSCFTmpFname)
            os.system("cat " + preSCFTmpFname + " >> " + scriptPath)

        getSCFFile(systems[s], xc, addons, sysFilePostfix, scfTmpFname)
        os.system("cat " + scfTmpFname + " >> " + scriptPath)



def run(inp, stdoutF = sys.stdout):
    sysJSON = inp["sysJSON"]
    rootDir = os.path.abspath(inp["rootDir"])
    sysFilePostfix = inp["sysFilePostfix"]
    header = inp["header"]
    outFname = inp["outFile"]
    basisStr = inp["basis"]
    xc = inp["xc"]
    xcType = xc["type"]
    startID = inp["sysRange"]["start"]
    endID = inp["sysRange"]["end"]
    slurm = inp["slurm"]
    modulesAndEnv = inp["modulesAndEnv"]
    jobscript = os.path.abspath(inp["jobscript"]+ "_" + str(startID) + "_" + str(endID))

    useGuessDefault = {"flag": False}
    useGuess = inp.get("useGuess", useGuessDefault)
    ptcFile = xc.get("ptcPath", None)
    tol = xc.get("tol", 0.0)
    sthres = xc.get("sthres", 0.0)
    xc["ptcPath"] = ptcFile
    xc["tol"] = tol
    xc["sthres"] = sthres
    convTol = inp.get("convTol", 1e-7)
    occupancyDefault = {"type": None}
    occupancy = inp.get("occupancy", occupancyDefault)
    printDM = inp.get("printDM", False)
    printOverlap = inp.get("printOverlap", False)
    evalForce = inp.get("evalForce", False)
    inpLengthUnit = inp.get("inputLengthUnit", "angs")
    gridLevel = inp.get("gridLevel", 4)
    evalRhoVH = inp.get("evalRhoVH", False)
    qbatch = inp.get("qbatch", 100000)

    print("start and end ID", startID, endID)
    # get systems based on start and end IDs
    systems = getSysData(sysJSON, startID, endID)
    addons = {"gridLevel": gridLevel,
              "inpLengthUnit": inpLengthUnit,
              "convTol": convTol,
              "occupancy": occupancy,
              "useGuess": useGuess,
              "printDM": printDM,
              "printOverlap": printOverlap,
              "evalForce": evalForce,
              "evalRhoVH": evalRhoVH,
              "qbatch": qbatch
            }
    createScripts(systems, sysFilePostfix, header, rootDir, basisStr, xc, addons = addons)

    print("Running for the following systems", file = stdoutF)
    print(list(systems.keys()), file = stdoutF)
    stdoutF.flush()

    nsys = len(systems)
    # if joscript exists create a new one using a number at the end
    if os.path.exists(jobscript):
        count = 2
        found = True
        while found:
            tmpname = jobscript + "_" + str(count)
            if os.path.exists(tmpname):
                count +=1
            else:
                jobscript = tmpname
                break

    with open(jobscript, "w") as f:
        print("#!/bin/sh", file = f)
        jobname = str(slurm["jobname"]) + "_" + str(startID) + "_" + str(endID)
        print("#SBATCH -A " + str(slurm["account"]), file = f)
        print("#SBATCH -J " + jobname, file = f)
        print("#SBATCH -o " + jobname+".out.%j", file = f)
        print("#SBATCH -e " + jobname+".err.%j", file = f)
        if slurm["partition"] is not None and slurm["partition"] != "":
            print("#SBATCH -p " + str(slurm["partition"]), file = f)

        if slurm["queue"] is not None and slurm["queue"] != "":
            print("#SBATCH -q " + str(slurm["queue"]), file = f)

        print("#SBATCH -t " + str(slurm["t"]), file = f)
        print("#SBATCH --nodes=" + str(slurm["nodesPerSys"]*nsys), file = f)
        for e in slurm["extras"]:
            print("#SBATCH " + e, file=f)

        print("#SBATCH --mail-type=BEGIN,END", file = f)
        print("#SBATCH --mail-user=" + str(slurm["email"]), file = f)
        print("\n", file = f)

        with open(modulesAndEnv) as f2:
            lines = f2.readlines()
            for line in lines:
                print(line.strip(), file = f)

        print("\n", file = f)
        print("export OMP_NUM_THREADS="+str(slurm["threads"]), file=f)
        for s in systems:
            words = s.split(":")
            subset = words[0]
            sysName = words[1]
            sysDir = os.path.join(rootDir, subset, sysName)
            if ptcFile is not None:
                ptcFilePath = os.path.join(rootDir, ptcFile)
                # copy ptc file to sysDir
                os.system("cp " + ptcFile + " " + sysDir)

            pyscfScript = sysName + sysFilePostfix + ".py"
            print("cd " + sysDir, file = f)
            nodesPerSys = "--nodes=" + str(slurm["nodesPerSys"])
            tasksPerNode = ""
            cpusPerTask = ""
            if slurm["tasksPerNode"] is not None:
                tasksPerNode = "--ntasks-per-node=" + str(slurm["tasksPerNode"])
                cpusPerTask = "--cpus-per-task=" + str(slurm["cpusPerTask"])
            threads = "OMP_NUM_THREADS=" + str(slurm["threads"])
            #srunLine =  threads + " srun -u " + nodesPerSys + " python " +\
            #           pyscfScript + " &> " + outFname + " &"
            srunLine = "srun -u " + nodesPerSys + " " + tasksPerNode + " " + cpusPerTask + " python " + pyscfScript + " &> " + outFname + " &"
            print(srunLine, file = f)

        print("wait", file = f)

    os.system("sbatch " + jobscript)

if __name__ == "__main__":
    correctUse = True
    usageMsg = """Usage: `python CreatePyscfScript.py inp.json [stdout_file]`,"""\
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
