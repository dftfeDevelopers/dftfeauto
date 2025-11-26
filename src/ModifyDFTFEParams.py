import sys
import os
import json

def getVal(x):
    if isinstance(x, bool):
        return str(x).lower()

    else:
        return x


def __getSubSecLines__(subSec, name, params, indent = 0):
    ## sort the subsections to arrange them in the order of nesting
    whiteSp = ' '*indent
    lines = [whiteSp + "subsection " + name]
    for p in subSec:
        dd = subSec[p]
        if not isinstance(dd, dict):
            cmd = dd[0]
            val = dd[1]
            if p in params:
                val = params[p]

            val = getVal(val)
            line = whiteSp + ' ' + cmd + " = " + str(val)
            lines = lines + [line]

    for k in subSec:
        dd = subSec[k]
        if isinstance(dd, dict):
            lines = lines + __getSubSecLines__(dd, k, params, indent+1)

    lines = lines + [whiteSp + "end"]
    return lines


def __getNewParamsFile__(params, outPath, keywordsDict):
    ff = open(outPath, "w")
    keys = list(keywordsDict.keys())
    for kk in keys:
        tmpdict = keywordsDict[kk]
        if kk == "Global":
            for p in tmpdict:
                cmd = tmpdict[p][0]
                val = tmpdict[p][1]
                if p in params:
                    val = params[p]

                val = getVal(val)
                print(cmd + " = " + str(val), file = ff)

        else:
            lines = __getSubSecLines__(tmpdict, kk, params, indent=0)
            for l in lines:
                print(l, file = ff)

    ff.close()

def modifyParamsFile(params, outPath, mode):
    keywordsDict = None
    allDict = None
    with open("DFTFEKeyCmds.json") as ff:
        allDict = json.load(ff)

    if mode.lower() == "gs":
        keywordsDict = allDict["gs"]

    elif mode.lower() == "inv":
        keywordsDict = allDict["inv"]

    else:
        raise ValueError(f'''Invalid mode + {mode} provided. Valid modes are: "gs" and "inv"''')

    __getNewParamsFile__(params, outPath, keywordsDict)

