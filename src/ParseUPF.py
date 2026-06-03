"""
@author Bikash Kanungo 
"""

"""
@brief Handles parsing of UPF (unified pseudopotential format) files.
       The UPF file can, typically, be parsed as an XML file.
       The current parsing implementation is based on xml.dom.minidom library
       from the standard Python library. xml.dom.minidom is simple
       to use but probably not as powerful or efficient, especially
       for large XML files. DOM (document object file) is one of the two
       main XML algorithms---DOM and SAX (Simple API for XML)---used to
       parse and modify XML files. While DOM loads the entire XML into RAM,
       SAX reads the XML as needed. Since most UPF files are lightweight,
       DOM, and to some extent DOM offered via xml.dom.minidom, should
       do the job fast enough. In case parsing the UPF via xml.dom.minidom
       becomes a choke point, one should consider switching to SAX or
       use a better Python standard library like xml.etree.ElementTree or
       any third-party library.
"""

import xml.dom.minidom as xmldom


def removeWhitespace(node):
    """
    @brief Takes an XML node and removes any trailing whitesspaces from any text node.
    This is helpful when using xml.dom.minidom where the newline characters
    and leading indentation are captured as separate tree elements, which is
    what the specification requires. Some parsers let you ignore these,
    but not the Python one. But we can collapse whitespace in such nodes manually.
    The following code recursively removes those unwanted whitespaces given a starting node.

    @param[in,out] node Node from which to start recursively removing unwanted whitespaces
    """
    if node.nodeType == xmldom.Node.TEXT_NODE:
        if node.nodeValue.strip() == "":
            node.nodeValue = ""
    for child in node.childNodes:
        removeWhitespace(child)


def parseHeader(root):
    """
    @brief Parses the header of the UPF file (i.e., attributes under PP_HEADER node)
           and returns it as a dictionary.
    @param[in] root Root node of the UPF
    @returns Dictionary containing the attributes under PP_HEADER node
    """
    header = root.getElementsByTagName("PP_HEADER")[0]
    a = dict(header.attributes.items())
    # remove trailing whitespaces
    for k in a:
        a[k] = a[k].strip()

    return a


def parseMesh(root):
    """
    @brief Parses and returns the radial mesh of the UPF file (i.e., PP_R and PP_RAB nodes)
    @param[in] root Root node of the UPF
    @returns A tuple of rVals, rabVals, nr, nrab, where
             rVals is a list containing the radial points in the mesh
             rabVals is a list of quadrature weights for the 1D radial integration
             nr is the number of radial points as read from the attributes of PP_R
             nrab is the number of quadrature weights as read from the attributes of PP_RAB
    """
    mesh = root.getElementsByTagName("PP_MESH")[0]
    r = mesh.getElementsByTagName("PP_R")[0]
    rAttrs = dict(r.attributes.items())
    nr = int(rAttrs["size"].strip())
    rValsStr = (r.childNodes[0].nodeValue).split()
    rVals = [float(x.strip()) for x in rValsStr]
    rab = mesh.getElementsByTagName("PP_RAB")[0]
    rabAttrs = dict(rab.attributes.items())
    nrab = int(rabAttrs["size"].strip())
    rabValsStr = (rab.childNodes[0].nodeValue).split()
    rabVals = [float(x.strip()) for x in rabValsStr]
    return rVals, rabVals, nr, nrab


def parseVloc(root):
    """
    @brief Parses and returns the local part of the pseudopoential (i.e., PP_LOCAL node)
    @param[in] root Root node of the UPF
    @returns A tuple of vals, nr, where
             vals is a list containing the local pseudopotentials on different radial points.
             nr is the number of radial points as read from the attributes of PP_LOCAL
    """
    node = root.getElementsByTagName("PP_LOCAL")[0]
    attrs = dict(node.attributes.items())
    n = int(attrs["size"].strip())
    valsStr = (node.childNodes[0].nodeValue).split()
    vals = [float(x.strip()) for x in valsStr]
    return vals, n


def parseVNonloc(root, lmax):
    """
    @brief Parses and returns the nonlocal projectors (i.e., PP_NONLOCAL node).

    @param[in] root Root node of the UPF
    @param[in] lmax Maximum angular momentum specified in the UPF file
    @returns A tuple proj, dij, where
             proj is a list of length lmax, such that for a give angular momentum 0<= l <= lmax ,
             proj[l] = [dict1, dict2, ...] (i.e., list of dictionary),
             where the number of dictionaries is the number of projectors for the l angular momentum.
             Each of the dictionary has two keys:
             (1) "vals" which store the radial values of the projector; and
             (2) "attrs" which stores the attributes of the projector as a dictionary
             dij is a dictionary that stores the coupling matrix (DIJ). It has two keys
             (1) "val" which stores the DIJ values as a list
             (2) "size" which stores the number of entries in DIJ as read from the attributes of PP_DIJ
    """
    parent = root.getElementsByTagName("PP_NONLOCAL")[0]
    proj = [[] for l in range(0, lmax + 1)]
    dij = {}
    for node in parent.childNodes:
        if True:  # node.localName != 'None':
            if "PP_BETA" in node.localName:
                attrs = dict(node.attributes.items())
                # remove trailing whitespaces
                for k in attrs:
                    attrs[k] = attrs[k].strip()

                l = int(attrs["angular_momentum"])

                if l > lmax:
                    raise Exception("Angular momentum greater than the supplied lmax")
                if l < 0:
                    raise Exception("Negative value of angular momentum detected")

                rc = float(attrs["cutoff_radius"])
                n = int(attrs["size"])
                meta_data = {"angular_momentum": l, "size": n, "cutoff_radius": rc}
                valsStr = (node.childNodes[0].nodeValue).split()
                vals = [float(x.strip()) for x in valsStr]
                d = {"vals": vals, "attrs": meta_data}
                proj[l].append(d)

            if "PP_DIJ" in node.localName:
                attrs = dict(node.attributes.items())
                # remove trailing whitespaces
                for k in attrs:
                    attrs[k] = attrs[k].strip()

                valsStr = (node.childNodes[0].nodeValue).split()
                vals = [float(x.strip()) for x in valsStr]
                dij = {"vals": vals, "size": int(attrs["size"])}

    return proj, dij


def parseDensity(root):
    """
    @brief Parses and returns the radial density of the atom (i.e., PP_RHOATOM node)
    @param[in] root Root node of the UPF
    @returns A tuple of vals, n, where
             vals is a list containing the values of the density on different radial points.
             nr is the number of radial points as read from the attributes of PP_RHOATOM
    """
    node = root.getElementsByTagName("PP_RHOATOM")[0]
    attrs = dict(node.attributes.items())
    n = int(attrs["size"].strip())
    valsStr = (node.childNodes[0].nodeValue).split()
    vals = [float(x.strip()) for x in valsStr]
    return vals, n

def parseCoreCorrection(root):
    """
    @brief Parses and returns the radial density of the atom (i.e., PP_RHOATOM node)
    @param[in] root Root node of the UPF
    @returns A tuple of vals, n, where
             vals is a list containing the values of the density on different radial points.
             nr is the number of radial points as read from the attributes of PP_RHOATOM
    """
    node = root.getElementsByTagName("PP_NLCC")[0]
    attrs = dict(node.attributes.items())
    n = int(attrs["size"].strip())
    valsStr = (node.childNodes[0].nodeValue).split()
    vals = [float(x.strip()) for x in valsStr]
    return vals, n

def char_to_bool(char_str):
    if char_str.upper() == 'T':
        return True
    elif char_str.upper() == 'F':
        return False
    else:
        raise ValueError("Input must be 'T' or 'F'")

class ParseUPF:
    """
    @brief Class to parse and store relevant information of from a given UPF file
    """

    def __init__(self, filename):
        """
        @brief Constructor
        @param[in] filename Name of the UPF file. Must have the .upf or .UPF extension
        """
        # list of data members
        self.filename = filename
        self.header = None  # stores the PP_HEADER attributes as a dictionary
        self.lmax = None  # the max angular momentum
        self.zvalence = None  # valence charge of the nuclei
        self.symbol = None  # symbol of the chemical element
        self.rhoCutoff = None  # cutoff radius for the density
        self.nr = None  # number of radial points in the 1D radial mesh
        self.r = None  # list of radial points at which the fields are defined
        self.rab = None  # quadrature weights for the radial points in the mesh
        self.vloc = None  # local part of the pseudopotential
        # stores the nonlocal projectors. The layout is that for each
        # angular momentum l, self.proj[l] = [dict1, dict2, ...] (i.e., list of dictionary),
        # where the length is the number of projectors for the l angular momentum.
        # Each of the dictionary has two keys:
        # (1) "vals" which store the radial values of the projector; and
        # (2) "attrs" which stores the attributes of the projector as a dictionary
        self.proj = None
        # stores the coupling matrix (DIJ) as a list
        self.dij = None
        self.rho = None  # density of the atom
        self.coreCorrectionRho = None

        ext = filename[-3:]
        if ext.lower() != "upf":
            raise Exception("File " + filename + " is not a upf file")

        doc = xmldom.parse(filename)
        if doc is None:
            raise Exception("Cannot parse the upf document in " + filename)

        # In xml.dom.minidom the newline characters and leading indentation
        # are captured as separate tree elements, which is what the
        # specification requires.
        # Some parsers let you ignore these, but not the Python one.
        # What you can do, however, is collapse whitespace in such nodes manually.
        # The following code removes those unwanted whitespaces
        removeWhitespace(doc)
        doc.normalize()

        root = doc.documentElement
        if root is None:
            raise Exception("Cannot find root element in the upf document " + filename)

        self.header = parseHeader(root)
        self.lmax = int(self.header["l_max"])
        self.zvalence = float(self.header["z_valence"])
        self.symbol = self.header["element"]
        self.nr = int(self.header["mesh_size"])
        self.rhoCutoff = float(self.header["rho_cutoff"])
        self.numProj = int(self.header["number_of_proj"])
        self.isCoreCorrectionPresent = char_to_bool(self.header["core_correction"])

        # parse the radial mesh
        self.r, self.rab, nr, nrab = parseMesh(root)
        if nr != self.nr or nrab != self.nr:
            raise Exception(
                """Mismatch in size of PP_R or PP_RAB and """
                """mesh_size in header of the upf document """ + filename
            )

        # parse the local potential
        self.vloc, n = parseVloc(root)
        if n != self.nr:
            raise Exception(
                """Mismatch in size of PP_LOCAL and """
                """mesh_size in header of the upf document """ + filename
            )
        ##print(self.vloc)

        if self.numProj != 0:
            # parse the nonlocal projectors
            self.proj, dij = parseVNonloc(root, self.lmax)
            for l in range(len(self.proj)):
                for d in self.proj[l]:
                    n = d["attrs"]["size"]
                    if n != self.nr:
                        raise Exception(
                            """Mismatch in size of PP_NONLOCAL and """
                            """ mesh_size in header of the upf document """ + filename
                        )

            if dij["size"] != len(dij["vals"]):
                raise Exception(
                    """Mismatch in the number of values and size """
                    """specified in PP_DIJ in the upf document """ + filename
                )

            self.dij = dij["vals"]
            # print(self.proj)
            # print(self.dij)
        else:
            self.proj = []
            self.dij = []
            self.lmax = -1

        # parse the density
        self.rho, n = parseDensity(root)
        if n != self.nr:
            raise Exception(
                """Mismatch in size of PP_RHOATOM and """
                """mesh_size in header of the upf document """ + filename
            )
        # print(self.rho)

        if self.isCoreCorrectionPresent:
            # parse the core correction to density
            self.coreCorrectionRho, n = parseCoreCorrection(root)
            if n != self.nr:
                raise Exception(
                    """Mismatch in size of PP_NLCC and """
                    """mesh_size in header of the upf document """ + filename
                )

    def getRadialMesh(self):
        """
        @brief Return radial mesh of the UPF file (i.e., points under PP_R node)
        @returns A list containing the radial points in the mesh
        """
        return self.r

    def getRadialQuadWeights(self):
        """
        @brief Return quadrature weights for the 1D radial mesh of the UPF file (i.e., weights PP_RAB node)
        @returns A list containing the quadrature weights
        """
        return self.rab

    def getNRadialPoints(self):
        """
        @brief Return number of points in the radial mesh (i.e., mesh_size attribute in PP_HEADER)
        @returns Number of points in radial mesh
        """
        return self.nr

    def getVloc(self):
        """
        @brief Returns the local part of the pseudopotential (i.e., values under PP_LOCAL node)
        @returns A list containing the values of the local part of the pseudopotential
        on the radial mesh provided in the UPF file (under PP_R node inside PP_MESH node)
        """
        return self.vloc

    def getNonlocProjectors(self):
        """
        @brief Returns the nonlocal part of the pseudopotential (i.e., PP_BETA projectors under PP_NONLOCAL node)
        @returns A list containing the values and other metadata of the nonlocal part of the pseudopotential
                 The list (say proj) is of length lmax (as provided in the UPF file),
                 such that for a give angular momentum 0<= l <= lmax ,
                 proj[l] = [dict1, dict2, ...] (i.e., list of dictionary),
                 where the number of dictionaries is the number of projectors for the l angular momentum.
                 Each of the dictionary has two keys:
                 (1) "vals" which store the radial values of the projector; and
                 (2) "attrs" which stores the attributes of the projector as a dictionary
        """
        return self.proj

    def getDIJ(self):
        """
        @brief Returns the coupling matrix (DIJ) (i.e., PP_DIJ values under PP_NONLOCAL node)
        @returns A list containing the DIJ values
        """
        return self.dij

    def getRho(self):
        """
        @brief Returns the radial density of the atom (i.e., values under PP_RHOATOM node)
        @returns A list containing the values of the density of the atom
        on the radial mesh provided in the UPF file (under PP_R node inside PP_MESH node)
        """
        return self.rho

    def getCoreCorrectionRho(self):
        """
        @brief Returns the radial core correction density of the atom (i.e., values under PP_NLCC node)
        @returns A list containing the values of the density of the atom
        on the radial mesh provided in the UPF file (under PP_R node inside PP_MESH node)
        """
        return self.coreCorrectionRho

    def isCoreCorrection(self):
        return self.isCoreCorrectionPresent

    def getLMax(self):
        """
        @brief Returns the maximum angular momentum of the pseudopotential (l_max attribute under PP_HEADER node)
        @returns Maximum angular momentum
        """
        return self.lmax

    def getZValence(self):
        """
        @brief Returns valence atomic number the atom (z_valence attribute under PP_HEADER node)
        @returns Valence atomic number (float)
        """
        return self.zvalence

    def getSymbol(self):
        """
        @brief Returns chemical symbol of the atom (element attribute under PP_HEADER node)
        @returns String containing the chemical symbol of the atom
        """
        return self.symbol

    def getRhoCutoff(self):
        """
        @brief Returns cutoff radius for the density
        @returns Cutoff radius (float)
        """
        return self.rhoCutoff

