#
#  runScheme.py
#  symmetrizeAndInterpolate.py
#
#  Created by VISHAL SUBRAMANIAN on 6/2/26.
#

from symmetrizeAndInterpolate import run_full_density_pipeline

outputs = run_full_density_pipeline(
    xsf_path="dmc_density_dt01.s002.SpinDensity_u+d.xsf",
    export_base="xsf_out",
    quad_in="densityQuadData_lda.txt",
    nelec_ref=256.0
)
print(outputs)

