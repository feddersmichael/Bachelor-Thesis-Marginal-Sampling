#include "amici/symbolic_functions.h"
#include "amici/defines.h"
#include "sundials/sundials_types.h"

#include "tcl.h"
#include "p.h"
#include "k.h"
#include "w.h"
#include "x.h"
#include "dwdx.h"

namespace amici {
namespace model_conversion_reaction_0 {

void dwdx_conversion_reaction_0(realtype *dwdx, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const realtype *tcl){
    dflux_r0_dA = 1.0*k1;  // dwdx[0]
    dflux_r1_dB = 1.0*k2;  // dwdx[1]
}

} // namespace amici
} // namespace model_conversion_reaction_0