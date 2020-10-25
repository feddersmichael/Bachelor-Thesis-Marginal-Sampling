#include "amici/symbolic_functions.h"
#include "amici/defines.h"
#include "sundials/sundials_types.h"

#include "tcl.h"
#include "p.h"
#include "k.h"
#include "w.h"
#include "x.h"
#include "dwdp.h"

namespace amici {
namespace model_conversion_reaction_0 {

void dwdp_conversion_reaction_0(realtype *dwdp, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w, const realtype *tcl, const realtype *dtcldp){
    dflux_r0_dk1 = 1.0*A;  // dwdp[0]
    dflux_r1_dk2 = 1.0*B;  // dwdp[1]
}

} // namespace amici
} // namespace model_conversion_reaction_0