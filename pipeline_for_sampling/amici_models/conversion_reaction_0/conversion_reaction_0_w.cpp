#include "amici/symbolic_functions.h"
#include "amici/defines.h"
#include "sundials/sundials_types.h"

#include "tcl.h"
#include "p.h"
#include "k.h"
#include "w.h"
#include "x.h"

namespace amici {
namespace model_conversion_reaction_0 {

void w_conversion_reaction_0(realtype *w, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *tcl){
    flux_r0 = 1.0*A*k1;  // w[0]
    flux_r1 = 1.0*B*k2;  // w[1]
}

} // namespace amici
} // namespace model_conversion_reaction_0