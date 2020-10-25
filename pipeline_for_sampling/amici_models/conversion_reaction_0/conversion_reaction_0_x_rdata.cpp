#include "amici/symbolic_functions.h"
#include "amici/defines.h"
#include "sundials/sundials_types.h"

#include "tcl.h"
#include "x.h"

namespace amici {
namespace model_conversion_reaction_0 {

void x_rdata_conversion_reaction_0(realtype *x_rdata, const realtype *x, const realtype *tcl){
    x_rdata[0] = A;
    x_rdata[1] = B;
}

} // namespace amici
} // namespace model_conversion_reaction_0