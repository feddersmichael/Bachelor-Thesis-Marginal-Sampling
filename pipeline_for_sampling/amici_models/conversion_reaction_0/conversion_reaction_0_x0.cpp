#include "amici/symbolic_functions.h"
#include "amici/defines.h"
#include "sundials/sundials_types.h"

#include "p.h"
#include "k.h"

namespace amici {
namespace model_conversion_reaction_0 {

void x0_conversion_reaction_0(realtype *x0, const realtype t, const realtype *p, const realtype *k){
    x0[0] = 1.0;
    x0[1] = 0.01;
}

} // namespace amici
} // namespace model_conversion_reaction_0