#include "amici/symbolic_functions.h"
#include "amici/defines.h"
#include "sundials/sundials_types.h"

#include "p.h"
#include "k.h"
#include "w.h"
#include "x.h"

namespace amici {
namespace model_conversion_reaction_0 {

void y_conversion_reaction_0(realtype *y, const realtype t, const realtype *x, const realtype *p, const realtype *k, const realtype *h, const realtype *w){
    y[0] = B + observableParameter1_obs_b;
}

} // namespace amici
} // namespace model_conversion_reaction_0