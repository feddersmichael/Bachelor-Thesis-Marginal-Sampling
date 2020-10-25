#include "amici/symbolic_functions.h"
#include "amici/defines.h"
#include "sundials/sundials_types.h"

#include "p.h"
#include "k.h"
#include "sigmay.h"

namespace amici {
namespace model_conversion_reaction_0 {

void sigmay_conversion_reaction_0(realtype *sigmay, const realtype t, const realtype *p, const realtype *k){
    sigmaobs_b = noiseParameter1_obs_b;  // sigmay[0]
}

} // namespace amici
} // namespace model_conversion_reaction_0