#include "amici/symbolic_functions.h"
#include "amici/defines.h"
#include "sundials/sundials_types.h"

#include "y.h"
#include "my.h"
#include "p.h"
#include "k.h"
#include "sigmay.h"

namespace amici {
namespace model_conversion_reaction_0 {

void dJydsigmay_conversion_reaction_0(realtype *dJydsigmay, const int iy, const realtype *p, const realtype *k, const realtype *y, const realtype *sigmay, const realtype *my){
    switch(iy) {
        case 0:
            dJydsigmay[0] = 1.0/sigmaobs_b - 1.0*std::pow(-std::log(mobs_b) + std::log(obs_b), 2)/std::pow(sigmaobs_b, 3);
            break;
    }
}

} // namespace amici
} // namespace model_conversion_reaction_0