#include "amici/symbolic_functions.h"
#include "amici/defines.h"
#include "sundials/sundials_types.h"

#include "y.h"
#include "my.h"
#include "p.h"
#include "k.h"
#include "sigmay.h"
#include "dJydy.h"

namespace amici {
namespace model_conversion_reaction_0 {

void dJydy_conversion_reaction_0(realtype *dJydy, const int iy, const realtype *p, const realtype *k, const realtype *y, const realtype *sigmay, const realtype *my){
    switch(iy) {
        case 0:
            dJydy[0] = (-1.0*std::log(mobs_b) + 1.0*std::log(obs_b))/(obs_b*std::pow(sigmaobs_b, 2));
            break;
    }
}

} // namespace amici
} // namespace model_conversion_reaction_0