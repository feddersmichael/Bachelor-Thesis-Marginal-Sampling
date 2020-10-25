#include "amici/symbolic_functions.h"
#include "amici/defines.h"
#include "sundials/sundials_types.h"

#include "p.h"
#include "k.h"

namespace amici {
namespace model_conversion_reaction_0 {

void dsigmaydp_conversion_reaction_0(realtype *dsigmaydp, const realtype t, const realtype *p, const realtype *k, const int ip){
    switch(ip) {
        case 3:
            dsigmaydp[0] = 1;
            break;
    }
}

} // namespace amici
} // namespace model_conversion_reaction_0