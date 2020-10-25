#include "amici/symbolic_functions.h"
#include "amici/defines.h"
#include "sundials/sundials_types.h"

#include "x_rdata.h"

namespace amici {
namespace model_conversion_reaction_0 {

void x_solver_conversion_reaction_0(realtype *x_solver, const realtype *x_rdata){
    x_solver[0] = A;
    x_solver[1] = B;
}

} // namespace amici
} // namespace model_conversion_reaction_0