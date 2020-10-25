#include "amici/sundials_matrix_wrapper.h"
#include "sundials/sundials_types.h"

#include <array>
#include <algorithm>

namespace amici {
namespace model_conversion_reaction_0 {

static constexpr std::array<std::array<sunindextype, 2>, 1> dJydy_colptrs_conversion_reaction_0_ = {{
    {0, 1}, 
}};

void dJydy_colptrs_conversion_reaction_0(SUNMatrixWrapper &dJydy, int index){
    dJydy.set_indexptrs(gsl::make_span(dJydy_colptrs_conversion_reaction_0_[index]));
}
} // namespace amici
} // namespace model_conversion_reaction_0