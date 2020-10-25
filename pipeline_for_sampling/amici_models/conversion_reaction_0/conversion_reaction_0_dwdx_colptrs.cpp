#include "amici/sundials_matrix_wrapper.h"
#include "sundials/sundials_types.h"

#include <array>
#include <algorithm>

namespace amici {
namespace model_conversion_reaction_0 {

static constexpr std::array<sunindextype, 3> dwdx_colptrs_conversion_reaction_0_ = {
    0, 1, 2
};

void dwdx_colptrs_conversion_reaction_0(SUNMatrixWrapper &dwdx){
    dwdx.set_indexptrs(gsl::make_span(dwdx_colptrs_conversion_reaction_0_));
}
} // namespace amici
} // namespace model_conversion_reaction_0