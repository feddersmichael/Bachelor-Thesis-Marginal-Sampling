#include "amici/sundials_matrix_wrapper.h"
#include "sundials/sundials_types.h"

#include <array>
#include <algorithm>

namespace amici {
namespace model_conversion_reaction_0 {

static constexpr std::array<sunindextype, 3> dxdotdw_colptrs_conversion_reaction_0_ = {
    0, 2, 4
};

void dxdotdw_colptrs_conversion_reaction_0(SUNMatrixWrapper &dxdotdw){
    dxdotdw.set_indexptrs(gsl::make_span(dxdotdw_colptrs_conversion_reaction_0_));
}
} // namespace amici
} // namespace model_conversion_reaction_0