#include "amici/sundials_matrix_wrapper.h"
#include "sundials/sundials_types.h"

#include <array>
#include <algorithm>

namespace amici {
namespace model_conversion_reaction_0 {

static constexpr std::array<sunindextype, 4> dxdotdw_rowvals_conversion_reaction_0_ = {
    0, 1, 0, 1
};

void dxdotdw_rowvals_conversion_reaction_0(SUNMatrixWrapper &dxdotdw){
    dxdotdw.set_indexvals(gsl::make_span(dxdotdw_rowvals_conversion_reaction_0_));
}
} // namespace amici
} // namespace model_conversion_reaction_0