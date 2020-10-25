#include "amici/sundials_matrix_wrapper.h"
#include "sundials/sundials_types.h"

#include <array>
#include <algorithm>

namespace amici {
namespace model_conversion_reaction_0 {

static constexpr std::array<sunindextype, 5> dwdp_colptrs_conversion_reaction_0_ = {
    0, 1, 2, 2, 2
};

void dwdp_colptrs_conversion_reaction_0(SUNMatrixWrapper &dwdp){
    dwdp.set_indexptrs(gsl::make_span(dwdp_colptrs_conversion_reaction_0_));
}
} // namespace amici
} // namespace model_conversion_reaction_0