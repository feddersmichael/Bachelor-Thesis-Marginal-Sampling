#include "amici/model.h"
#include "wrapfunctions.h"
#include "conversion_reaction_0.h"

namespace amici {
namespace generic_model {

std::unique_ptr<amici::Model> getModel() {
    return std::unique_ptr<amici::Model>(
        new amici::model_conversion_reaction_0::Model_conversion_reaction_0());
}


} // namespace generic_model

} // namespace amici
