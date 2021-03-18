// Copyright 2021 The Google Research Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef SCANN_DISTANCE_MEASURES_ONE_TO_ONE_DOT_PRODUCT_AVX1_H_
#define SCANN_DISTANCE_MEASURES_ONE_TO_ONE_DOT_PRODUCT_AVX1_H_
#ifdef __x86_64__

#include "scann/data_format/datapoint.h"
#include "scann/utils/intrinsics/attributes.h"

namespace research_scann {
namespace dp_internal {

SCANN_AVX1_OUTLINE double DenseDotProductAvx1(const DatapointPtr<int8_t>& a,
                                              const DatapointPtr<float>& b);

SCANN_AVX1_OUTLINE double DenseDotProductAvx1(const DatapointPtr<int8_t>& a,
                                              const DatapointPtr<float>& b,
                                              const DatapointPtr<float>& c);

SCANN_AVX1_OUTLINE double DenseDotProductAvx1(const DatapointPtr<int8_t>& a,
                                              const DatapointPtr<int8_t>& b,
                                              const DatapointPtr<float>& c);

}  // namespace dp_internal
}  // namespace research_scann

#endif
#endif
