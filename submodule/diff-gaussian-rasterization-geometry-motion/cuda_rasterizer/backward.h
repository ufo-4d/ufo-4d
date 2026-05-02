/*
* Copyright (C) 2023, Inria
* GRAPHDECO research group, https://team.inria.fr/graphdeco
* All rights reserved.
*
* This software is free for non-commercial, research and evaluation use 
* under the terms of the LICENSE.md file.
*
* For inquiries contact  george.drettakis@inria.fr
*/

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace BACKWARD
{
	void render(
		const dim3 grid, const dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float* bg_color,
		const float2* means2D,
		const float4* conic_opacity,
		const float* colors,
		const float* final_Ts,
		const uint32_t* n_contrib,
		const float* dL_dpixels,
		float3* dL_dmean2D,
		float4* dL_dconic2D,
		float* dL_dopacity,
		float* dL_dcolors,
		const float3* points, // from geomState
		const float* dL_dpixels_point,  // Backproped gradient from the loss
		float3* dL_dgeom_point,  // Gradient to be calculated for geomState points
		const float3* motions, // from geomState
		const float* dL_dpixels_motion,  // Backproped gradient from the loss
		float3* dL_dgeom_motion  // Gradient to be calculated for geomState motions
		);
		// TODO why const?

	void preprocess(
		int P, int D, int M,
		const float3* means,  //means3D
		const int* radii,
		const float* shs,
		const bool* clamped,
		const glm::vec3* scales,
		const glm::vec4* rotations,
		const float scale_modifier,
		const float* cov3Ds,
		const float* view,
		const float* proj,
		const float* proj_raw,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		const glm::vec3* campos,
		const float3* dL_dmean2D,
		const float* dL_dconics,
		glm::vec3* dL_dmeans,
		float* dL_dcolor,
		float* dL_dcov3D,
		float* dL_dsh,
		glm::vec3* dL_dscale,
		glm::vec4* dL_drot,
		float* dL_dtau,
		float3* dL_dgeom_point,
		float3* dL_dgeom_motion,
		glm::vec3* dL_dmotion
);
}

#endif