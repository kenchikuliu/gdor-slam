/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 * 
 * This file is Derivative Works of Gaussian Splatting,
 * created by Longwei Li, Huajian Huang, Hui Cheng and Sai-Kit Yeung in 2023,
 * as part of Photo-SLAM.
 */

#include "include/gaussian_renderer.h"

/**
 * @brief 
 * 
 * @return std::tuple<render, viewspace_points, visibility_filter, radii>, which are all `torch::Tensor`
 */
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
GaussianRenderer::render(
    std::shared_ptr<GaussianKeyframe> viewpoint_camera,
    int image_height,
    int image_width,
    std::shared_ptr<GaussianModel> pc,
    GaussianPipelineParams& pipe,
    torch::Tensor& bg_color,
    torch::Tensor& override_color,
    float scaling_modifier,
    bool use_override_color)
{
    /* Render the scene. 

       Background tensor (bg_color) must be on GPU!
     */

    // Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    auto screenspace_points = torch::zeros_like(pc->getXYZ(),
        torch::TensorOptions().dtype(pc->getXYZ().dtype()).requires_grad(true).device(torch::kCUDA));
    try {
        screenspace_points.retain_grad();
    }
    catch (const std::exception& e) {
        ; // pass
    }

    // Set up rasterization configuration
    float tanfovx = std::tan(viewpoint_camera->FoVx_ * 0.5f);
    float tanfovy = std::tan(viewpoint_camera->FoVy_ * 0.5f);

    GaussianRasterizationSettings raster_settings(
        image_height,
        image_width,
        tanfovx,
        tanfovy,
        bg_color,
        scaling_modifier,
        viewpoint_camera->world_view_transform_,
        viewpoint_camera->full_proj_transform_,
        pc->active_sh_degree_,
        viewpoint_camera->camera_center_,
        false
    );

    GaussianRasterizer rasterizer(raster_settings);

    auto means3D = pc->getXYZ();
    auto means2D = screenspace_points;
    auto opacity = pc->getOpacityActivation();

    /* If precomputed 3d covariance is provided, use it. If not, then it will be computed from
       scaling / rotation by the rasterizer. 
     */
    bool has_scales = false,
         has_rotations = false,
         has_cov3D_precomp = false;
    torch::Tensor scales,
                  rotations,
                  cov3D_precomp;
    if (pipe.compute_cov3D_) {
        cov3D_precomp = pc->getCovarianceActivation();
        has_cov3D_precomp = true;
    }
    else {
        scales = pc->getScalingActivation();
        rotations = pc->getRotationActivation();
        has_scales = true;
        has_rotations = true;
    }

    /* If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
       from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
     */
    bool has_shs = false,
         has_color_precomp = false;
    torch::Tensor shs,
                  colors_precomp;
    if (use_override_color) {
        colors_precomp = override_color;
        has_color_precomp = true;
    }
    else {
        if (pipe.convert_SHs_) {
            int max_sh_degree = pc->max_sh_degree_ + 1;
            torch::Tensor shs_view = pc->getFeatures().transpose(1, 2).view({-1, 3, max_sh_degree * max_sh_degree});
            torch::Tensor dir_pp = (pc->getXYZ() - viewpoint_camera->camera_center_.repeat({pc->getFeatures().size(0), 1}));
            auto dir_pp_normalized = dir_pp / torch::frobenius_norm(dir_pp, /*dim=*/{1}, /*keepdim=*/true);
            auto sh2rgb = sh_utils::eval_sh(pc->active_sh_degree_, shs_view, dir_pp_normalized);
            colors_precomp = torch::clamp_min(sh2rgb + 0.5, 0.0);
            has_color_precomp = true;
        }
        else {
            shs = pc->getFeatures();
            has_shs = true;
        }
    }

    // Rasterize visible Gaussians to image, obtain their radii (on screen).
    auto rasterizer_result = rasterizer.forward(
        means3D,
        means2D,
        opacity,
        has_shs,
        has_color_precomp,
        has_scales,
        has_rotations,
        has_cov3D_precomp,
        shs,
        colors_precomp,
        scales,
        rotations,
        cov3D_precomp
    );
    auto rendered_image = std::get<0>(rasterizer_result);
    auto radii = std::get<1>(rasterizer_result);
    // auto rendered_depth = std::get<2>(rasterizer_result);  // Available but not returned for backward compatibility

    /* Those Gaussians that were frustum culled or had a radius of 0 were not visible.
       They will be excluded from value updates used in the splitting criteria.
     */
    return std::make_tuple(
        rendered_image,     /*render*/
        screenspace_points, /*viewspace_points*/
        radii > 0,          /*visibility_filter*/
        radii               /*radii*/
    );
}

std::tuple<torch::Tensor, torch::Tensor>
GaussianRenderer::renderDepthWithOpacity(
    std::shared_ptr<GaussianKeyframe> viewpoint_camera,
    int image_height,
    int image_width,
    std::shared_ptr<GaussianModel> pc,
    float scaling_modifier)
{
    // Set up rasterization configuration
    float tanfovx = std::tan(viewpoint_camera->FoVx_ * 0.5f);
    float tanfovy = std::tan(viewpoint_camera->FoVy_ * 0.5f);

    // Use zero background for depth rendering
    torch::Tensor bg_color = torch::zeros({3}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    GaussianRasterizationSettings raster_settings(
        image_height,
        image_width,
        tanfovx,
        tanfovy,
        bg_color,
        scaling_modifier,
        viewpoint_camera->world_view_transform_,
        viewpoint_camera->full_proj_transform_,
        0,  // sh_degree = 0 (not needed for depth)
        viewpoint_camera->camera_center_,
        false
    );

    GaussianRasterizer rasterizer(raster_settings);

    auto means3D = pc->getXYZ();
    auto means2D = torch::zeros_like(means3D, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    auto opacity = pc->getOpacityActivation();

    // Use precomputed 3D covariance for efficiency
    auto cov3D_precomp = pc->getCovarianceActivation();

    // Ensure all tensors have correct dtype
    if (means3D.scalar_type() != torch::kFloat32) {
        means3D = means3D.to(torch::kFloat32);
    }
    if (means2D.scalar_type() != torch::kFloat32) {
        means2D = means2D.to(torch::kFloat32);
    }
    if (opacity.scalar_type() != torch::kFloat32) {
        opacity = opacity.to(torch::kFloat32);
    }
    if (cov3D_precomp.scalar_type() != torch::kFloat32) {
        cov3D_precomp = cov3D_precomp.to(torch::kFloat32);
    }

    // Encode camera-space depth and unit opacity in differentiable color
    // channels. The rasterizer composites each channel as sum(value*alpha*T),
    // so channel 0 is the depth numerator and channel 1 is accumulated
    // opacity. Unlike the auxiliary depth output, color channels participate
    // in the existing rasterizer backward pass.
    auto homogeneous_means = torch::cat(
        {means3D, torch::ones(
            {means3D.size(0), 1},
            means3D.options())},
        1);
    auto camera_means =
        torch::matmul(homogeneous_means,
                      viewpoint_camera->world_view_transform_);
    auto camera_depth = camera_means.index(
        {torch::indexing::Slice(), 2}).unsqueeze(1);
    auto unit_channel = torch::ones_like(camera_depth);
    auto colors_precomp = torch::cat(
        {camera_depth, unit_channel, torch::zeros_like(camera_depth)}, 1);

    // Create empty tensors with correct dtype/device for unused parameters
    torch::Tensor shs_empty = torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    torch::Tensor scales_empty = torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    torch::Tensor rotations_empty = torch::empty({0}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));

    // Rasterize the depth numerator and opacity together.
    auto rasterizer_result = rasterizer.forward(
        means3D,
        means2D,
        opacity,
        false,  // has_shs
        true,   // has_color_precomp
        false,  // has_scales
        false,  // has_rotations
        true,   // has_cov3D_precomp
        shs_empty,         // shs (empty but typed)
        colors_precomp,
        scales_empty,      // scales (empty but typed)
        rotations_empty,   // rotations (empty but typed)
        cov3D_precomp
    );

    auto rendered_features = std::get<0>(rasterizer_result);
    auto depth_numerator = rendered_features.index({0});
    auto rendered_opacity = rendered_features.index({1});
    constexpr float kMinNormalizationOpacity = 1e-6f;
    auto rendered_depth = torch::where(
        rendered_opacity > kMinNormalizationOpacity,
        depth_numerator /
            torch::clamp_min(rendered_opacity, kMinNormalizationOpacity),
        torch::zeros_like(depth_numerator));

    return std::make_tuple(rendered_depth, rendered_opacity);
}

torch::Tensor
GaussianRenderer::renderDepth(
    std::shared_ptr<GaussianKeyframe> viewpoint_camera,
    int image_height,
    int image_width,
    std::shared_ptr<GaussianModel> pc,
    float scaling_modifier)
{
    return std::get<0>(renderDepthWithOpacity(
        viewpoint_camera,
        image_height,
        image_width,
        pc,
        scaling_modifier));
}
