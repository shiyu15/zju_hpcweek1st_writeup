#pragma once

#include <vector>

/**
 * Filtered Back-Projection (FBP) CT Reconstruction
 * 
 * Reconstructs 3D volume from sinogram data using the FBP algorithm:
 * 1. Apply ramp filter to each projection (in frequency domain)
 * 2. Backproject filtered projections to reconstruct image
 * 
 * @param sino_buffer   Input sinogram [n_slices, n_angles, n_det] - modified in-place during filtering
 * @param recon_buffer  Output reconstructed volume [n_slices, n_det, n_det]
 * @param n_slices      Number of slices (z-dimension)
 * @param n_angles      Number of projection angles
 * @param n_det         Number of detector pixels per projection
 * @param angles_deg    Projection angles in degrees [n_angles]
 */
void fbp_reconstruct_3d(
    float* sino_buffer,
    float* recon_buffer,
    int n_slices,
    int n_angles,
    int n_det,
    const std::vector<float>& angles_deg
);
