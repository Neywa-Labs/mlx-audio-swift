// Copyright © 2025 Apple Inc. Adapted from mlx-lm under the MIT License.
import MLX
import MLXFast

enum Zonos3DeltaKernel {
    private static let kernel = MLXFast.metalKernel(
        name: "zonos3_gated_delta",
        inputNames: ["q", "k", "v", "g", "beta", "state_in"], outputNames: ["y", "state_out"],
        source: """

            auto n = thread_position_in_grid.z;
            auto b_idx = n / Hv;
            auto hv_idx = n % Hv;
            auto hk_idx = hv_idx / (Hv / Hk);
            constexpr int n_per_t = Dk / 32;

            // q, k: [B, T, Hk, Dk]
            auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
            auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;

            // v, y: [B, T, Hv, Dv]
            auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
            y += b_idx * T * Hv * Dv + hv_idx * Dv;

            auto dk_idx = thread_position_in_threadgroup.x;
            auto dv_idx = thread_position_in_grid.y;

            // state_in, state_out: [B, Hv, Dv, Dk]
            auto i_state = state_in + (n * Dv + dv_idx) * Dk;
            auto o_state = state_out + (n * Dv + dv_idx) * Dk;

            float state[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = static_cast<float>(i_state[s_idx]);
            }

            // g: [B, T, Hv]
            auto g_ = g + b_idx * T * Hv;
            auto beta_ = beta + b_idx * T * Hv;

            for (int t = 0; t < T; ++t) {
              float kv_mem = 0.0f;
              for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * dk_idx + i;
                state[i] = state[i] * g_[hv_idx];
                kv_mem += state[i] * k_[s_idx];
              }
              kv_mem = simd_sum(kv_mem);

              auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];

              float out = 0.0f;
              for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * dk_idx + i;
                state[i] = state[i] + k_[s_idx] * delta;
                out += state[i] * q_[s_idx];
              }
              out = simd_sum(out);
              if (thread_index_in_simdgroup == 0) {
                y[dv_idx] = static_cast<InT>(out);
              }
              // Increment data pointers to next time step
              q_ += Hk * Dk;
              k_ += Hk * Dk;
              v_ += Hv * Dv;
              y += Hv * Dv;
              g_ += Hv;
              beta_ += Hv;
            }
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              o_state[s_idx] = static_cast<StT>(state[i]);
            }

            """)
    static func callAsFunction(
        q: MLXArray, k: MLXArray, v: MLXArray, decay: MLXArray,
        beta: MLXArray, state: MLXArray
    ) -> (MLXArray, MLXArray) {
        let b = q.dim(0)
        let t = q.dim(1)
        let h = q.dim(2)
        let d = q.dim(3)
        if t > 1 && d % 32 == 0 && Device.defaultDevice() == .gpu {
            let out = kernel(
                [q, k, v, decay, beta, state],
                template: [
                    ("InT", DType.float32), ("StT", DType.float32), ("T", t),
                    ("Hk", h), ("Hv", h), ("Dk", d), ("Dv", d),
                ],
                grid: (32, d, b * h), threadGroup: (32, 4, 1),
                outputShapes: [v.shape, state.shape], outputDTypes: [.float32, .float32])
            return (out[0], out[1])
        }
        return reference(q: q, k: k, v: v, decay: decay, beta: beta, state: state)
    }
    static func reference(
        q: MLXArray, k: MLXArray, v: MLXArray, decay: MLXArray,
        beta: MLXArray, state initial: MLXArray
    ) -> (MLXArray, MLXArray) {
        var state = initial
        var outputs: [MLXArray] = []
        for t in 0 ..< q.dim(1) {
            let key = k[0..., t, 0..., .newAxis, 0...]
            state = state * decay[0..., t, 0..., .newAxis, .newAxis]
            let delta = (v[0..., t] - (state * key).sum(axis: -1)) * beta[0..., t, 0..., .newAxis]
            state = state + key * delta[0..., 0..., 0..., .newAxis]
            outputs.append((state * q[0..., t, 0..., .newAxis, 0...]).sum(axis: -1))
        }
        return (stacked(outputs, axis: 1), state)
    }
}
