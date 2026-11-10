import Foundation
import MLX
import MLXAudioCore
import MLXNN
import Testing

@testable import MLXAudioTTS

@Suite(.serialized)
struct Zonos3Tests {
    // Synthetic reference from mlx-audio PR #1162: deterministic mod-17 weights,
    // CPU backbone logits, and independent NumPy overlap-add/FIR/resampling.
    private func fixture() throws -> [String: Any] {
        let json = #"""
            {
                "config": {
                    "model_type": "zonos3",
                    "backbone": {
                        "n_layers": 4,
                        "max_seqlen": 32,
                        "dim": 8,
                        "head_dim": 4,
                        "n_heads": 2,
                        "n_kv_heads": 1,
                        "ffn_hidden": 12,
                        "n_experts": 3,
                        "moe_topk": 2,
                        "n_codebooks": 3,
                        "codebook_size": 8,
                        "audio_vocab": 10,
                        "eoa_id": 8,
                        "audio_pad_id": 9,
                        "span_end_id": null,
                        "text_vocab": 20,
                        "text_extra_tokens": 0,
                        "gdn_heads": 2,
                        "gdn_head_dim": 4,
                        "gdn_kernel": 4,
                        "speaker_embedding_dim": 4,
                        "condition_count": 3,
                        "condition_bucket_counts": [2, 2, 3],
                        "condition_names": ["loudness", "fmax", "speaking_rate"],
                        "condition_units": {},
                        "condition_edges": {"loudness": [-20.0], "fmax": [10000.0], "speaking_rate": [10.0, 20.0]},
                        "rope_theta": 10000.0,
                        "norm_eps": 1e-05,
                        "qk_eps": 1e-06,
                        "loss_softcap": 15.0,
                        "gdn_period": 4,
                        "silence_frame": [0, 1, 2],
                        "text_tokenizer_spec": {
                            "kind": "superbpe",
                            "path": "tokenizer.json",
                            "sha256": "",
                            "vocab_size": 16,
                            "offset": 4,
                            "bos": 2,
                            "eos": 3
                        },
                        "tag_cfg_trained": false,
                        "audio_codec": "opus_dual_3n_600k_sched600k_r4096_step600000",
                        "speaker_embedding_codec": "ownmect_s2",
                        "speaker_reference_prefix_enabled": true,
                        "speaker_reference_max_seconds": 60.0
                    },
                    "tensors": {
                        "audio_embd.0.weight": {"shape": [10, 8], "matrix": false, "dtype": "float32"},
                        "audio_embd.1.weight": {"shape": [10, 8], "matrix": false, "dtype": "float32"},
                        "audio_embd.2.weight": {"shape": [10, 8], "matrix": false, "dtype": "float32"},
                        "text_embd.weight": {"shape": [21, 8], "matrix": false, "dtype": "float32"},
                        "speaker_proj.weight": {"shape": [8, 4], "matrix": true, "dtype": "float32"},
                        "speaker_proj.bias": {"shape": [8], "matrix": false, "dtype": "float32"},
                        "cond_embd.0.weight": {"shape": [3, 8], "matrix": false, "dtype": "float32"},
                        "cond_embd.1.weight": {"shape": [3, 8], "matrix": false, "dtype": "float32"},
                        "cond_embd.2.weight": {"shape": [4, 8], "matrix": false, "dtype": "float32"},
                        "attnres_q.weight": {"shape": [9, 8], "matrix": false, "dtype": "float32"},
                        "output_norm.weight": {"shape": [8], "matrix": false, "dtype": "float32"},
                        "output.weight": {"shape": [30, 8], "matrix": true, "dtype": "float32"},
                        "blk.0.attn_norm.weight": {"shape": [8], "matrix": false, "dtype": "float32"},
                        "blk.0.ffn_norm.weight": {"shape": [8], "matrix": false, "dtype": "float32"},
                        "blk.0.attn_output.weight": {"shape": [8, 8], "matrix": true, "dtype": "float32"},
                        "blk.0.gdn_in.weight": {"shape": [36, 8], "matrix": true, "dtype": "float32"},
                        "blk.0.gdn_conv.weight": {"shape": [24, 4], "matrix": false, "dtype": "float32"},
                        "blk.0.gdn_A_log": {"shape": [2], "matrix": false, "dtype": "float32"},
                        "blk.0.gdn_dt_bias": {"shape": [2], "matrix": false, "dtype": "float32"},
                        "blk.0.gdn_norm.weight": {"shape": [4], "matrix": false, "dtype": "float32"},
                        "blk.0.ffn_gate.weight": {"shape": [12, 8], "matrix": true, "dtype": "float32"},
                        "blk.0.ffn_up.weight": {"shape": [12, 8], "matrix": true, "dtype": "float32"},
                        "blk.0.ffn_down.weight": {"shape": [8, 12], "matrix": true, "dtype": "float32"},
                        "blk.1.attn_norm.weight": {"shape": [8], "matrix": false, "dtype": "float32"},
                        "blk.1.ffn_norm.weight": {"shape": [8], "matrix": false, "dtype": "float32"},
                        "blk.1.attn_output.weight": {"shape": [8, 8], "matrix": true, "dtype": "float32"},
                        "blk.1.gdn_in.weight": {"shape": [36, 8], "matrix": true, "dtype": "float32"},
                        "blk.1.gdn_conv.weight": {"shape": [24, 4], "matrix": false, "dtype": "float32"},
                        "blk.1.gdn_A_log": {"shape": [2], "matrix": false, "dtype": "float32"},
                        "blk.1.gdn_dt_bias": {"shape": [2], "matrix": false, "dtype": "float32"},
                        "blk.1.gdn_norm.weight": {"shape": [4], "matrix": false, "dtype": "float32"},
                        "blk.1.router_norm.weight": {"shape": [8], "matrix": false, "dtype": "float32"},
                        "blk.1.router.weight": {"shape": [3, 8], "matrix": true, "dtype": "float32"},
                        "blk.1.router_bias": {"shape": [3], "matrix": false, "dtype": "float32"},
                        "blk.1.ffn_latent_down.weight": {"shape": [4, 8], "matrix": true, "dtype": "float32"},
                        "blk.1.ffn_latent_up.weight": {"shape": [8, 4], "matrix": true, "dtype": "float32"},
                        "blk.1.ffn_gate_exps.weight": {"shape": [3, 6, 4], "matrix": true, "dtype": "float32"},
                        "blk.1.ffn_up_exps.weight": {"shape": [3, 6, 4], "matrix": true, "dtype": "float32"},
                        "blk.1.ffn_down_exps.weight": {"shape": [3, 4, 6], "matrix": true, "dtype": "float32"},
                        "blk.2.attn_norm.weight": {"shape": [8], "matrix": false, "dtype": "float32"},
                        "blk.2.ffn_norm.weight": {"shape": [8], "matrix": false, "dtype": "float32"},
                        "blk.2.attn_output.weight": {"shape": [8, 8], "matrix": true, "dtype": "float32"},
                        "blk.2.gdn_in.weight": {"shape": [36, 8], "matrix": true, "dtype": "float32"},
                        "blk.2.gdn_conv.weight": {"shape": [24, 4], "matrix": false, "dtype": "float32"},
                        "blk.2.gdn_A_log": {"shape": [2], "matrix": false, "dtype": "float32"},
                        "blk.2.gdn_dt_bias": {"shape": [2], "matrix": false, "dtype": "float32"},
                        "blk.2.gdn_norm.weight": {"shape": [4], "matrix": false, "dtype": "float32"},
                        "blk.2.router_norm.weight": {"shape": [8], "matrix": false, "dtype": "float32"},
                        "blk.2.router.weight": {"shape": [3, 8], "matrix": true, "dtype": "float32"},
                        "blk.2.router_bias": {"shape": [3], "matrix": false, "dtype": "float32"},
                        "blk.2.ffn_latent_down.weight": {"shape": [4, 8], "matrix": true, "dtype": "float32"},
                        "blk.2.ffn_latent_up.weight": {"shape": [8, 4], "matrix": true, "dtype": "float32"},
                        "blk.2.ffn_gate_exps.weight": {"shape": [3, 6, 4], "matrix": true, "dtype": "float32"},
                        "blk.2.ffn_up_exps.weight": {"shape": [3, 6, 4], "matrix": true, "dtype": "float32"},
                        "blk.2.ffn_down_exps.weight": {"shape": [3, 4, 6], "matrix": true, "dtype": "float32"},
                        "blk.3.attn_norm.weight": {"shape": [8], "matrix": false, "dtype": "float32"},
                        "blk.3.ffn_norm.weight": {"shape": [8], "matrix": false, "dtype": "float32"},
                        "blk.3.attn_output.weight": {"shape": [8, 8], "matrix": true, "dtype": "float32"},
                        "blk.3.attn_q.weight": {"shape": [8, 8], "matrix": true, "dtype": "float32"},
                        "blk.3.attn_k.weight": {"shape": [4, 8], "matrix": true, "dtype": "float32"},
                        "blk.3.attn_v.weight": {"shape": [4, 8], "matrix": true, "dtype": "float32"},
                        "blk.3.attn_gate.weight": {"shape": [2, 8], "matrix": true, "dtype": "float32"},
                        "blk.3.attn_temp": {"shape": [2], "matrix": false, "dtype": "float32"},
                        "blk.3.router_norm.weight": {"shape": [8], "matrix": false, "dtype": "float32"},
                        "blk.3.router.weight": {"shape": [3, 8], "matrix": true, "dtype": "float32"},
                        "blk.3.router_bias": {"shape": [3], "matrix": false, "dtype": "float32"},
                        "blk.3.ffn_latent_down.weight": {"shape": [4, 8], "matrix": true, "dtype": "float32"},
                        "blk.3.ffn_latent_up.weight": {"shape": [8, 4], "matrix": true, "dtype": "float32"},
                        "blk.3.ffn_gate_exps.weight": {"shape": [3, 6, 4], "matrix": true, "dtype": "float32"},
                        "blk.3.ffn_up_exps.weight": {"shape": [3, 6, 4], "matrix": true, "dtype": "float32"},
                        "blk.3.ffn_down_exps.weight": {"shape": [3, 4, 6], "matrix": true, "dtype": "float32"}
                    }
                },
                "ids": [
                    [0, 1, 2, 0],
                    [3, 4, 5, 1],
                    [6, 7, 0, 2],
                    [1, 2, 3, 3],
                    [4, 5, 6, 4],
                    [7, 0, 1, 5],
                    [2, 3, 4, 6],
                    [5, 6, 7, 7],
                    [0, 1, 2, 8],
                    [3, 4, 5, 9]
                ],
                "logits": [
                    0.3121940791606903, 0.004913776647299528, -0.0918201208114624, 0.043329332023859024,
                    -0.27684488892555237, 0.08174432069063187, -0.2315278798341751, 0.12015824019908905,
                    0.034805577248334885, 0.1585705727338791, -0.15555362403392792, 0.19698084890842438,
                    -0.35670605301856995, 0.23538854718208313, -0.3158690929412842, 0.27379310131073, -0.03350185230374336,
                    0.3121940791606903, 0.004913776647299528, -0.0918201208114624, 0.043329332023859024,
                    -0.27684488892555237, 0.08174432069063187, -0.2315278798341751, 0.12015824019908905,
                    0.034805577248334885, 0.1585705727338791, -0.15555362403392792, 0.19698084890842438,
                    -0.35670605301856995, 0.5666362643241882, -0.2459779530763626, 0.2960856258869171, -0.1443822979927063,
                    0.060867998749017715, -0.04277338832616806, -0.08324100822210312, 0.05883943662047386,
                    -0.23725968599319458, 0.1604468822479248, -0.3394424617290497, 0.26203954219818115,
                    -0.39587441086769104, 0.3636082410812378, -0.3973749876022339, 0.4651435315608978,
                    -0.34755104780197144, 0.5666362643241882, -0.2459779530763626, 0.2960856258869171, -0.1443822979927063,
                    0.060867998749017715, -0.04277338832616806, -0.08324100822210312, 0.05883943662047386,
                    -0.23725968599319458, 0.1604468822479248, -0.3394424617290497, 0.26203954219818115,
                    -0.3958744406700134, -0.3045860826969147, 0.5011081695556641, -0.3923989534378052, 0.4004592299461365,
                    -0.42152872681617737, 0.299774169921875, -0.4075421392917633, 0.1990620493888855, -0.3270226716995239,
                    0.098331980407238, -0.21276432275772095, -0.0024069531355053186, 0.01859559863805771,
                    -0.10314565896987915, 0.2559676766395569, -0.2038750797510147, 0.6017120480537415, -0.3045860826969147,
                    0.5011081695556641, -0.3923989534378052, 0.4004592299461365, -0.42152872681617737, 0.299774169921875,
                    -0.4075421392917633, 0.1990620493888855, -0.3270226716995239, 0.098331980407238, -0.21276432275772095,
                    -0.0024069512728601694, 0.018595602363348007, 0.27225807309150696, -0.29213443398475647,
                    0.6537094116210938, -0.221597820520401, 0.4010937213897705, -0.15105141699314117, 0.1795814335346222,
                    -0.0804983377456665, -0.01501214038580656, -0.00994169618934393, -0.16779494285583496,
                    0.06061539426445961, -0.27093175053596497, 0.13116978108882904, -0.32887107133865356,
                    0.20171838998794556, -0.36265814304351807, 0.27225807309150696, -0.29213443398475647,
                    0.6537094116210938, -0.221597820520401, 0.4010937213897705, -0.15105141699314117, 0.1795814335346222,
                    -0.0804983377456665, -0.01501214038580656, -0.00994169618934393, -0.16779494285583496,
                    0.06061539426445961, -0.27093178033828735, 0.0012552441330626607, -0.2748773992061615,
                    0.29689154028892517, -0.24036717414855957, 0.6728153228759766, -0.2058544158935547, 0.6056005358695984,
                    -0.17133943736553192, 0.30148249864578247, -0.1368226706981659, 0.04001735895872116,
                    -0.10230444371700287, -0.13080939650535583, -0.06778516620397568, -0.24596616625785828,
                    -0.0332651324570179, -0.3093847632408142, 0.0012552441330626607, -0.2748773992061615,
                    0.29689154028892517, -0.24036717414855957, 0.6728153228759766, -0.2058544158935547, 0.6056005358695984,
                    -0.17133943736553192, 0.30148249864578247, -0.1368226706981659, 0.04001735895872116,
                    -0.10230445116758347, -0.13080939650535583, 0.19089800119400024, 0.3214960992336273,
                    -0.18460634350776672, 0.3051736056804657, -0.41816386580467224, 0.2888503670692444,
                    -0.5593938231468201, 0.27252650260925293, -0.5905889868736267, 0.25620195269584656, -0.511965274810791,
                    0.23987674713134766, -0.3384784758090973, 0.22355102002620697, -0.03978048637509346,
                    0.2072247415781021, 0.33781787753105164, 0.19089800119400024, 0.3214960992336273, -0.18460634350776672,
                    0.3051736056804657, -0.41816386580467224, 0.2888503670692444, -0.5593938231468201, 0.27252650260925293,
                    -0.5905889868736267, 0.25620195269584656, -0.511965274810791, 0.23987674713134766, -0.3384784758090973,
                    0.05746354162693024, -0.2876800298690796, 0.45978763699531555, -0.24454626441001892, 0.642463207244873,
                    -0.20140840113162994, 0.5678135752677917, -0.15826724469661713, 0.23360998928546906,
                    -0.11512347310781479, -0.034723781049251556, -0.07197780162096024, -0.20514549314975739,
                    -0.028830936178565025, -0.2975865602493286, 0.01431641262024641, -0.3308090567588806,
                    0.05746354162693024, -0.2876800298690796, 0.45978763699531555, -0.24454626441001892, 0.642463207244873,
                    -0.20140840113162994, 0.5678135752677917, -0.15826724469661713, 0.23360998928546906,
                    -0.11512347310781479, -0.034723781049251556, -0.07197779417037964, -0.20514550805091858,
                    -0.020874585956335068, 0.2385159730911255, -0.1706218272447586, 0.20609678328037262,
                    -0.08807821571826935, 0.17367568612098694, 0.1478944569826126, 0.14125293493270874,
                    -0.20696775615215302, 0.1088288426399231, -0.39792513847351074, 0.07640375196933746,
                    -0.3694482445716858, 0.04397794231772423, -0.16510096192359924, 0.011551735922694206,
                    0.2709329426288605, -0.020874585956335068, 0.2385159730911255, -0.1706218272447586,
                    0.20609678328037262, -0.08807821571826935, 0.17367568612098694, 0.1478944569826126,
                    0.14125293493270874, -0.20696775615215302, 0.1088288426399231, -0.39792513847351074,
                    0.07640374451875687, -0.3694482147693634, 0.2645196318626404, -0.32291218638420105, 0.5044957399368286,
                    -0.24950045347213745, 0.31831419467926025, -0.1760767251253128, 0.2792630195617676,
                    -0.10264456272125244, 0.35438141226768494, -0.029207492247223854, -0.06157631054520607,
                    0.04423099383711815, -0.3017936944961548, 0.11766733229160309, -0.4339899718761444,
                    0.19109807908535004, -0.396308571100235, 0.2645196318626404, -0.32291218638420105, 0.5044957399368286,
                    -0.24950045347213745, 0.31831419467926025, -0.1760767251253128, 0.2792630195617676,
                    -0.10264456272125244, 0.35438141226768494, -0.029207492247223854, -0.06157631054520607,
                    0.04423099383711815, -0.3017936944961548, 0.051151715219020844, 0.250297486782074,
                    0.011215968057513237, 0.225407674908638, 0.15038836002349854, 0.20051658153533936,
                    -0.24331910908222198, 0.17562438547611237, -0.44859111309051514, 0.15073125064373016,
                    -0.5200315713882446, 0.1258372962474823, -0.4178254008293152, 0.10094258934259415,
                    -0.16320782899856567, 0.0760473683476448, 0.2751859426498413, 0.051151715219020844, 0.250297486782074,
                    0.011215968057513237, 0.225407674908638, 0.15038836002349854, 0.20051658153533936,
                    -0.24331910908222198, 0.17562438547611237, -0.44859111309051514, 0.15073126554489136,
                    -0.5200315713882446, 0.1258372813463211, -0.4178254306316376
                ],
                "istft": {
                    "spectra": [
                        [
                            [
                                -0.0783739686012268, 0.00998408067971468, -0.03303803130984306, -0.03359387814998627,
                                0.012523175217211246, -0.04514221474528313, 0.059384554624557495, -0.009778572246432304,
                                0.11361393332481384, -0.13140642642974854
                            ],
                            [
                                -0.07213199883699417, -0.17121660709381104, -0.33399227261543274, -0.44649988412857056,
                                0.012023933231830597, -0.16610178351402283, -0.15102741122245789, -0.11614282429218292,
                                0.08354472368955612, 0.11356171220541
                            ],
                            [
                                0.1372034102678299, 0.1849593073129654, 0.5965042114257812, 0.058943815529346466,
                                0.030873438343405724, -0.4364403188228607, -0.3368121385574341, 0.4385901689529419,
                                0.5491572618484497, 0.08926909416913986
                            ],
                            [
                                -0.016056379303336143, 0.11141777038574219, -0.40705594420433044, -0.10612142831087112,
                                -0.081083744764328, 0.3470998704433441, -0.19824807345867157, 0.07260415703058243,
                                0.1448204666376114, 0.21606306731700897
                            ],
                            [
                                -0.08580593019723892, -0.137413889169693, -0.43535080552101135, 0.22951751947402954,
                                0.2364400029182434, -0.09798634797334671, 0.04622652381658554, -0.2792120575904846,
                                -0.04185347631573677, 0.14551398158073425
                            ],
                            [
                                -0.0966801643371582, 0.012617383152246475, 0.2199951559305191, 0.12008704245090485,
                                -0.12256623804569244, 0.3229476809501648, 0.35492098331451416, 0.2006954848766327,
                                0.12013381719589233, 0.14315366744995117
                            ],
                            [
                                0.17636993527412415, -0.2525344491004944, 0.21590782701969147, -0.16809557378292084,
                                0.040014129132032394, -0.019775981083512306, 0.09494414925575256, 0.35172009468078613,
                                0.1065741777420044, -0.023714059963822365
                            ]
                        ],
                        [
                            [
                                0.13391155004501343, 0.02317728102207184, -0.3307519853115082, -0.3622294068336487,
                                -0.10667180269956589, 0.3981839418411255, -0.05619920417666435, 0.13108745217323303,
                                0.25547194480895996, -0.2660653293132782, 0.2695649564266205, 0.17019689083099365,
                                0.10692889988422394, 0.19174796342849731, -0.4142383337020874, 0.03201684728264809,
                                -0.3126509487628937, 0.0658976137638092, -0.038105495274066925, -0.08953758329153061,
                                0.11351648718118668, -0.3577808737754822, 0.013333678245544434, 0.4346694350242615,
                                0.2055450826883316, -0.1934676170349121
                            ],
                            [
                                0.08297362178564072, -0.03603781759738922, 0.09869223088026047, -0.23834794759750366,
                                0.09422223269939423, 0.00012875895481556654, -0.2340921014547348, 0.13972240686416626,
                                0.27422139048576355, 0.23074059188365936, -0.5281195640563965, 0.06962679326534271,
                                0.07305517792701721, -0.03882984444499016, -0.035024967044591904, 0.08540673553943634,
                                0.13483528792858124, -0.14524178206920624, 0.04375261813402176, -0.04731712490320206,
                                -0.15017622709274292, 0.07451587170362473, -0.1151009351015091, -0.35310909152030945,
                                -0.10768534243106842, -0.3554694652557373
                            ],
                            [
                                0.18329393863677979, -0.05693317577242851, 0.19987301528453827, -0.40699777007102966,
                                0.32320940494537354, 0.09610087424516678, -0.22938627004623413, 0.41105037927627563,
                                -0.14402352273464203, -0.23041564226150513, -0.032581884413957596, 0.08088110387325287,
                                0.17404413223266602, 0.468875527381897, -0.1727224886417389, -0.3628709018230438,
                                0.061988476663827896, -0.25675109028816223, -0.26962196826934814, -0.011944829486310482,
                                0.2658756375312805, -0.17220908403396606, -0.150114968419075, -0.04718136787414551,
                                -0.07126720994710922, 0.22231470048427582
                            ],
                            [
                                0.0022614572662860155, -0.2539618909358978, 0.47950902581214905, -0.1278383582830429,
                                -0.33407944440841675, 0.11804981529712677, 0.008459600619971752, 0.12513217329978943,
                                -0.017733125016093254, 0.0042512319050729275, 0.03361235931515694, 0.18049074709415436,
                                -0.44480839371681213, 0.02597092278301716, 0.001163648790679872, 0.44169601798057556,
                                -0.1484464854001999, 0.25011053681373596, 0.18976643681526184, -0.1370687186717987,
                                0.1214904710650444, 0.17950932681560516, 0.25997692346572876, 0.18556217849254608,
                                -0.010224463418126106, 0.1128588616847992
                            ],
                            [
                                -0.12274457514286041, -0.17955656349658966, -0.08826588094234467, -0.14009815454483032,
                                -0.1644231379032135, 0.3342423737049103, -0.3551694452762604, 0.07773910462856293,
                                0.084773488342762, -0.2304197996854782, 0.2785221040248871, -0.19601239264011383,
                                0.0448942705988884, -0.33936357498168945, -0.2770017087459564, -0.16442488133907318,
                                0.06046118214726448, 0.31463930010795593, -0.2421518713235855, 0.17123082280158997,
                                -0.11502715200185776, 0.1499720811843872, 0.1466141641139984, 0.04290435463190079,
                                0.06741683185100555, -0.3739858865737915
                            ],
                            [
                                0.14416341483592987, 0.25124484300613403, -0.02027410827577114, 0.16897886991500854,
                                0.15928149223327637, 0.25114694237709045, -0.15532375872135162, -0.10338893532752991,
                                0.1618310958147049, 0.15490934252738953, 0.28785017132759094, -0.04671073704957962,
                                0.005842945072799921, -0.14078480005264282, -0.04153516888618469, 0.28382763266563416,
                                0.004089691210538149, 0.11147715896368027, -0.00813097320497036, -0.05855690315365791,
                                -0.000478917034342885, -0.004705379251390696, -0.19710595905780792, -0.15013806521892548,
                                -0.03807573765516281, 0.11911899596452713
                            ],
                            [
                                -0.015850810334086418, -0.3150700032711029, 0.5370386242866516, 0.0630897656083107,
                                -0.06292960047721863, 0.02337479777634144, 0.0007432223064824939, 0.00845913402736187,
                                0.11760544776916504, 0.09690993279218674, 0.034676022827625275, 0.22584384679794312,
                                -0.1853477954864502, 0.42015302181243896, 0.0578746534883976, -0.19493712484836578,
                                -0.3887796998023987, 0.060377880930900574, 0.15722590684890747, -0.06533075869083405,
                                -0.04076697677373886, -0.07681643962860107, 0.12929733097553253, 0.0632961243391037,
                                -0.2907032072544098, 0.09195901453495026
                            ]
                        ]
                    ],
                    "windows": [
                        [0.0, 0.1464466154575348, 0.5, 0.8535534143447876, 1.0, 0.8535534143447876, 0.5, 0.1464466154575348],
                        [
                            0.0, 0.017037086188793182, 0.0669872984290123, 0.1464466154575348, 0.25, 0.370590478181839,
                            0.5, 0.6294095516204834, 0.75, 0.8535534143447876, 0.9330127239227295, 0.9829629063606262, 1.0,
                            0.9829629063606262, 0.9330127239227295, 0.8535534143447876, 0.75, 0.6294095516204834, 0.5,
                            0.370590478181839, 0.25, 0.1464466154575348, 0.0669872984290123, 0.017037086188793182
                        ]
                    ],
                    "mixer": [
                        [0.10000000149011612, 0.6000000238418579, 0.10000000149011612],
                        [-0.05000000074505806, 0.30000001192092896, -0.05000000074505806]
                    ],
                    "kernel": [
                        [
                            0.00901979859918356, 0.04988808557391167, -0.08353324234485626, 0.08717194944620132,
                            0.14347465336322784
                        ],
                        [
                            -0.21297770738601685, 0.09952937066555023, -0.0017934504430741072, -0.004738915245980024,
                            -0.023515379056334496
                        ]
                    ],
                    "audio": [
                        0.013879626989364624, -0.002334382850676775, -0.02638687565922737, 0.0029561102855950594,
                        -0.008950991556048393, 0.0034143601078540087, 0.0127077866345644, 0.014473735354840755,
                        0.0043990593403577805, -0.007078712806105614
                    ]
                }
            }
            """#
        return try JSONSerialization.jsonObject(with: Data(json.utf8)) as! [String: Any]
    }
    private func tiny() throws -> (Zonos3Configuration, Zonos3Backbone, [String: Any]) {
        let fixture = try fixture()
        let decoder = JSONDecoder()
        let c = try decoder.decode(
            Zonos3Configuration.self,
            from: JSONSerialization.data(withJSONObject: fixture["config"]!))
        var tensors: [String: MLXArray] = [:]
        for (name, spec) in c.tensors {
            var value = ((MLXArray(0 ..< spec.shape.reduce(1, *)).asType(.float32) % 17) - 8) / 64
            if name.contains("norm.weight") || name.hasSuffix("attn_temp") { value = value + 1 }
            let key =
                (spec.matrix ? "matrices." : "tables.")
                + name.replacingOccurrences(of: ".", with: "__") + (spec.matrix ? ".weight" : "")
            tensors[key] = value.reshaped(spec.shape)
        }
        return (
            c,
            Zonos3Backbone(
                c.backbone, params: try Zonos3TensorStore(specs: c.tensors, weights: tensors)),
            fixture
        )
    }
    private func close(_ actual: MLXArray, _ expected: MLXArray, tolerance: Float = 3e-5) {
        #expect(actual.shape == expected.shape)
        let delta = abs(actual - expected).max().item(Float.self)
        if delta >= tolerance {
            print(
                "ZONOS3 mismatch", actual.shape, "max", delta, "actual",
                Array(actual.asArray(Float.self).prefix(20)), "expected",
                Array(expected.asArray(Float.self).prefix(20)))
        }
        #expect(delta < tolerance)
    }

    @Test func configurationConditionsAndRegistration() throws {
        let (c, _, _) = try tiny()
        try c.backbone.validate()
        #expect(TTS.resolveModelType(modelRepo: "mlx-community/ZONOS3-Mini") == "zonos3")
        #expect(
            TTS.resolveModelType(modelRepo: "renamed/model", modelType: "zonos3_mini")
                == "zonos3_mini")
        #expect(try Zonos3Generation.conditions(c.backbone, options: .init()) == [2, 2, 3])
        var options = Zonos3Options()
        options.conditionValues = ["loudness": -20, "speaking_rate": 15]
        #expect(try Zonos3Generation.conditions(c.backbone, options: options) == [1, 2, 1])
        options.speed = 2
        #expect(throws: Zonos3Error.self) {
            try Zonos3Generation.conditions(c.backbone, options: options)
        }
        options = .init()
        options.conditionBuckets = ["unknown": 1]
        #expect(throws: Zonos3Error.self) {
            try Zonos3Generation.conditions(c.backbone, options: options)
        }
    }

    @Test func backboneMatchesPythonAndCachedDecode() throws {
        try Device.withDefaultDevice(.cpu) {
            let (_, model, fixture) = try tiny()
            let rows = fixture["ids"] as! [[Int]]
            let ids = MLXArray(rows.flatMap { $0 }).asType(.int32).reshaped(rows.count, 4)
            let expected = MLXArray((fixture["logits"] as! [Double]).map(Float.init)).reshaped(
                10, 3, 10)
            let full = try model(
                ids, state: Zonos3LMState(), speaker: .ones([4]), conditions: [0, 1, 2])
            close(full, expected)
            let state = Zonos3LMState()
            var pieces = [
                try model(ids[0 ..< 5], state: state, speaker: .ones([4]), conditions: [0, 1, 2])
            ]
            eval(pieces + state.arrays)
            for i in 5 ..< 10 {
                pieces.append(try model(ids[i ..< (i + 1)], state: state))
                eval(pieces + state.arrays)
            }
            close(concatenated(pieces), expected)
            #expect(state.position == 10)
            state.position = 32
            #expect(throws: Zonos3Error.self) { try model(ids[0 ..< 1], state: state) }
        }
    }

    @Test func packedMatricesLoadWithoutConversion() throws {
        let dense =
            ((MLXArray(0 ..< (3 * 8 * 32)).asType(.float32) % 31) - 15).reshaped(3, 8, 32) / 16
        let (w, s, optionalBias) = quantized(dense, groupSize: 32, bits: 4)
        let b = try #require(optionalBias)
        let spec = Zonos3TensorSpecification(
            shape: [3, 8, 32], matrix: true, dtype: "float32", bits: 4)
        let weights = [
            "matrices.experts__weight.weight": w, "matrices.experts__weight.scales": s,
            "matrices.experts__weight.biases": b,
        ]
        let store = try Zonos3TensorStore(specs: ["experts.weight": spec], weights: weights)
        let input = MLXArray.ones([1, 1, 1, 32])
        let choices = MLXArray([[0, 2]])
        let output = store.matrices["experts.weight"]!(input, indices: choices)
        let unpacked = dequantized(w, scales: s, biases: b, groupSize: 32, bits: 4)
        close(output, matmul(input, unpacked[choices].swappedAxes(-1, -2)), tolerance: 1e-5)
        #expect(throws: Zonos3Error.self) {
            try Zonos3TensorStore(specs: ["experts.weight": spec], weights: [:])
        }
    }

    @Test func deltaMetalMatchesRecurrence() {
        func array(_ count: Int, _ shape: [Int], _ divisor: Float) -> MLXArray {
            ((MLXArray(0 ..< count).asType(.float32) % 13) - 6).reshaped(shape) / divisor
        }
        let q = array(7 * 2 * 32, [1, 7, 2, 32], 100)
        let k = q * 0.5
        let v = q * 2
        let decay = MLXArray.full([1, 7, 2], values: MLXArray(Float(0.8)))
        let beta = decay * 0.6
        let state = MLXArray.zeros([1, 2, 32, 32])
        let actual = Zonos3DeltaKernel.callAsFunction(
            q: q, k: k, v: v, decay: decay, beta: beta, state: state)
        let reference = Zonos3DeltaKernel.reference(
            q: q, k: k, v: v, decay: decay, beta: beta, state: state)
        close(actual.0, reference.0, tolerance: 1e-6)
        close(actual.1, reference.1, tolerance: 1e-6)
    }

    @Test func delaySamplingAndLimiter() throws {
        let (config, _, _) = try tiny()
        let c = config.backbone
        let codes = MLXArray(0 ..< 12).asType(.int32).reshaped(4, 3)
        let delayed = Zonos3Generation.delay(codes, pad: 99)
        #expect(delayed[0].asArray(Int32.self) == [0, 99, 99])
        #expect(
            Zonos3Generation.aligned(delayed, codebooks: 3).asArray(Int32.self)
                == codes.asArray(Int32.self))
        #expect(Zonos3Generation.aligned(delayed, codebooks: 3, stop: 2).shape == [2, 3])
        #expect(
            Zonos3Generation.aligned(.zeros([0, 3], dtype: .int32), codebooks: 3).shape == [0, 3])
        let logits = MLXArray.full([3, 10], values: MLXArray(Float(-20)))
        logits[0..., 0] = MLXArray(Float(10))
        logits[0..., 1] = MLXArray(Float(7))
        logits[0..., 9] = MLXArray(Float(100))
        let history = MLXArray([[0, -1, 8], [9, -1, 8], [0, 8, 7]])
        var options = Zonos3Options()
        options.repetitionCodebooks = 1
        let params = GenerateParameters(temperature: 0, repetitionPenalty: 2)
        let token = Zonos3Generation.sample(
            logits, config: c, step: 5, history: history,
            parameters: params, options: options, hasPrefix: true)
        #expect(token.asArray(Int32.self) == [1, 0, 0])
        let cold = Zonos3Generation.sample(
            logits, config: c, step: 0, history: .zeros([0, 3], dtype: .int32),
            parameters: params, options: options, hasPrefix: false)
        #expect(cold.asArray(Int32.self) == [0, 9, 9])
        let limited = Zonos3Generation.output(MLXArray([Float(-100), 0, 100]), gain: 1)
        close(limited, MLXArray([Float(-1), 0, 1]), tolerance: 1e-6)
    }

    @Test func streamingHeadAndResamplerMatchNumpy() throws {
        try Device.withDefaultDevice(.cpu) {
            let f = try fixture()["istft"] as! [String: Any]
            let spectra = (f["spectra"] as! [[[Double]]]).map { a in
                MLXArray(a.flatMap { $0 }.map(Float.init)).reshaped(a.count, a[0].count)
            }
            let windows = (f["windows"] as! [[Double]]).map { MLXArray($0.map(Float.init)) }
            let mixer = MLXArray((f["mixer"] as! [[Double]]).flatMap { $0 }.map(Float.init))
                .reshaped(2, 3)
            let kernel = MLXArray((f["kernel"] as! [[Double]]).flatMap { $0 }.map(Float.init))
                .reshaped(2, 5)
            let expected = MLXArray((f["audio"] as! [Double]).map(Float.init))
            for partition in [[7], [1, 2, 1, 3]] {
                let head = Zonos3ISTFTStream(
                    nFFT: [8, 24], hop: 2, windows: windows, mixer: mixer, maxLogMagnitude: 4.6)
                let rs = Zonos3ResamplerStream(kernel: kernel, orig: 3, new: 2, width: 1)
                var output: [MLXArray] = []
                var start = 0
                for count in partition {
                    output.append(rs.push(head.push(spectra.map { $0[start ..< (start + count)] })))
                    start += count
                }
                output.append(rs.push(head.flush()))
                output.append(rs.flush())
                close(concatenated(output), expected, tolerance: 1e-6)
                #expect(head.flush().size == 0)
            }
        }
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["MLXAUDIO_ZONOS3_MODEL_DIR"] != nil))
    func publishedCheckpoint() async throws {
        let directory = URL(
            fileURLWithPath: ProcessInfo.processInfo.environment["MLXAUDIO_ZONOS3_MODEL_DIR"]!)
        let loaded = try await TTS.loadModel(modelRepo: directory.path)
        let model = try #require(loaded as? Zonos3Model)
        #expect(model.voices.count == 27)
        #expect(model.sampleRate == 44100)
        #expect(
            model.tokenizer.encode("Hello from Swift!") == [
                2, 45, 1158, 84, 695, 363, 92, 4354, 6, 3,
            ])
        if let path = ProcessInfo.processInfo.environment["MLXAUDIO_ZONOS3_REFERENCE"] {
            let reference = try loadArrays(url: URL(fileURLWithPath: path))
            let state = Zonos3LMState()
            let logits = try model.backbone(
                reference["prompt"]!, state: state,
                speaker: reference["speaker"], conditions: [10, 10, 30])
            // Python and Swift can select different Metal GEMM/GEMV precision paths.
            // The small CPU fixture above checks the same architecture at strict FP32 tolerance.
            close(logits, reference["logits"]!, tolerance: 1.2e-2)
            #expect(abs(logits - reference["logits"]!).mean().item(Float.self) < 1e-3)
            let next = try model.backbone(reference["next_row"]!, state: state)
            close(next, reference["next_logits"]!, tolerance: 1.2e-2)
            #expect(abs(next - reference["next_logits"]!).mean().item(Float.self) < 1e-3)
            let decoder = try Zonos3Codec(directory.appendingPathComponent("codec"))
            close(
                try decoder.decode(reference["codes"]!), reference["codec_audio"]!, tolerance: 3e-3)
            let speaker = try model.extractSpeakerEmbedding(
                reference["reference_pcm"]!.materialized())
            close(speaker, reference["speaker_embedding"]!, tolerance: 3e-3)
        }
        var parameters = model.defaultGenerationParameters
        parameters.temperature = 0
        parameters.maxTokens = 100
        let audio = try await model.generate(
            text: "Hello from Swift!", voice: "jett", generationParameters: parameters)
        #expect(audio.size > 4410)
        #expect(all(isFinite(audio)).item(Bool.self))
        try AudioUtils.writeWavFile(
            samples: audio.asArray(Float.self), sampleRate: model.sampleRate,
            fileURL: URL(fileURLWithPath: "/tmp/zonos3-swift.wav"))
        var chunks: [MLXArray] = []
        for try await event in model.generateStream(
            text: "Hello from Swift!", voice: "jett",
            generationParameters: parameters, streamingInterval: 0.2)
        {
            if case .audio(let chunk) = event { chunks.append(chunk.reshaped(-1)) }
        }
        #expect(chunks.count > 1)
        close(concatenated(chunks), audio, tolerance: 3e-3)
        let reference = audio[0 ..< min(audio.size, 44100)].materialized()
        let speaker = try model.extractSpeakerEmbedding(reference)
        #expect(speaker.shape == [192])
        #expect(all(isFinite(speaker)).item(Bool.self))
        parameters.maxTokens = 45
        let clone = try await model.generate(
            text: "Welcome back.", refAudio: reference, generationParameters: parameters)
        #expect(clone.size > 0)
        #expect(all(isFinite(clone)).item(Bool.self))
    }
}
