# =========================================================
# Compute QCD / VCD / PCD
# =========================================================
from causal_drive_metrics import compute_causal_drives

metrics = compute_causal_drives(
    baseline_score=null_image_with_prefix,                  
    visual_only_score=empty_prompt_with_prefix[target_img_idx],
    joint_scores=cache_with_prefix,                         
    target_visual_id=target_img_idx,
    target_question_id=target_q_idx,
    p_v=Pv,
    p_q_given_v=Pq_given_v,
    prefixes=target_prefixes,
    tokens=target_step_tokens,
    step_interval=2,
)

causal_rows = metrics["per_step"]

avg_qaci = metrics["avg_qcd"]
avg_vaci = metrics["avg_vcd"]
avg_paci = metrics["avg_pcd"]

qaci_curve = metrics["qcd_curve"]
vaci_curve = metrics["vcd_curve"]
paci_curve = metrics["pcd_curve"]
