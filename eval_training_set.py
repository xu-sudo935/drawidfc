import os
import json
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import torch.nn.functional as F

# 引入您的一致性网络
try:
    from model_sparse import ViewConsistencyNet
except ImportError:
    print("❌ 错误: 找不到 model_sparse.py，请确保该文件在当前目录下。")
    exit()

# ==========================================
# 1. 配置区域
# ==========================================
CONFIG = {
    # 数据路径
    "JSON_PATH": r"./train_consistency.json",
    "RAW_IMG_DIR": r"D:\drawidfc\盘类零件", 
    
    # 模型路径
    "LAYOUT_MODEL": r"D:\drawidfc\best_layout.pt",
    "DETAIL_MODEL": r"D:\drawidfc\best_detail.pt",
    "CONSISTENCY_MODEL": r"D:\drawidfc\1\training_results_original_paramsA+B\best_model.pth",
    
    # 参数
    "IMG_SIZE": 512,
    "CONF_THRESHOLD": 0.25,     
    "YOLO_NMS_IOU": 0.7,        
    "EVAL_MATCH_IOU": 0.5,      
    
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    
    # 输出配置
    "OUTPUT_JSON": "thesis_data_180_final.json", 
    "SAVE_CROPS_DIR": "saved_view_crops_180",  
    "SAVE_IMAGES": True                        
}

# 创建保存图片的目录
if CONFIG["SAVE_IMAGES"]:
    os.makedirs(CONFIG["SAVE_CROPS_DIR"], exist_ok=True)

# ==========================================
# 2. 核心工具函数 (包含关键修复)
# ==========================================
def find_raw_image_path(json_filename, search_dir):
    """
    智能查找文件: 
    包含修复逻辑：处理 001->1 和 801->801f
    """
    # 提取基础名
    base_name = json_filename.replace("_front.png", "").replace("_front.jpg", "")
    exts = ['.png', '.jpg', '.jpeg', '.bmp']
    
    # 生成候选列表
    candidates = []
    candidates.append(base_name)               # 原始名
    if base_name.isdigit():
        candidates.append(str(int(base_name))) # 去零名
        
    # --- 关键修复：尝试加 'f' 后缀 ---
    candidates.append(base_name + "f")         # 801 -> 801f
    if base_name.isdigit():
        candidates.append(str(int(base_name)) + "f") # 001 -> 1f
    # --------------------------------
    
    # 遍历查找
    for cand in candidates:
        for ext in exts:
            # 1. 直接匹配文件名
            p = os.path.join(search_dir, cand + ext)
            if os.path.exists(p): return p
            
            # 2. 尝试匹配加 _front 的情况
            p_front = os.path.join(search_dir, cand + "_front" + ext)
            if os.path.exists(p_front): return p_front

    return None

def compute_iou(box_a, box_b):
    """计算两个圆(外接矩形)的IoU"""
    def to_rect(c):
        r = c[2]
        return [c[0]-r, c[1]-r, c[0]+r, c[1]+r]
    a = to_rect(box_a)
    b = to_rect(box_b)
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (a[2] - a[0]) * (a[3] - a[1])
    boxBArea = (b[2] - b[0]) * (b[3] - b[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

# ==========================================
# 3. 数据采集管线
# ==========================================
class DataCollector:
    def __init__(self):
        print("📥 正在加载模型...")
        self.layout_model = YOLO(CONFIG["LAYOUT_MODEL"])
        self.detail_model = YOLO(CONFIG["DETAIL_MODEL"])
        self.cons_model = ViewConsistencyNet(d_model=256).to(CONFIG["DEVICE"])
        try:
            ckpt = torch.load(CONFIG["CONSISTENCY_MODEL"], map_location=CONFIG["DEVICE"], weights_only=False)
        except:
            ckpt = torch.load(CONFIG["CONSISTENCY_MODEL"], map_location=CONFIG["DEVICE"])
        if 'model' in ckpt: ckpt = ckpt['model']
        self.cons_model.load_state_dict(ckpt)
        self.cons_model.eval()
        self.img_size = CONFIG["IMG_SIZE"]

    def process_image(self, raw_img_path, flip_mode=0):
        record = {"layout_success": False, "predictions": None, "error_msg": None}

        # 1. 读取
        raw_img = cv2.imread(raw_img_path)
        if raw_img is None: 
            record["error_msg"] = "io_error"
            return record

        # 获取真实的文件名 (例如 801f) 用于保存
        real_base_name = os.path.splitext(os.path.basename(raw_img_path))[0]

        # 2. Layout YOLO
        layout_res = self.layout_model.predict(
            raw_img, conf=CONFIG["CONF_THRESHOLD"], iou=CONFIG["YOLO_NMS_IOU"], verbose=False
        )[0]
        
        box_front, box_left = None, None
        for box in layout_res.boxes:
            cls_name = self.layout_model.names[int(box.cls[0])].lower()
            coords = box.xyxy[0].cpu().numpy().astype(int)
            if 'front' in cls_name: box_front = coords
            elif 'left' in cls_name: box_left = coords

        if box_front is None or box_left is None:
            record["error_msg"] = "layout_fail"
            return record
        
        record["layout_success"] = True

        # 3. Crop & Flip
        img_front = raw_img[box_front[1]:box_front[3], box_front[0]:box_front[2]]
        img_left = raw_img[box_left[1]:box_left[3], box_left[0]:box_left[2]]

        suffix = ""
        if flip_mode == 1:
            img_front = cv2.flip(img_front, 0)
            img_left = cv2.flip(img_left, 0)
            suffix = "_flip"

        # --- 保存图片逻辑 ---
        if CONFIG["SAVE_IMAGES"]:
            # 使用找到的真实文件名 (real_base_name) 保存，这样就能看到 801f_front.png 了
            f_name = f"{real_base_name}{suffix}_front.png"
            l_name = f"{real_base_name}{suffix}_left.png"
            cv2.imwrite(os.path.join(CONFIG["SAVE_CROPS_DIR"], f_name), img_front)
            cv2.imwrite(os.path.join(CONFIG["SAVE_CROPS_DIR"], l_name), img_left)
        # ------------------

        # 4. Resize & Predict (后续逻辑保持不变)
        img_front_512 = cv2.resize(img_front, (self.img_size, self.img_size))
        img_left_512 = cv2.resize(img_left, (self.img_size, self.img_size))

        detail_res = self.detail_model.predict(
            img_front_512, conf=CONFIG["CONF_THRESHOLD"], iou=CONFIG["YOLO_NMS_IOU"], verbose=False
        )[0]
        
        pred_circles = []
        for box in detail_res.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx, cy = (x1 + x2)/2, (y1 + y2)/2
            w, h = x2 - x1, y2 - y1
            r = (w + h) / 4.0
            pred_circles.append([cx, cy, r])

        def to_tensor(img):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0
            img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
            img = np.transpose(img, (2, 0, 1))
            return torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(CONFIG["DEVICE"])

        t_front = to_tensor(img_front_512)
        t_left = to_tensor(img_left_512)
        t_rois = [torch.tensor(pred_circles, dtype=torch.float32).to(CONFIG["DEVICE"])]

        with torch.no_grad():
            p_has, p_ok, p_global, _ = self.cons_model(t_front, t_left, t_rois)

        prob_global = F.softmax(p_global, dim=1)
        global_pred_cls = torch.argmax(prob_global, dim=1).item()
        global_prob_err = prob_global[0, 1].item()

        circle_res = []
        if len(pred_circles) > 0 and len(p_has) > 0:
            probs_has = F.softmax(torch.cat(p_has, dim=0), dim=1).cpu().numpy()
            probs_ok = F.softmax(torch.cat(p_ok, dim=0), dim=1).cpu().numpy()
            for i, roi in enumerate(pred_circles):
                circle_res.append({
                    'roi': [float(x) for x in roi],
                    'pred_has': int(np.argmax(probs_has[i])),
                    'pred_ok': int(np.argmax(probs_ok[i])),
                    'prob_has': float(probs_has[i, 1]),
                    'prob_ok': float(probs_ok[i, 1])
                })
        
        record["predictions"] = {
            "global_pred": global_pred_cls,
            "global_prob_error": global_prob_err,
            "circles": circle_res
        }
        return record

# ==========================================
# 4. 主程序
# ==========================================
if __name__ == "__main__":
    print(f"🚀 开始数据采集 + 图片保存")
    print(f"📂 图片保存目录: {CONFIG['SAVE_CROPS_DIR']}")
    
    with open(CONFIG["JSON_PATH"], 'r', encoding='utf-8') as f:
        full_data = json.load(f)["images"]
        
    test_tasks = []
    indices = list(range(0, len(full_data), 10))
    for idx in indices:
        test_tasks.append({'idx': idx, 'flip': 0})
        img_id = idx + 1
        if (501 <= img_id <= 600) or (img_id >= 701):
            test_tasks.append({'idx': idx, 'flip': 1})
            
    print(f"📋 任务数: {len(test_tasks)}")
    
    collector = DataCollector()
    final_data = []

    for i, task in enumerate(test_tasks):
        gt_item = full_data[task['idx']]
        flip = task['flip']
        
        raw_path = find_raw_image_path(gt_item['filename'], CONFIG["RAW_IMG_DIR"])
        
        if raw_path:
            res = collector.process_image(raw_path, flip)
        else:
            print(f"⚠️ 找不到文件: {gt_item['filename']}")
            res = {"layout_success": False, "error_msg": "file_not_found"}

        # 构造 GT 和 Pred 的简单记录 (用于匹配)
        # 这里简化处理，因为 JSON 主要是给分析用的，重点是 process_image 里的保存
        # 如果需要完整论文数据，下面的匹配逻辑保持不变...
        
        gt_circles = []
        W, H = CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"]
        for c in gt_item.get('circles', []):
            box = c['roi']
            nx, ny = box[0], box[1]
            if flip == 1: ny = 1.0 - ny
            cx, cy = nx * W, ny * H
            r = ((box[2]*W) + (box[3]*H)) / 4.0
            gt_circles.append({'roi': [cx, cy, r], 'has': int(c['has_proj']), 'ok': int(c['proj_ok'])})

        matched_pairs = []
        available_preds = []
        if res.get("predictions"):
             for p_idx, p in enumerate(res["predictions"]["circles"]):
                 p_copy = p.copy()
                 p_copy['_index'] = p_idx
                 available_preds.append(p_copy)
        
        for g_idx, g in enumerate(gt_circles):
            best_iou = 0
            best_p_idx = -1
            for p_idx, p in enumerate(available_preds):
                iou = compute_iou(p['roi'], g['roi'])
                if iou > best_iou:
                    best_iou = iou
                    best_p_idx = p_idx
            
            match_info = {"gt_idx": g_idx, "gt_data": g, "best_iou": float(best_iou), "matched": False, "pred_data": None}
            if best_iou >= CONFIG['EVAL_MATCH_IOU'] and best_p_idx != -1:
                match_info["matched"] = True
                match_info["pred_data"] = available_preds[best_p_idx]
                available_preds.pop(best_p_idx)
            matched_pairs.append(match_info)

        unmatched_preds = available_preds
        gt_global = 1 if gt_item.get('global_error', 0) == 1 else 0

        entry = {
            "image_id": gt_item['filename'],
            "flip_mode": flip,
            "status": {"layout_success": res.get("layout_success", False), "error": res.get("error_msg")},
            "global": {"gt": gt_global, "pred": res["predictions"]["global_pred"] if res.get("predictions") else -1},
            "matching_results": matched_pairs,
            "unmatched_predictions": unmatched_preds
        }
        final_data.append(entry)

        if (i+1) % 20 == 0:
            print(f"⏳ 进度: {i+1}/{len(test_tasks)}")

    with open(CONFIG["OUTPUT_JSON"], 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)
        
    print(f"\n✅ 全部完成！请检查文件夹: {CONFIG['SAVE_CROPS_DIR']}")
