
import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import json
import shutil
from pathlib import Path
from ultralytics import YOLO
from model_sparse import ViewConsistencyNet

# ==========================================
# 配置参数
# ==========================================
CONFIG = {
    # 模型路径
    "LAYOUT_MODEL": r"D:\drawidfc\best_layout.pt",
    "DETAIL_MODEL": r"D:\drawidfc\best_detail.pt",
    "CONSISTENCY_MODEL": r"D:\drawidfc\1\training_results_original_paramsA+B\best_model.pth",
    
    # 输入输出
    "INPUT_DIR": r"D:\drawidfc\待检测文件",
    "OUTPUT_DIR": r"D:\drawidfc\inference_results1",
    
    # 参数
    "IMG_SIZE": 512,      # 一致性网络输入尺寸
    "CONF_THRESHOLD": 0.25, # YOLO 置信度
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu"
}

class InferencePipeline:
    def __init__(self):
        self.device = CONFIG["DEVICE"]
        print(f"Loading models on {self.device}...")
        
        # 1. 加载 YOLO
        if not os.path.exists(CONFIG["LAYOUT_MODEL"]):
            raise FileNotFoundError(f"Layout model not found: {CONFIG['LAYOUT_MODEL']}")
        if not os.path.exists(CONFIG["DETAIL_MODEL"]):
            raise FileNotFoundError(f"Detail model not found: {CONFIG['DETAIL_MODEL']}")
            
        self.model_layout = YOLO(CONFIG["LAYOUT_MODEL"])
        self.model_detail = YOLO(CONFIG["DETAIL_MODEL"])
        
        # 2. 加载 ConsistencyNet
        self.consistency_net = ViewConsistencyNet(d_model=256)
        if os.path.exists(CONFIG["CONSISTENCY_MODEL"]):
            state_dict = torch.load(CONFIG["CONSISTENCY_MODEL"], map_location=self.device)
            self.consistency_net.load_state_dict(state_dict)
            print("ConsistencyNet weights loaded.")
        else:
            print(f"[Warning] Consistency weights not found at {CONFIG['CONSISTENCY_MODEL']}, using random weights.")
            
        self.consistency_net.to(self.device)
        self.consistency_net.eval()
        
        # 创建输出目录
        if os.path.exists(CONFIG["OUTPUT_DIR"]):
            shutil.rmtree(CONFIG["OUTPUT_DIR"])
        os.makedirs(CONFIG["OUTPUT_DIR"])
        
    def preprocess_tensor(self, img_bgr):
        """
        图片预处理: Resize -> BGR2RGB -> Tensor -> Normalize (0-1)
        """
        img_resized = cv2.resize(img_bgr, (CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"]))
        # BGR -> RGB is optional depending on training. Assuming RGB usually for PyTorch.
        # But dataset.py used cv2.imread (BGR) directly? 
        # Checking dataset.py (Step 17): t_front = torch.from_numpy(img_front).permute(2,0,1)
        # It kept BGR order. I will keep BGR to match training data logic.
        
        img_bhwc = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0)
        img_norm = img_bhwc / 255.0
        return img_norm.to(self.device), img_resized

    def get_view_crop(self, img, bbox, pad_ratio=0.05):
        """
        根据 bbox (x1, y1, x2, y2) 裁剪并 Resize
        增加了 pad_ratio 参数，默认 0.05 (即宽高各扩充 5%)
        """
        h_orig, w_orig = img.shape[:2]
        x1, y1, x2, y2 = map(float, bbox) # 先转 float 计算
        
        # 1. 计算宽高
        w = x2 - x1
        h = y2 - y1
        
        # 2. 计算扩充量 (5%)
        pad_x = w * pad_ratio
        pad_y = h * pad_ratio
        
        # 3. 应用扩充
        x1 = x1 - pad_x
        y1 = y1 - pad_y
        x2 = x2 + pad_x
        y2 = y2 + pad_y
        
        # 4. 转回整数并进行边界保护 (Clamp)
        # 这一步非常重要，防止扩充后坐标变成负数或超过图片大小
        x1, y1 = int(max(0, x1)), int(max(0, y1))
        x2, y2 = int(min(w_orig, x2)), int(min(h_orig, y2))
        
        # 检查有效性
        if x2 <= x1 or y2 <= y1:
            return None, None
            
        # 5. 裁剪
        crop = img[y1:y2, x1:x2]
        h_crop, w_crop = crop.shape[:2]
        
        # Resize 到 512
        crop_resized = cv2.resize(crop, (CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"]))
        
        # 计算缩放比例 (基于扩充后的 crop 尺寸)
        scale_x = w_crop / CONFIG["IMG_SIZE"]
        scale_y = h_crop / CONFIG["IMG_SIZE"]
        
        # 返回 resize 后的图，以及用于还原坐标的信息
        # 注意：这里的 x1, y1 已经是扩充后的左上角坐标，
        # 后续可视化映射回原图时，会自动对齐到正确位置，不需要改 visualize 函数
        return crop_resized, (scale_x, scale_y, x1, y1)

    def run(self):
        results_data = []
        image_files = glob_images(CONFIG["INPUT_DIR"])
        print(f"Found {len(image_files)} images to process.")
        
        for idx, img_path in enumerate(image_files):
            filename = os.path.basename(img_path)
            print(f"Processing [{idx+1}/{len(image_files)}]: {filename}")
            
            img0 = cv2.imread(img_path)
            if img0 is None: continue
            
            # Record for this image
            record = {
                "filename": filename,
                "status": "success",
                "valid": True,
                "global_error": None,
                "circles": []
            }
            
            # ==========================
            # Stage 1: Layout Detection
            # ==========================
            res_layout = self.model_layout(img0, verbose=False)[0]
            boxes_layout = res_layout.boxes
            
            rect_front = None
            rect_left = None
            
            # 简单策略：取置信度最高的 Front(0) 和 Left(1)
            best_conf_front = 0
            best_conf_left = 0
            
            for box in boxes_layout:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy()
                
                if conf < CONFIG["CONF_THRESHOLD"]: continue
                
                if cls_id == 0 and conf > best_conf_front:
                    rect_front = xyxy
                    best_conf_front = conf
                elif cls_id == 1 and conf > best_conf_left:
                    rect_left = xyxy
                    best_conf_left = conf
            
            if rect_front is None or rect_left is None:
                print(f"  [Skip] Missing views: Front={rect_front is not None}, Left={rect_left is not None}")
                record["status"] = "missing_views"
                record["valid"] = False
                results_data.append(record)
                continue
                
            # 裁剪并 Resize 到 512
            img_front_512, info_front = self.get_view_crop(img0, rect_front)
            img_left_512,  _          = self.get_view_crop(img0, rect_left)
            
            if img_front_512 is None or img_left_512 is None:
                record["status"] = "crop_error"
                results_data.append(record)
                continue

            # ==========================
            # Stage 2: Detail Detection
            # ==========================
            # 直接在 512x512 的图上跑 Detail Model
            # 这样得到的坐标 directly corresponds to ConsistencyNet input
            res_detail = self.model_detail(img_front_512, verbose=False)[0]
            boxes_detail = res_detail.boxes
            
            rois_512 = [] # For consistency net [cx, cy, r]
            det_results = [] # For saving/vis
            
            for box in boxes_detail:
                cls_id = int(box.cls[0]) # 0: Solid, 1: Dashed
                conf = float(box.conf[0])
                if conf < CONFIG["CONF_THRESHOLD"]: continue
                
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # 计算 512 尺度下的中心和半径
                cx_512 = (x1 + x2) / 2
                cy_512 = (y1 + y2) / 2
                r_512 = ( (x2-x1) + (y2-y1) ) / 4.0
                
                rois_512.append([cx_512, cy_512, r_512])
                det_results.append({
                    "type": "solid" if cls_id == 0 else "dashed", 
                    "bbox_512": [float(x1), float(y1), float(x2), float(y2)],
                    "center_r_512": [float(cx_512), float(cy_512), float(r_512)]
                })

            if len(rois_512) == 0:
                print("  [Info] No circles detected.")
                # We still run consistency net to get global error
            
            # ==========================
            # Stage 3: Consistency Check
            # ==========================
            t_front, _ = self.preprocess_tensor(img_front_512)
            t_left, _  = self.preprocess_tensor(img_left_512)
            
            # Prepare ROIs tensor
            if len(rois_512) > 0:
                t_rois = torch.tensor(rois_512).float().to(self.device).unsqueeze(0) # [1, N, 3] (Batch list implicit handled if list passed, but model takes list)
                t_rois_list = [torch.tensor(rois_512).float().to(self.device)]
            else:
                t_rois_list = [torch.zeros((0, 3)).float().to(self.device)]
            
            with torch.no_grad():
                # Forward: img_front, img_left, rois
                outs_has, outs_ok, logits_global, _ = self.consistency_net(t_front, t_left, t_rois_list)
            
            # 解析 Global Error
            # logits_global: [1, 2]
            prob_global = F.softmax(logits_global, dim=1)
            is_global_error = prob_global[0, 1].item() > 0.5 # Class 1 = Error
            record["global_error"] = {
                "is_error": bool(is_global_error),
                "prob": float(prob_global[0, 1].item())
            }
            
            # 解析 Local Results
            # outs_has: list of tensors, take [0]
            if len(rois_512) > 0:
                pred_has = F.softmax(outs_has[0], dim=1) # [N, 2]
                pred_ok = F.softmax(outs_ok[0], dim=1)   # [N, 2]
                
                for i, det in enumerate(det_results):
                    p_has = float(pred_has[i, 1].item())
                    p_ok = float(pred_ok[i, 1].item())
                    
                    det["consistency"] = {
                        "has_proj": p_has > 0.5,
                        "has_proj_prob": p_has,
                        "proj_ok": p_ok > 0.5,
                        "proj_ok_prob": p_ok
                    }
                    record["circles"].append(det)

            results_data.append(record)
            
            # ==========================
            # Visualization (On Original)
            # ==========================
            self.visualize(img0, rect_front, rect_left, det_results, info_front, record, filename)

        # Save Final JSON
        json_path = os.path.join(CONFIG["OUTPUT_DIR"], "inference_results.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump({"summary": results_data}, f, indent=2, ensure_ascii=False)
        print(f"\nSaved results to {json_path}")


    def visualize(self, img_orig, rect_front, rect_left, det_results, info_front, record, filename):
        """
        在原图上画框：
        Layout: 蓝色
        Circles: 绿色(一致), 红色(不一致)
        """
        vis_img = img_orig.copy()
        
        # 1. Draw Views (画视图框)
        # 获取左视图坐标 lx1, ly1
        fx1, fy1, fx2, fy2 = map(int, rect_front)
        lx1, ly1, lx2, ly2 = map(int, rect_left)
        
        cv2.rectangle(vis_img, (fx1, fy1), (fx2, fy2), (255, 0, 0), 3) # Blue
        cv2.rectangle(vis_img, (lx1, ly1), (lx2, ly2), (255, 0, 0), 3)
        cv2.putText(vis_img, "Front", (fx1, fy1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(vis_img, "Left", (lx1, ly1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        
        # 2. Draw Circles (画圆和一致性结果)
        scale_x, scale_y, offset_x, offset_y = info_front
        
        for i, det in enumerate(det_results):
            # 将 512 坐标映射回原图
            cx_512, cy_512, r_512 = det["center_r_512"]
            
            # 坐标变换公式
            cx_orig = int(cx_512 * scale_x + offset_x)
            cy_orig = int(cy_512 * scale_y + offset_y)
            # 半径也要缩放 (取平均缩放)
            r_orig = int(r_512 * (scale_x + scale_y) / 2.0)
            
            # 颜色判定
            cons = det.get("consistency", {})
            if cons.get("has_proj") and cons.get("proj_ok"):
                color = (0, 255, 0) # Green (OK)
                status_txt = "OK"
            elif not cons.get("has_proj"):
                color = (0, 0, 255) # Red (Missing Proj)
                status_txt = "NoProj"
            else:
                color = (0, 165, 255) # Orange (Proj Wrong)
                status_txt = "Wrong"
                
            # 实线/虚线区分
            thickness = 2 
            if det["type"] == "dashed":
                status_txt += "(D)"
            else:
                status_txt += "(S)"
                
            cv2.circle(vis_img, (cx_orig, cy_orig), r_orig, color, thickness)
            cv2.putText(vis_img, f"{i}:{status_txt}", (cx_orig-10, cy_orig), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 3. Global Information (修改处：位置改到左视图上方)
        is_err = record["global_error"]["is_error"] if record["global_error"] else False
        global_txt = "Global Error: YES" if is_err else "Global Error: NO"
        color_g = (0, 0, 255) if is_err else (0, 255, 0)
        
        # 计算文字位置：
        # lx1 是左视图左边界
        # ly1 是左视图上边界
        # ly1 - 10 已经被 "Left" 标签占用了
        # 所以我们往上提 40像素，并在最上方保留 40 像素防止画出界
        text_x = lx1
        text_y = max(40, ly1 - 40)
        
        cv2.putText(vis_img, global_txt, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_g, 2)
        
        save_path = os.path.join(CONFIG["OUTPUT_DIR"], f"vis_{filename}")
        cv2.imwrite(save_path, vis_img)

def glob_images(folder):
    exts = ['*.jpg', '*.png', '*.jpeg', '*.bmp']
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(folder, ext)))
        files.extend(glob.glob(os.path.join(folder, ext.upper())))
    return files

import glob

if __name__ == "__main__":
    pipeline = InferencePipeline()
    pipeline.run()
