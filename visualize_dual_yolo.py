import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO

# ==========================================
# 1. 配置区域 (请根据实际情况修改路径)
# ==========================================
CONFIG = {
    "INPUT_DIR": r"D:\drawidfc\盘类零件",       # 待检测图片文件夹
    "OUTPUT_DIR": r"./output_visualized", # 可视化结果保存文件夹
    
    "LAYOUT_MODEL": r"best_layout.pt",    # 第一级 YOLO 模型路径 (版面)
    "DETAIL_MODEL": r"best_detail.pt",    # 第二级 YOLO 模型路径 (细节)
    
    "IMG_SIZE": 512,                      # Detail 模型需要的输入尺寸
    "CONF_THRESHOLD": 0.25,               # 置信度阈值
}

# 颜色配置 (B, G, R)
COLORS = {
    "front_box": (255, 0, 0),    # 主视图框：蓝色
    "left_box": (255, 255, 0),   # 左视图框：青色
    "solid_circle": (0, 255, 0), # 实线圆：绿色
    "dashed_circle": (0, 165, 255) # 虚线圆：橙色
}

def get_view_crop_info(img, bbox, pad_ratio=0.05):
    """裁剪视图并计算缩放映射参数"""
    h_orig, w_orig = img.shape[:2]
    x1, y1, x2, y2 = map(float, bbox)
    
    # 扩充边界
    w, h = x2 - x1, y2 - y1
    pad_x, pad_y = w * pad_ratio, h * pad_ratio
    
    x1, y1 = int(max(0, x1 - pad_x)), int(max(0, y1 - pad_y))
    x2, y2 = int(min(w_orig, x2 + pad_x)), int(min(h_orig, y2 + pad_y))
    
    if x2 <= x1 or y2 <= y1:
        return None, None
        
    crop = img[y1:y2, x1:x2]
    h_crop, w_crop = crop.shape[:2]
    
    crop_resized = cv2.resize(crop, (CONFIG["IMG_SIZE"], CONFIG["IMG_SIZE"]))
    
    # 计算用于还原原图坐标的比例和偏移量
    scale_x = w_crop / CONFIG["IMG_SIZE"]
    scale_y = h_crop / CONFIG["IMG_SIZE"]
    mapping_info = (scale_x, scale_y, x1, y1)
    
    return crop_resized, mapping_info

def process_and_visualize():
    os.makedirs(CONFIG["OUTPUT_DIR"], exist_ok=True)
    
    print("📥 正在加载双 YOLO 模型...")
    layout_model = YOLO(CONFIG["LAYOUT_MODEL"])
    detail_model = YOLO(CONFIG["DETAIL_MODEL"])
    
    # 获取图片列表
    exts = ['*.jpg', '*.png', '*.jpeg', '*.bmp']
    img_paths = []
    for ext in exts:
        img_paths.extend(glob.glob(os.path.join(CONFIG["INPUT_DIR"], ext)))
        img_paths.extend(glob.glob(os.path.join(CONFIG["INPUT_DIR"], ext.upper())))
        
    print(f"📋 共找到 {len(img_paths)} 张图片，开始处理...")
    
    for idx, img_path in enumerate(img_paths):
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None: continue
        
        vis_img = img.copy()
        
        # ==========================================
        # 阶段 1: Layout 检测 (找主视图和左视图)
        # ==========================================
        res_layout = layout_model(img, verbose=False)[0]
        
        rect_front, rect_left = None, None
        best_conf_front, best_conf_left = 0, 0
        
        for box in res_layout.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < CONFIG["CONF_THRESHOLD"]: continue
            
            # 0: Front, 1: Left (假设)
            if cls_id == 0 and conf > best_conf_front:
                rect_front = box.xyxy[0].cpu().numpy()
                best_conf_front = conf
            elif cls_id == 1 and conf > best_conf_left:
                rect_left = box.xyxy[0].cpu().numpy()
                best_conf_left = conf

        # 画 Layout 框
        if rect_left is not None:
            lx1, ly1, lx2, ly2 = map(int, rect_left)
            cv2.rectangle(vis_img, (lx1, ly1), (lx2, ly2), COLORS["left_box"], 3)
            cv2.putText(vis_img, "Left View", (lx1, ly1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS["left_box"], 2)
            
        if rect_front is not None:
            fx1, fy1, fx2, fy2 = map(int, rect_front)
            cv2.rectangle(vis_img, (fx1, fy1), (fx2, fy2), COLORS["front_box"], 3)
            cv2.putText(vis_img, "Front View", (fx1, fy1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS["front_box"], 2)
            
            # ==========================================
            # 阶段 2: Detail 检测 (在主视图找圆)
            # ==========================================
            img_front_512, mapping_info = get_view_crop_info(img, rect_front)
            
            if img_front_512 is not None:
                res_detail = detail_model(img_front_512, verbose=False)[0]
                scale_x, scale_y, offset_x, offset_y = mapping_info
                
                # 画圆
                for box in res_detail.boxes:
                    conf = float(box.conf[0])
                    if conf < CONFIG["CONF_THRESHOLD"]: continue
                    
                    cls_id = int(box.cls[0]) # 0: Solid, 1: Dashed
                    
                    # 提取 512 尺度下的中心和半径
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cx_512 = (x1 + x2) / 2
                    cy_512 = (y1 + y2) / 2
                    r_512 = ((x2 - x1) + (y2 - y1)) / 4.0
                    
                    # 核心：将 512 尺度坐标映射回原始大图
                    cx_orig = int(cx_512 * scale_x + offset_x)
                    cy_orig = int(cy_512 * scale_y + offset_y)
                    r_orig = int(r_512 * (scale_x + scale_y) / 2.0)
                    
                    # 确定颜色和标签
                    is_solid = (cls_id == 0)
                    color = COLORS["solid_circle"] if is_solid else COLORS["dashed_circle"]
                    label = "Solid" if is_solid else "Dashed"
                    
                    # 绘制圆和文字
                    thickness = 2 if is_solid else 1 # 虚线画细一点
                    cv2.circle(vis_img, (cx_orig, cy_orig), r_orig, color, thickness)
                    cv2.putText(vis_img, f"{label} {conf:.2f}", (cx_orig-10, cy_orig), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # 保存结果
        save_path = os.path.join(CONFIG["OUTPUT_DIR"], f"result_{filename}")
        cv2.imwrite(save_path, vis_img)
        print(f"[{idx+1}/{len(img_paths)}] 已保存: {save_path}")

    print("✅ 所有图片处理完成！")

if __name__ == "__main__":
    process_and_visualize()
