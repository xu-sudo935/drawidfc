import os
from PIL import Image, ImageDraw, ImageFont
import math

# ================= 配置区域 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "output_viz_figure1")
OUTPUT_FILE = os.path.join(INPUT_DIR, "Figure_1_Pipeline_PIL.png")

# 图片列表 (3行4列)
IMG_FILES = [
    "01_raw_gray.png", "02_binary_inverted.png", "03_morph_connected.png", "04_view_segmentation.png",
    "05_front_view_cropped.png", "06_front_binary.png", "07_solid_detected.png", "08_masked_dashed_raw.png",
    "09_dashed_connected.png", "10_dashed_fitted.png", "11_final_result_vis.png", "12_json_structure.png"
]

TITLES = [
    "(a) Raw Input (Gray Scale)", "(b) Binarization (Inverted)", "(c) Morphology (Connect Lines)", "(d) View Segmentation",
    "(e) Front View (ROI Cropped)", "(f) ROI Binary (Preprocessing)", "(g) Solid Circles (Detection)", "(h) Residual Map (Masking)",
    "(i) Dashed Lines (Connect)", "(j) Circle Fitting (Least Sq.)", "(k) Final Verification", "(l) Data Encapsulation (JSON)"
]

PHASE_LABELS = ["Phase I: View Segmentation", "Phase II: Solid Extraction", "Phase III: Dashed & Final"]

# 布局参数
CELL_W, CELL_H = 800, 600  # 每个子图的标准化尺寸
PADDING = 50               # 图片间距
HEADER_H = 80              # 标题高度
LEFT_MARGIN = 150          # 左侧留给 Phase 标签的空间

def main():
    print(f"正在扫描文件夹: {INPUT_DIR}")
    if not os.path.exists(INPUT_DIR):
        print("错误: 找不到输入文件夹！")
        return

    # 1. 计算大图总尺寸
    cols = 4
    rows = 3
    total_w = LEFT_MARGIN + cols * (CELL_W + PADDING) + PADDING
    total_h = rows * (CELL_H + HEADER_H + PADDING) + PADDING
    
    # 2. 创建白色背景大图
    print(f"创建画布: {total_w}x{total_h} 像素...")
    canvas = Image.new('RGB', (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    # 尝试加载字体 (如果没有 arial，就用默认)
    try:
        font_title = ImageFont.truetype("arial.ttf", 36)
        font_phase = ImageFont.truetype("arialbd.ttf", 40) # 粗体
    except:
        print("未找到 Arial 字体，使用默认字体 (可能较小)")
        font_title = ImageFont.load_default()
        font_phase = ImageFont.load_default()

    # 3. 开始拼图
    for i, filename in enumerate(IMG_FILES):
        row = i // cols
        col = i % cols
        
        # 计算当前格子的坐标
        x_base = LEFT_MARGIN + PADDING + col * (CELL_W + PADDING)
        y_base = PADDING + row * (CELL_H + HEADER_H + PADDING)
        
        # A. 读取并缩放小图
        path = os.path.join(INPUT_DIR, filename)
        if os.path.exists(path):
            try:
                img = Image.open(path)
                # 保持比例缩放以适应格子 (contain模式)
                img.thumbnail((CELL_W, CELL_H), Image.Resampling.LANCZOS)
                # 居中粘贴
                paste_x = x_base + (CELL_W - img.width) // 2
                paste_y = y_base + HEADER_H + (CELL_H - img.height) // 2
                canvas.paste(img, (paste_x, paste_y))
                
                # 画个淡灰色边框美化一下
                draw.rectangle([paste_x-2, paste_y-2, paste_x+img.width+2, paste_y+img.height+2], outline=(200,200,200), width=2)
                
            except Exception as e:
                print(f"图片损坏: {filename} - {e}")
        else:
            print(f"缺失图片: {filename}")
            
        # B. 写子图标题
        title = TITLES[i]
        # 计算文字宽度以居中
        text_bbox = draw.textbbox((0, 0), title, font=font_title)
        text_w = text_bbox[2] - text_bbox[0]
        text_x = x_base + (CELL_W - text_w) // 2
        draw.text((text_x, y_base + 20), title, fill=(0, 0, 0), font=font_title)

        # C. 画箭头 (除了每行最后一个)
        if col < cols - 1:
            arrow_x = x_base + CELL_W + 5
            arrow_y = y_base + HEADER_H + CELL_H // 2
            # 简单的灰色箭头
            draw.line([(arrow_x, arrow_y), (arrow_x + PADDING - 10, arrow_y)], fill=(150,150,150), width=5)
            # 箭头头
            draw.polygon([
                (arrow_x + PADDING - 10, arrow_y - 10),
                (arrow_x + PADDING - 10, arrow_y + 10),
                (arrow_x + PADDING, arrow_y)
            ], fill=(150,150,150))

    # 4. 添加左侧 Phase 标签 (竖向排列比较麻烦，我们用旋转图像的方式)
    for r in range(rows):
        label_text = PHASE_LABELS[r]
        # 创建一个临时图像来画文字
        txt_img = Image.new('RGBA', (600, 100), (255,255,255,0))
        txt_draw = ImageDraw.Draw(txt_img)
        txt_draw.text((10, 10), label_text, font=font_phase, fill=(0, 0, 139)) # 深蓝色
        
        # 旋转90度
        rotated_txt = txt_img.rotate(90, expand=True)
        
        # 粘贴到左侧
        y_center = PADDING + r * (CELL_H + HEADER_H + PADDING) + (CELL_H + HEADER_H) // 2
        dest_x = 20
        dest_y = y_center - rotated_txt.height // 2
        canvas.paste(rotated_txt, (dest_x, dest_y), rotated_txt)

    # 5. 保存
    print(f"正在保存最终大图至: {OUTPUT_FILE}")
    try:
        canvas.save(OUTPUT_FILE)
        print(f"\n✅✅✅ 成功！绝对成功！请查看: {OUTPUT_FILE}")
    except Exception as e:
        print(f"保存依然失败 (这不应该发生): {e}")

if __name__ == "__main__":
    main()
