#!/usr/bin/env python3
"""
YOLO to ONNX Converter Script
将YOLO的.pt文件转换为ONNX格式的脚本
"""

import os
import argparse
from ultralytics import YOLO

def convert_yolo_to_onnx(model_path, output_path=None, img_size=640, simplify=True):
    """
    将YOLO .pt文件转换为ONNX格式
    
    Args:
        model_path (str): YOLO .pt模型文件路径
        output_path (str, optional): 输出ONNX文件路径，如果不指定则自动生成
        img_size (int): 输入图像尺寸，默认640
        simplify (bool): 是否简化ONNX模型，默认True
    
    Returns:
        str: 输出的ONNX文件路径
    """
    # 检查输入文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 加载YOLO模型
    print(f"加载YOLO模型: {model_path}")
    model = YOLO(model_path)
    
    # 生成输出路径
    if output_path is None:
        base_name = os.path.splitext(model_path)[0]
        output_path = f"{base_name}.onnx"
    
    # 转换为ONNX
    print(f"开始转换为ONNX格式...")
    print(f"输入尺寸: {img_size}x{img_size}")
    print(f"简化模型: {simplify}")
    
    # 执行转换
    model.export(
        format='onnx',
        imgsz=img_size,
        simplify=simplify,
        opset=11,  # ONNX opset版本
        dynamic=False,  # 是否支持动态输入尺寸
    )
    
    # 输出文件通常会保存在模型同目录下
    expected_output = os.path.splitext(model_path)[0] + '.onnx'
    
    if os.path.exists(expected_output):
        if output_path != expected_output and output_path != expected_output:
            # 如果指定了不同的输出路径，移动文件
            os.rename(expected_output, output_path)
        else:
            output_path = expected_output
        
        print(f"✅ 转换成功! ONNX文件保存在: {output_path}")
        
        # 显示文件大小信息
        pt_size = os.path.getsize(model_path) / (1024*1024)
        onnx_size = os.path.getsize(output_path) / (1024*1024)
        print(f"📊 文件大小对比:")
        print(f"   原始.pt文件: {pt_size:.2f} MB")
        print(f"   转换后ONNX: {onnx_size:.2f} MB")
        
        return output_path
    else:
        raise Exception("ONNX转换失败，输出文件未生成")

def batch_convert(input_dir, output_dir=None, img_size=640, simplify=True):
    """
    批量转换目录下的所有.pt文件
    
    Args:
        input_dir (str): 包含.pt文件的目录
        output_dir (str, optional): 输出目录，如果不指定则在原目录生成
        img_size (int): 输入图像尺寸
        simplify (bool): 是否简化ONNX模型
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")
    
    # 查找所有.pt文件
    pt_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.pt'):
                pt_files.append(os.path.join(root, file))
    
    if not pt_files:
        print(f"在目录 {input_dir} 中未找到.pt文件")
        return
    
    print(f"找到 {len(pt_files)} 个.pt文件:")
    for pt_file in pt_files:
        print(f"  - {pt_file}")
    
    # 批量转换
    successful_conversions = 0
    for pt_file in pt_files:
        try:
            print(f"\n{'='*60}")
            if output_dir:
                # 保持相对路径结构
                rel_path = os.path.relpath(pt_file, input_dir)
                output_path = os.path.join(output_dir, os.path.splitext(rel_path)[0] + '.onnx')
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            else:
                output_path = None
            
            convert_yolo_to_onnx(pt_file, output_path, img_size, simplify)
            successful_conversions += 1
        except Exception as e:
            print(f"❌ 转换失败 {pt_file}: {str(e)}")
    
    print(f"\n🎉 批量转换完成! 成功转换 {successful_conversions}/{len(pt_files)} 个文件")

def main():
    parser = argparse.ArgumentParser(description='YOLO to ONNX Converter')
    parser.add_argument('input', help='输入.pt文件路径或包含.pt文件的目录')
    parser.add_argument('-o', '--output', help='输出ONNX文件路径或目录')
    parser.add_argument('-s', '--size', type=int, default=640, help='输入图像尺寸 (默认: 640)')
    parser.add_argument('--no-simplify', action='store_true', help='不简化ONNX模型')
    parser.add_argument('--batch', action='store_true', help='批量转换模式 (输入为目录)')
    
    args = parser.parse_args()
    
    try:
        if args.batch or os.path.isdir(args.input):
            # 批量转换模式
            batch_convert(
                input_dir=args.input,
                output_dir=args.output,
                img_size=args.size,
                simplify=not args.no_simplify
            )
        else:
            # 单文件转换模式
            convert_yolo_to_onnx(
                model_path=args.input,
                output_path=args.output,
                img_size=args.size,
                simplify=not args.no_simplify
            )
    except Exception as e:
        print(f"❌ 错误: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())