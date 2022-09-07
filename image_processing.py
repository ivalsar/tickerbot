from PIL import Image, ImageDraw, ImageFont
import dataframe_image as dfi
import pandas as pd
import uuid
import os
from matplotlib import pyplot as plt
import plotly.graph_objects as go


IMAGES_FOLDER = os.path.join(os.getcwd(), 'data/images')


if not os.path.exists(IMAGES_FOLDER):
    os.makedirs(IMAGES_FOLDER)


def apply_watermark(img):
    w, h = img.size
    font_size = min(w, h) // 10
    box_size = min(w, h)

    drawing = ImageDraw.Draw(img)
    font = ImageFont.truetype("data/YuGothB.ttc", font_size)

    watermark_text = "© Largo Winch"
    text_w, text_h = drawing.textsize(watermark_text, font)

    c_text = Image.new('RGB', (text_w + 8, text_h + 8), color='#ffffff')
    drawing = ImageDraw.Draw(c_text)
    drawing.text((4, 4), watermark_text, fill='#eeeeee', font=font, stroke_width=4, stroke_fill='#000000')
    c_text.putalpha(10)
    c_text = c_text.rotate(45, expand=1)

    wm_w, wm_h = c_text.size

    h_offset = (w - wm_w) // 2
    v_offset = (box_size - wm_h) // 2
    max_v_offset = h - wm_h

    while v_offset < max_v_offset:
        pos = h_offset, v_offset
        img.paste(c_text, pos, c_text)
        v_offset += box_size

    return img


def process_image(image_file, output_file, watermark):
    img = Image.open(image_file)
    if watermark:
        img = apply_watermark(img)

    # img = img.convert('RGB')
    # img.save(output_file, format='JPEG')
    img.save(output_file)

    return output_file


def make_image_from_table(s):

    image_id = str(uuid.uuid4()).replace("-", "")

    image_name_png = os.path.join(IMAGES_FOLDER, "img-%s.png" % image_id)
    # image_name_jpg = os.path.join(IMAGES_FOLDER, "img-%s.png" % image_id)
    image_name_png_w = os.path.join(IMAGES_FOLDER, "img-%s-w.png" % image_id)

    dfi.export(s, os.path.join(IMAGES_FOLDER, image_name_png))

    # process_image(image_name_png, image_name_jpg, False)
    process_image(image_name_png, image_name_png_w, True)
    os.remove(image_name_png)

    return image_name_png_w


def make_image_from_figure(f: plt.Figure):

    image_id = str(uuid.uuid4()).replace("-", "")
    image_name_png = os.path.join(IMAGES_FOLDER, "img-%s.png" % image_id)
    image_name_png_w = os.path.join(IMAGES_FOLDER, "img-%s-w.png" % image_id)

    f.savefig(image_name_png)

    process_image(image_name_png, image_name_png_w, True)

    return image_name_png_w


def make_image_from_plotly_figure(f: go.Figure):
    image_id = str(uuid.uuid4()).replace("-", "")
    image_name_png = os.path.join(IMAGES_FOLDER, "img-%s.png" % image_id)
    image_name_png_w = os.path.join(IMAGES_FOLDER, "img-%s-w.png" % image_id)

    f.write_image(image_name_png)

    process_image(image_name_png, image_name_png_w, True)

    return image_name_png_w
