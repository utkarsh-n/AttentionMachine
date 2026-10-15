from PIL import Image, ImageDraw
from pathlib import Path
import json

root = Path(__file__).resolve().parents[1]
image = Image.new('RGB', (1024, 1024), '#091923')
draw = ImageDraw.Draw(image)
for i in range(80, 1024, 80):
    draw.line((i, 0, i, 1024), fill='#102631', width=2)
    draw.line((0, i, 1024, i), fill='#102631', width=2)
ports = [(240, 240), (512, 240), (784, 240)]
colors = ['#ff8d7c', '#ad9bff', '#f5cd7a']
for (x, y), color in zip(ports, colors):
    for k in range(4):
        end_x = 310 + k * 135
        draw.line((x, y + 58, end_x, 445), fill=color, width=5)
    draw.ellipse((x-52, y-52, x+52, y+52), fill='#122e39', outline=color, width=7)
    draw.ellipse((x-12, y-12, x+12, y+12), fill=color)
for r in range(4):
    for c in range(4):
        intensity = [0.25, 0.4, 0.9, 0.32][(c+r) % 4]
        color = tuple(int(a * intensity) for a in (119, 233, 207))
        draw.rounded_rectangle((240+c*140, 420+r*100, 365+c*140, 505+r*100), radius=16, fill=color)
draw.line((300, 865, 724, 865), fill='#77e9cf', width=7)
draw.ellipse((495, 848, 529, 882), fill='#edf5ef')
icon = root / 'App/Assets.xcassets/AppIcon.appiconset'
image.save(icon/'Attention.png')
(icon/'Contents.json').write_text(json.dumps({'images':[{'filename':'Attention.png','idiom':'universal','platform':'ios','size':'1024x1024'}],'info':{'author':'xcode','version':1}}, indent=2))
(icon.parent/'Contents.json').write_text('{"info":{"author":"xcode","version":1}}\n')
