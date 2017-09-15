from PIL import Image
import sys
filename = sys.argv[1]
im = Image.open(filename).convert("RGB")
pixel = im.load()
for i in range(im.size[0]):
	for j in range(im.size[1]):
		(r,g,b) = pixel[i,j]
		r//=2
		g//=2
		b//=2
		pixel[i,j] = (r,g,b)
im.save('Q2.png',"PNG")