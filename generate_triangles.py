import colorsys
import ctypes
from itertools import product
from multiprocessing import Process, Array
from PIL import Image, ImageDraw, ImageFilter
import random
from scipy.spatial import Delaunay
import numpy
import sys

# Global parameters that are not (yet) automatically computed
POINT_COUNT = 150
EDGE_THRESHOLD = 172
EDGE_RATIO = .98
DARKENING_FACTOR = 35
SPEEDUP_FACTOR_X = 1
SPEEDUP_FACTOR_Y = 1
CONCURRENCY_FACTOR = 3

def generate_random(im, points):
    prop_x, prop_y = get_point_propagation(*im.size)
    point_distance = get_point_distance(*im.size)
    for _ in range(POINT_COUNT):
        x = int((random.randrange(round((im.size[0] + prop_x) / point_distance)) * \
            point_distance - (prop_x / 2)))
        y = int((random.randrange(round((im.size[1] + prop_y) / point_distance)) * \
            point_distance - (prop_y / 2)))
        if x < 0:
            x=0
        if y < 0:
            y=0
        points.append([x, y])

def get_point_distance(width, height):
    return min(width, height) / 16

def get_point_propagation(width, height):
    return (width / 4, height / 4)

def generate_edges(im, points):
    im_edges = im.filter(ImageFilter.SHARPEN).filter(ImageFilter.FIND_EDGES)
    for x, y in product(range(im.size[0] - 1), range(im.size[1] - 1)):
        if get_grayscale(*im_edges.getpixel((x, y))) > EDGE_THRESHOLD and \
                        random.random() > EDGE_RATIO:
            points.append([x, y])

# get_grayscale returns the gray level of a pixel based on its RGB colors
def get_grayscale(r, g, b):
    return 0.2126*r + 0.7152*g + 0.0722*b

def triangulate(im, points):
    points = numpy.array(points)
    triangles = Delaunay(points)
    avg_colors_of_triangles = []
    triangles_vector = points[triangles.simplices]
    for i in range(len(triangles_vector)):
        colors_in_poly = Barycentric(im, *triangles_vector[i])
        avg_colors = numpy.average(colors_in_poly, axis=0)
        avg_colors = [int(c) for c in avg_colors]  # convert to int
        avg_colors_of_triangles.append(avg_colors)
    return triangles_vector, avg_colors_of_triangles

def PointInsideTriangle2(pt,tri):
    '''checks if point pt(2) is inside triangle tri(3x2). @Developer'''
    a = 1/(-tri[1][1]*tri[2][0]+tri[0][1]*(-tri[1][0]+tri[2][0])+ \
           tri[0][0]*(tri[1][1]-tri[2][1])+tri[1][0]*tri[2][1])
    s = a*(tri[2][0]*tri[0][1]-tri[0][0]*tri[2][1]+(tri[2][1]-tri[0][1])*pt[0]+ \
           (tri[0][0]-tri[2][0])*pt[1])
    if s<0: return False
    else: t = a*(tri[0][0]*tri[1][1]-tri[1][0]*tri[0][1]+(tri[0][1]-tri[1][1])*pt[0]+ \
                 (tri[1][0]-tri[0][0])*pt[1])
    return ((t>0) and (1-s-t>0))

def Barycentric(im, vector1, vector2, vector3):
    colors_in_poly = []
    #print(vector1, vector2, vector3)
    maxY = int(max(vector1[1], vector2[1], vector3[1]))
    minY = int(min(vector1[1], vector2[1], vector3[1]))
    maxX = int(max(vector1[0], vector2[0], vector3[0]))
    minX = int(min(vector1[0], vector2[0], vector3[0]))

    for y in range(minY, maxY+1):
        for x in range(minX, maxX+1):
            if PointInsideTriangle2((x, y), [vector1, vector2, vector3]):
                if x >= 512:
                    x=511
                if y >= 512:
                    y=511
                colors_in_poly.append(im.getpixel((x, y)))
    return colors_in_poly or [im.getpixel((minX, minY))]

def draw(im, triangles_vector, colors):
    d = ImageDraw.Draw(im, "RGB")
    for i in range(len(colors)):
        print([tuple(v) for v in triangles_vector[i]], colors[i])
        d.polygon([tuple(v) for v in triangles_vector[i]], fill=tuple(colors[i]))


if __name__ == "__main__":
    im = Image.open("lena.jpg")
    POINT_COUNT = 300
    # Random point generation
    points = []
    generate_random(im, points)
    #print(points)
    # Generation of "interesting" points
    generate_edges(im, points)
    # print(points)
    # Triangulation and color listing
    triangles_vector, colors = triangulate(im, points)

    print(len(points), len(triangles_vector), len(colors))
    # Final color calculation and drawing
    out_im = Image.new('RGB', im.size)

    draw(out_im, triangles_vector, colors)

    out_im.show()