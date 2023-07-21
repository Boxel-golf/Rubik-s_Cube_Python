from tkinter import*
import math
import numpy as np
import time
import copy
import random

class Triangle:

    def __init__(self, dot1, dot2, dot3, colour, Canvas=Canvas):
        self.dot1 = np.array(dot1)
        self.dot2 = np.array(dot2)
        self.dot3 = np.array(dot3)
        self.colour = colour
        self.pos = self.update_pos()
        
    def move(self, x, y, z, update=True):
        vector = np.array([x, y, z])
        self.dot1 = self.dot1 + vector
        self.dot2 = self.dot2 + vector
        self.dot3 = self.dot3 + vector
        if update:
            self.pos = self.update_pos()

    def rotate(self, x, y, z, rotate_around):
        if rotate_around == 'self':
            rotate_around = np.array(self.pos)

        start = copy.copy(self.pos)
        self.move(-rotate_around[0], -rotate_around[1], -rotate_around[2])

        x = math.radians(x)
        y = math.radians(y)
        z = math.radians(z)

        rotation_matrix = np.dot(np.dot(
            np.array([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]]),
            np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])),
            np.array([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]]))

        vertices = np.vstack([self.dot1, self.dot2, self.dot3]).T
        vertices = np.dot(rotation_matrix, vertices).T
        self.dot1, self.dot2, self.dot3 = vertices

        self.move(rotate_around[0], rotate_around[1], rotate_around[2])
        self.pos = self.update_pos()

    def inverse_rotate(self, x, y, z, rotate_around):
        if rotate_around == 'self':
            rotate_around = np.array(self.pos)

        start = copy.copy(self.pos)
        self.move(-rotate_around[0], -rotate_around[1], -rotate_around[2])

        x = math.radians(x)
        y = math.radians(y)
        z = math.radians(z)

        rotation_matrix = np.dot(np.dot(
            np.array([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]]),
            np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])),
            np.array([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]])).T

        vertices = np.vstack([self.dot1, self.dot2, self.dot3]).T
        vertices = np.dot(rotation_matrix, vertices).T
        self.dot1, self.dot2, self.dot3 = vertices

        self.move(rotate_around[0], rotate_around[1], rotate_around[2])
        self.pos = self.update_pos()
        
    def plot(self, camera, focal_length=50):
        if self.colour != 'black':
            X = [self.dot1[0], self.dot2[0], self.dot3[0], self.dot1[0]]
            
            Y = [self.dot1[1], self.dot2[1], self.dot3[1], self.dot1[1]]
            
            Z = [self.dot1[2], self.dot2[2], self.dot3[2], self.dot1[2]]

            for item, index in zip(X, range(len(X))):
                X[index] = (focal_length * item) / (focal_length + Z[index])
                X[index] += camera.pos[0]

            for item, index in zip(Y, range(len(Y))):
                Y[index] = (focal_length * item) / (focal_length + Z[index])
                Y[index] += camera.pos[1]

            for item, index in zip(Z, range(len(Z))):
                Z[index] += camera.pos[2]

            polygon_data = []
            for Xitem, Yitem in zip(X, Y):
                polygon_data.append(Xitem)
                polygon_data.append(Yitem)
                
            id = screen.create_polygon(polygon_data[0], polygon_data[1],
                                       polygon_data[2], polygon_data[3],
                                       polygon_data[4], polygon_data[5],
                                       polygon_data[6], polygon_data[7],
                                       fill=self.colour)

    def update_pos(self, update=False):
        midX = (self.dot1[0] + self.dot2[0] + self.dot3[0]) / 3
        midY = (self.dot1[1] + self.dot2[1] + self.dot3[1]) / 3
        midZ = (self.dot1[2] + self.dot2[2] + self.dot3[2]) / 3
        if update:
            self.pos = [midX, midY, midZ]
        return [midX, midY, midZ]

class Camera:

    def __init__(self, pos):
        self.pos = pos
        self.FOV = 90

class Scene:

    def __init__(self, objects, names):
        self.objects = objects
        self.names = names

    def add_object(self, new_object, new_name):
        self.objects.append(new_object)
        self.names.append(new_name)

    def plot(self, camera, focal_length):
        triangles = []
        for item in self.objects:
            item.update_pos()
        for item in self.objects:
            if type(item) == Group:
                new_items = item.plot(camera, focal_length)
            else:
                new_items = [item]
            for tri in new_items:
                triangles.append(tri)
        triangles.sort(key=lambda x: -x.pos[2])
        for item in triangles:
            item.plot(camera, focal_length)

    def move(self, item, x, y, z):
        if item[0] == 'all':
            for thing in self.objects:
                if isinstance(thing, Triangle):
                    thing.move(x, y, z)
                else:
                    thing.move(['all'], x, y, z)
        else:
            try:
                if isinstance(item[0], Triangle):
                    self.objects[self.names.index(item[0])].move(x, y, z)
                else:
                    if len(item) > 2:
                        self.objects[self.names.index(item[0])].move(item[1:], x, y, z)
                    else:
                        self.objects[self.names.index(item[0])].move([item[1]], x, y, z)
            except:
                pass

    def rotate(self, item, x, y, z, rotate_around=None, inverse=False):
        triangles = []
        if item[0] == 'all':
            for thing in self.objects:
                if isinstance(thing, Triangle):
                    triangles.append(thing)
                else:
                    new_triangles = thing.rotate(['all'], x, y, z,
                                                 rotate_around=[0, 0, 0])
                    for tri in new_triangles:
                        triangles.append(tri)
        else:
            if isinstance(item[0], Triangle):
                triangles.append(self.objects[self.names.index(item[0])])
            else:
                if len(item) > 2:
                    new_triangles = self.objects[self.names.index(item[0])].rotate(
                                                            item[1:], x, y, z)
                    for tri in new_triangles:
                        triangles.append(tri)
                        
                else:
                    new_triangles = self.objects[self.names.index(item[0])].rotate(
                                                            [item[1]], x, y, z)
                    for tri in new_triangles:
                        triangles.append(tri)

        if rotate_around == None:
            if item[0] == 'all':
                rotate_around = [0, 0, 0]
            else:
                rotating_object = self
                for i in range(0, len(item)):
                    if item[i] != 'all':
                        rotating_object = rotating_object.objects[rotating_object.names.index(item[i])]
                rotating_object.update_pos()
                rotate_around = rotating_object.pos
        else:
            pass
            
        for triangle in triangles:
            if inverse == False:
                triangle.rotate(x, y, z, rotate_around)
            else:
                triangle.inverse_rotate(x, y, z, rotate_around)
    

        

    def save(self, file):
        with open(file, 'wb') as f:
            pickle.dump(copy.deepcopy(self), f)

    def load(self, file):
        with open(file, 'rb') as f:
              self = pickle.load(f)

class Group:
    def __init__(self, objects=[], names=[]):
        self.objects = objects.copy()
        self.names = names.copy()
        self.update_pos()

    def add_object(self, new_object, new_name):
        self.objects.append(new_object)
        self.names.append(new_name)

    def plot(self, camera, focal_length):
        triangles = []
        for item in self.objects:
            item.update_pos()
        for item in self.objects:
            if type(item) == Group:
                new_items = item.plot(camera, focal_length)
            else:
                new_items = [item]
            for tri in new_items:
                triangles.append(tri)
        return triangles

    def rotate(self, item, x, y, z, rotate_around=None):
        if rotate_around == None:
            rotate_around = self.pos
        triangles = []
        if item[0] == 'all':
            self.update_pos()
            for thing in self.objects:
                if isinstance(thing, Triangle):
                    triangles.append(thing)
                else:
                    new_triangles = thing.rotate(['all'], x, y, z,
                                                 rotate_around=rotate_around)
                    for tri in new_triangles:
                        triangles.append(tri)
        else:
            try:
                if isinstance(item[0], Triangle):
                    self.objects[self.names.index(item[0])].rotate(x, y, z, 'self')
                else:
                    if len(item) > 2:
                        new_triangles = self.objects[self.names.index(item[0])].rotate(item[1:], x, y, z)
                        for tri in new_triangles:
                            triangles.append(tri)
                    else:
                        new_triangles = self.objects[self.names.index(item[0])].rotate([item[1]], x, y, z)
                        for tri in new_triangles:
                            triangles.append(tri)
            except:
                pass
            
        return triangles
        
                
    def move(self, item, x, y, z):
        if item[0] == 'all':
            for thing in self.objects:
                if isinstance(thing, Triangle):
                    thing.move(x, y, z)
                else:
                    thing.move(['all'], x, y, z)
        else:
            try:
                if isinstance(item[0], Triangle):
                    self.objects[self.names.index(item[0])].move(x, y, z)
                else:
                    if len(item) > 2:
                        self.objects[self.names.index(item[0])].move(item[1:], x, y, z)
                    else:
                        self.objects[self.names.index(item[0])].move([item[1]], x, y, z)
            except:
                pass
        self.update_pos()

    def update_pos(self, update=True):
        objects_Xpos = []
        objects_Ypos = []
        objects_Zpos = []
        for item in self.objects:
            item_pos = item.update_pos(True)
            objects_Xpos.append(item_pos[0])
            objects_Ypos.append(item_pos[1])
            objects_Zpos.append(item_pos[2])

        if objects_Xpos != []:
            if update:
                self.pos = [sum(objects_Xpos) / len(objects_Xpos),
                            sum(objects_Ypos) / len(objects_Ypos),
                            sum(objects_Zpos) / len(objects_Zpos)]
            return [sum(objects_Xpos) / len(objects_Xpos),
                    sum(objects_Ypos) / len(objects_Ypos),
                    sum(objects_Zpos) / len(objects_Zpos)]

    def save(self, file):
        with open(file, 'wb') as f:
            pickle.dump(copy.deepcopy(self), f)

    def load(self, file):
        with open(file, 'rb') as f:
              self = pickle.load(f)

class Turn:
    def __init__(self, name, axis, axis_pos, direction):
        self.name = name
        self.axis = axis
        self.axis_pos = axis_pos
        self.direction = direction
        self.pieces = []
        self.piece_types = []
        self.orders = []
        
    def add_piece_type(self, piece_type):
        self.piece_types.append(piece_type)
        self.pieces.append([])
        self.orders.append([])

    def add_piece(self, piece_type, pos):
        if not (piece_type in self.piece_types):
            self.add_piece_type(piece_type)
        self.pieces[self.piece_types.index(piece_type)].append(pos)

    def cube_to_index(self, piece):
        to_return = ((int(piece[0]) - 1) * 9) + ((int(piece[1]) - 1) * 3) + (int(piece[2]) - 1)
        return to_return

    def index_to_cube(self, index):
        x = round((index + 4) / 9)
        z = ((x - 1) % 3) + 1
        y = (((x - z) % 9) / 3) + 1
        to_return = [int(x), int(y), int(z)]
        return to_return

    def make_order(self, cubes, scene):
        start_scene = []
        for thing in scene.objects:
            start_scene.append([round(copy.copy(thing.pos)[0] / 67) + 3,
                                round(copy.copy(thing.pos)[1] / 67) + 3,
                                round(copy.copy(thing.pos)[2] / 67) + 3])
    
        for d in range(0, len(self.pieces)):
            for b in self.pieces[d]:
                scene.rotate([cubes[self.cube_to_index(b)], 'all'],
                                self.direction[0] * 90,
                                self.direction[1] * 90,
                                self.direction[2] * 90,
                                [-50, -50, -50])
        scene.plot(camera, 600)

        end_scene = []
        for thing in scene.objects:
            end_scene.append([round(copy.copy(thing.pos)[0] / 67) + 3,
                              round(copy.copy(thing.pos)[1] / 67) + 3,
                              round(copy.copy(thing.pos)[2] / 67) + 3])

            

        for i in range(len(self.pieces)):
            if len(self.pieces[i]) != 1:
                for f in range(len(self.pieces[i])):
                    self.orders[i].append([])
                self.orders[i][0] = self.pieces[i][0]
                    
                for z in range(1, len(self.pieces[i])):
                    self.orders[i][z] = end_scene[start_scene.index(self.orders[i][z-1])]
                        
                for o in range(len(self.orders[i])):
                    self.orders[i][o] = self.pieces[i].index(self.orders[i][o])

        for d in range(0, len(self.pieces)):
            for b in self.pieces[d]:
                scene.rotate([cubes[self.cube_to_index(b)], 'all'],
                                self.direction[0] * -90,
                                self.direction[1] * -90,
                                self.direction[2] * -90,
                                [-50, -50, -50])
        scene.plot(camera, 600)

    def change_corner_ori(self, piece):
        if self.name == ('U' or 'D'):
            if piece == 0:
                pass
            elif piece == 1:
                piece = 2
            elif piece == 2:
                piece = 1
        
        elif self.name == ('F' or 'B'):
            if piece == 1:
                pass
            elif piece == 0:
                piece = 2
            elif piece == 2:
                piece = 0

        elif self.name == ('R' or 'L'):
            if piece == 2:
                pass
            elif piece == 0:
                piece = 1
            elif piece == 1:
                piece = 0

        return piece
        

    def turn(self, amount, cubes, scene, scene_rotation, frames, oris):
        while amount > 2:
            amount -= 4
        start_cubes = copy.copy(cubes)
        start_oris = copy.copy(oris)
        if amount == 2:
            frames = round(frames * 1.5)
        
        for c in range(0, frames):
            scene.rotate(['all'], scene_rotation[0], scene_rotation[1], scene_rotation[2], [-50, -50, -50], True)
            for d in range(0, len(self.pieces)):
                for b in self.pieces[d]:
                    scene.rotate([cubes[self.cube_to_index(b)], 'all'],
                                    self.direction[0] * amount * (90 / frames),
                                    self.direction[1] * amount * (90 / frames),
                                    self.direction[2] * amount * (90 / frames),
                                    [-50, -50, -50])
                    

            scene.rotate(['all'], scene_rotation[0], scene_rotation[1], scene_rotation[2], [-50, -50, -50])
            
            scene.plot(camera, 600)
            screen.update()
            screen.delete('all')

        for i in range(len(self.pieces)):
            if len(self.pieces[i]) != 1:

                buffer = start_cubes[self.cube_to_index(self.pieces[i][self.orders[i][0]])]
                ori_buffer = start_oris[self.cube_to_index(self.pieces[i][self.orders[i][0]])]

                for j in range(0, len(self.orders[i])):
                    try:
                        cubes[self.cube_to_index(self.pieces[i][self.orders[i][j]])] = start_cubes[self.cube_to_index(self.pieces[i][self.orders[i][j - amount]])]
                        oris[self.cube_to_index(self.pieces[i][self.orders[i][j]])] = start_oris[self.cube_to_index(self.pieces[i][self.orders[i][j - amount]])]
                            
                    except:     
                        cubes[self.cube_to_index(self.pieces[i][self.orders[i][j]])] = buffer
                        oris[self.cube_to_index(self.pieces[i][self.orders[i][j]])] = ori_buffer

                if amount != 2:
                    for j in range(0, len(self.orders[i])):
                        if (self.piece_types[i] == 'edge') and (self.name == ('F' or 'B' or 'M' or 'S' or 'E')):
                            oris[self.cube_to_index(self.pieces[i][self.orders[i][j]])] += 1
                            while oris[self.cube_to_index(self.pieces[i][self.orders[i][j]])] > 1:
                                oris[self.cube_to_index(self.pieces[i][self.orders[i][j]])] -= 2

                        elif (self.piece_types[i] == 'corner'):
                            oris[self.cube_to_index(self.pieces[i][j])] = self.change_corner_ori(oris[self.cube_to_index(self.pieces[i][j])])
                    
                
        return cubes, oris
                    

def make_cube(pos, colours, name, size):
    global Group, scene
    size -= 0.5
    cube = Group([], [])
    left = -(size / 2)
    right = (size / 2)

    points = [(left,left,left), (left,right,left), (right,right,left), (right,left,left),
              (left,left,right), (left,right,right), (right,right,right), (right,left,right)]

    triangles = [Triangle(points[0], points[1], points[3], colours[0]),
                 Triangle(points[1], points[2], points[3], colours[0]),
                 Triangle(points[1], points[5], points[2], colours[1]),
                 Triangle(points[5], points[6], points[2], colours[1]),
                 Triangle(points[4], points[5], points[0], colours[2]),
                 Triangle(points[5], points[1], points[0], colours[2]),
                 Triangle(points[3], points[2], points[6], colours[3]),
                 Triangle(points[3], points[6], points[7], colours[3]),
                 Triangle(points[4], points[0], points[7], colours[4]),
                 Triangle(points[0], points[3], points[7], colours[4]),
                 Triangle(points[6], points[5], points[4], colours[5]),
                 Triangle(points[6], points[4], points[7], colours[5])]              
    
    for i, colour in zip(range(0, 12, 2), colours):
        square = Group([], [])
        square.add_object(triangles[i], 'tri1')
        
        square.add_object(triangles[i + 1], 'tri2')
        
        cube.add_object(square, colour + ' square')


    scene.add_object(cube, name)
    scene.move([name, 'all'], pos[0], pos[1], pos[2])
    cube.update_pos()

def make_nxn_cube(nxn_pos, colours, n, size, scene):
    turns = []
    for x in range(1, n+1):
        x_copy = copy.copy(x)
        turn_direction = [1, 0, 0]
        turn_name = 'L'
        if x > (n / 2) + 0.5:
            x_copy = round((x_copy - (n/2)) - (2 * (x_copy - (n/2))) + (n/2) + 1)
            turn_name = 'R'
            turn_direction = [-1, 0, 0]
        if x == (n / 2) + 0.5:
            turn_name = 'M'
        else:
            if x_copy != 1:
                turn_name = str(x_copy) + turn_name
                
        turns.append(Turn(turn_name, 1, x, turn_direction))

    for y in range(1, n+1):
        y_copy = copy.copy(y)
        turn_direction = [0, 1, 0]
        turn_name = 'U'
        if y > (n / 2) + 0.5:
            y_copy = round((y_copy - (n/2)) - (2 * (y_copy - (n/2))) + (n/2) + 1)
            turn_name = 'D'
            turn_direction = [0, -1, 0]
        if y == (n / 2) + 0.5:
            turn_name = 'E'
        else:
            if y_copy != 1:
                turn_name = str(y_copy) + turn_name
            
        turns.append(Turn(turn_name, 2, y, turn_direction))

    for z in range(1, n+1):
        z_copy = copy.copy(z)
        turn_direction = [0, 0, 1]
        turn_name = 'F'
        if z > (n / 2) + 0.5:
            z_copy = round((z_copy - (n/2)) - (2 * (z_copy - (n/2))) + (n/2) + 1)
            turn_name = 'B'
            turn_direction = [0, 0, -1]
        if z == (n / 2) + 0.5:
            turn_name = 'S'
        else:
            if z_copy != 1:
                turn_name = str(z_copy) + turn_name
            
        turns.append(Turn(turn_name, 3, z, turn_direction))
    
    cube_size = size / n
    names = []
    cubes = []
    pieces_in_turn = []
    for i in range(len(turns)):
        pieces_in_turn.append(0)
    for x in range(1, n+1):
        for y in range(1, n+1):
            for z in range(1, n+1):
                if x == 1 or x == n or y == 1 or y == n or z == 1 or z == n:
                    curr_colours = copy.copy(colours)
                    curr_name = ''
                    if x != 1:
                        curr_colours[2] = 'black'
                    if x != n:
                        curr_colours[3] = 'black'
                    if y != 1:
                        curr_colours[4] = 'black'
                    if y != n:
                        curr_colours[1] = 'black'
                    if z != 1:
                        curr_colours[0] = 'black'
                    if z != n:
                        curr_colours[5] = 'black'

                    if x == 1:
                        curr_name += 'o'
                    if y == 1:
                        curr_name += 'w'
                    if z == 1:
                        curr_name += 'g'
                    if z == n:
                        curr_name += 'b'
                    if y == n:
                        curr_name += 'y'
                    if x == n:
                        curr_name += 'r'

                    name_no_number = copy.copy(curr_name)
                    if curr_name in names:
                        curr_name += '1'
                    i = 2
                    while curr_name in names:
                        curr_name = name_no_number + str(i)
                        i += 1
                    names.append(curr_name)

                    if len(name_no_number) == 3:
                        curr_piece_type = 'corner'
                    if len(name_no_number) == 2:
                        curr_piece_type = 'edge'
                    if len(name_no_number) == 1:
                        curr_piece_type = 'center'
                    
                    index = names.index(curr_name)
                    matrix_pos = [int(x), int(y), int(z)]

                    curr_turns = []
                    curr_turns_place = []
                    for turn in turns:
                        if matrix_pos[turn.axis-1] == turn.axis_pos:
                            turn.add_piece(curr_piece_type, matrix_pos)
                        
                    curr_pos = [((x * cube_size) - (cube_size / 2)) + nxn_pos[0],
                               ((y * cube_size) - (cube_size / 2)) + nxn_pos[1],
                               ((z * cube_size) - (cube_size / 2)) + nxn_pos[2]]
                               
                    make_cube(curr_pos, curr_colours, curr_name, cube_size)
                    
                else:
                    names.append('empty')
                    curr_pos = nxn_pos
                    curr_colours = ['black',
                                    'black',
                                    'black',
                                    'black',
                                    'black',
                                    'black',]
                    make_cube(curr_pos, curr_colours, 'empty', cube_size)
                    
    turn_names = []
    for turn in turns:
        turn_names.append(turn.name)
        turn.make_order(names, scene)

    return turns, turn_names, names

def run_alg(alg, cubes, turns, turn_names, scene, scene_rotation, frames, oris):
    i = 0
    while i < len(alg):
        curr_turn = alg[i]
        try:
            if alg[i + 1] == "2":
                amount = 2
                i += 3
            elif alg[i + 1] == "'":
                amount = 3
                i += 3
            else:
                amount = 1
                i += 2
        except:
            amount = 1
            i += 2
        cubes = turns[turn_names.index(curr_turn)].turn(amount, cube_names, scene, scene_rotation, frames, oris)

    return cubes

def reverse_alg(alg):
    alg_list = []
    reverse_alg_list = []
    i = 0
    while i < len(alg):
        curr_turn = alg[i]
        try:
            if alg[i + 1] == "2":
                amount = 2
                i += 3
            elif alg[i + 1] == "'":
                amount = 3
                i += 3
            else:
                amount = 1
                i += 2
        except:
            amount = 1
            i += 2
        if amount == 1:
            alg_list.append(str(curr_turn) + "'")
        elif amount == 3:
            alg_list.append(str(curr_turn))
        else:
            alg_list.append(str(curr_turn) + str(amount))

    i = len(alg_list) - 1
    while i > -1:
        reverse_alg_list.append(alg_list[i])
        i -= 1

    reverse_alg = ''
    for move in reverse_alg_list:
        reverse_alg += str(move) + ' '
    
    return reverse_alg

def make_ori_state():
    to_return = []
    for i in range(0, 27):
        to_return.append(0)
    return to_return
                

if __name__ == '__main__':
    root = Tk()
    screen = Canvas(root, bg="Black", height=600, width=600)
    screen.pack()
    camera = Camera([350, 350, -50])
    scene = Scene([], [])
    turns, turn_names, cube_names = make_nxn_cube([-150, -150, -150],
                                                  ['green', 'yellow', 'orange', 'red', 'white', 'blue'], 3, 200, scene)

    scene_rotation = [45, 25, 0]
    

    scene.rotate(['all'], scene_rotation[0], scene_rotation[1], scene_rotation[2], [-50, -50, -50])
    scene.plot(camera, 600)
    screen.update()
    screen.delete('all')

    ori_state = make_ori_state()

    y_perm = "F R U' R' U' R U R' F' R U R' U' R' F R F'"
    t_perm = "R U R' U' R' F R2 U' R' U' R U R' F'"
    r_perm = "R U' R' U' R U R D R' U' R D' R' U2 R'"

    index = 'ABCDEFGHIJKLMNOPQRSTUVWX'

    solved_state = copy.copy(cube_names)

    corner_setups = [None, "R'", "R' F'", "F' D F'", None, "F' D R", "D F'", 'D F2', 'F', 'R2 R2', "F'", 'F2', "R2 D' F'", "R D' F'", "D' F'", "R' D' F'", "R2 F'", None, "D2 F'", 'R2', 'D R', 'R', "D' R",
                     'D2 R']
    corner_index = []
    edge_setups = ["M2 D' L2", None, 'M2 D L2', 'R2 R2', 'L E L', 'E L', "L' E L", "E' L'", "M D' L2", 'E2 L', "D' L' E L", "L'", 'None', "E' L", "D2 L' E L", "E L'", "M' D L2", 'L', "M' D' L2", "E2 L'",
                   "D' L2", 'D2 L2', 'D L2', 'L2']
    edge_index = []

    for cube in range(len(cube_names)):
        if len(cube_names[cube]) == 3:
            corner_index.append(cube)
        elif len(cube_names[cube]) == 2 and cube_names[cube] != 'wr':
            edge_index.append(cube)

    edge_pieces = ['wb0', 'wr0', 'wg1', 'ow0', 'ow1', 'og1', 'oy1', 'ob1', 'wg0', 'gr0', 'gy1', 'og0', 'wr1', 'br1', 'yr1', 'gr1', 'wb1', 'ob0', 'by1', 'br0', 'gy0', 'yr0', 'b0', 'oy0']
    corner_pieces = ['owb0', 'wbr0', 'wgr0', 'owg1', 'owb2', 'owg0', 'ogy2', 'oby2', 'owg2', 'wgr1', 'gyr1', 'ogy1', 'wgr2', 'wbr2', 'byr2', 'gyr2', 'wbr1', 'owb1', 'oby1', 'byr1', 'ogy0',
                   'gyr0', 'byr0', 'oby0']

    scramble = "B' R2 B2 L2 F' R2 U2 L2 F R2 F R' F' L' D' F2 L U2 R'"
    cube_names, ori_state = run_alg(scramble, cube_names, turns, turn_names, scene, scene_rotation, 1, ori_state)
    time.sleep(0.5)
    #cube_names, ori_state = run_alg(t_perm, cube_names, turns, turn_names, scene, scene_rotation, 1, ori_state)
    #cube_names, ori_state = run_alg(reverse_alg(scramble), cube_names, turns, turn_names, scene, scene_rotation, 1, ori_state)
    #cube_names, ori_state = run_alg(t_perm, cube_names, turns, turn_names, scene, scene_rotation, 1, ori_state)
    
    print(ori_state)

    #exit()
    edges_done = False
    next_piece = None
    swaps = 0
    #solve edges
    while edges_done == False:
        if next_piece != None:
            move = next_piece
            next_piece = None
        else:
            move = edge_pieces.index(str(cube_names[19]) + str(ori_state[19]))
        alg = edge_setups[move]
        cube_names, ori_state = run_alg(alg, cube_names, turns, turn_names, scene, scene_rotation, 1, ori_state)
        cube_names, ori_state = run_alg(t_perm, cube_names, turns, turn_names, scene, scene_rotation, 1, ori_state)
        cube_names, ori_state = run_alg(reverse_alg(alg), cube_names, turns, turn_names, scene, scene_rotation, 1, ori_state)
        swaps += 1
        time.sleep(0.5)

        if cube_names[19] == 'wr':
            temp_edges_done = True
            o = 0
            for i in edge_index:
                if cube_names[i] != solved_state[i]:
                    temp_edges_done = False
                    next_piece = edge_pieces.index(str(solved_state[i]) + str(ori_state[i]))
                    break
                if ori_state[i] != 0:
                    temp_edges_done = False
                    if o != 1:
                        next_piece = edge_pieces.index(str(solved_state[i]) + str(ori_state[i]))
                        break
                o += 1

            if temp_edges_done == True:
                edges_done = True

    if (swaps % 2) == 1:
        cube_names, ori_state = run_alg(r_perm, cube_names, turns, turn_names, scene, scene_rotation, 1, ori_state)


    corners_done = False
    next_piece = None
    #solve corners
    while corners_done == False:
        if next_piece != None:
            move = next_piece
            next_piece = None
        else:
            move = corner_pieces.index(str(cube_names[2]) + str(ori_state[2]))
        alg = corner_setups[move]
        cube_names, ori_state = run_alg(alg, cube_names, turns, turn_names, scene, scene_rotation, 1, ori_state)
        cube_names, ori_state = run_alg(y_perm, cube_names, turns, turn_names, scene, scene_rotation, 1, ori_state)
        cube_names, ori_state = run_alg(reverse_alg(alg), cube_names, turns, turn_names, scene, scene_rotation, 1, ori_state)
        time.sleep(0.5)

        if cube_names[2] == 'owb':
            temp_corners_done = True
            o = 0
            for i in corner_index:
                if cube_names[i] != solved_state[i]:
                    temp_corners_done = False
                    next_piece = corner_pieces.index(str(solved_state[i]) + str(ori_state[i]))
                    break
                if ori_state[i] != 0:
                    temp_corners_done = False
                    if o != 1:
                        next_piece = corner_pieces.index(str(solved_state[i]) + str(ori_state[i]))
                        break
                o += 1

            if temp_corners_done == True:
                corners_done = True
            print(ori_state)


    print(ori_state)
    while True:
        scene.rotate(['all'], 2, 0, 0, [-50, -50, -50])
        scene.plot(camera, 600)
        screen.update()
        screen.delete('all')



        
