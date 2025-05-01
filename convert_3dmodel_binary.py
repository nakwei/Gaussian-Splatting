import struct

def read_points(input_path):
    points = []
    with open(input_path, "rb") as f:
        while True:
            data = f.read(8)
            if not data:
                break 
            
            # point_id = struct.unpack("<Q", data)[0]

            xyz = struct.unpack("<ddd", f.read(24))

            rgb = struct.unpack("<BBB", f.read(3))

            _ = f.read(1)
            error = struct.unpack("<d", f.read(8))[0]

            track_len = int(struct.unpack("<Q", f.read(8))[0])
            f.read(8 * track_len)

            points.append((xyz, rgb, error))
    
    return points