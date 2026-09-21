def parse_scene(scene_des):
    (environment_prompt, environment_details) = ("", "")
    environment_prompt += scene_des["query"]
    environment_details += "Floor plan:\n"
    floor_plans = scene_des["rooms"]
    room_plans = [
        room["id"] + " | [" + ", ".join(f"({p['x']}, {p['z']})" for p in room["floorPolygon"]) + "]"
        for room in floor_plans
    ]
    room_plans = "\n".join(room_plans)
    environment_details += room_plans + "\n"
    wallheight = scene_des["wall_height"]
    environment_details += f"Wallheight: {wallheight:.2f}\n"
    environment_details += "Doors:\n"
    raw_doors = scene_des["doors"]
    for raw_door in raw_doors:
        door_id = raw_door["id"] + " "
        room_0 = " " + raw_door["room0"] + " "
        room_1 = " " + raw_door["room1"] + " "
        doorSegment = raw_door["doorSegment"]
        line = f" [({doorSegment[1][0]:.2f}, {doorSegment[1][1]:.2f}), ({doorSegment[0][0]:.2f}, {doorSegment[0][1]:.2f})]"
        door_properties = "|".join([door_id, room_0, room_1, line])
        environment_details += door_properties + "\n"
    environment_details += "Windows:\n"
    windows = scene_des["windows"]
    for window in windows:
        window_id = window["id"] + " "
        room = " " + window["room0"] + " "
        windowSegment = window["windowSegment"]
        line = f" [({windowSegment[1][0]:.2f}, {windowSegment[1][1]:.2f}), ({windowSegment[0][0]:.2f}, {windowSegment[0][1]:.2f})]"
        window_properties = "|".join([window_id, room, line])
        environment_details += window_properties + "\n"
    environment_details += "Floor objects:\n"
    floor_objects = scene_des["floor_objects"]
    for floor_object in floor_objects:
        floor_object_id = floor_object["id"] + " "
        room = " " + floor_object["roomId"] + " "
        object_center = floor_object["position"]
        object_center_height = object_center["y"]
        object_vertices = floor_object["vertices"]
        x = [
            object_vertices[0][0],
            object_vertices[1][0],
            object_vertices[2][0],
            object_vertices[3][0],
        ]
        z = [
            object_vertices[0][1],
            object_vertices[1][1],
            object_vertices[2][1],
            object_vertices[3][1],
        ]
        (x_min, x_max) = (min(x) / 100, max(x) / 100)
        (z_min, z_max) = (min(z) / 100, max(z) / 100)
        (y_min, y_max) = (0, 2 * object_center_height)
        cube = (
            " "
            + f"[({x_min:.2f}, {y_min:.2f}, {z_min:.2f}), ({x_max:.2f}, {y_max:.2f}, {z_max:.2f})]"
        )
        floor_object_properties = "|".join([floor_object_id, room, cube])
        environment_details += floor_object_properties + "\n"
    environment_details += "Wall objects:\n"
    wall_objects = scene_des["wall_objects"]
    for wall_object in wall_objects:
        wall_object_id = wall_object["id"] + " "
        room = " " + wall_object["roomId"] + " "
        object_vertices = wall_object["vertices"]
        x = [
            object_vertices[0][0],
            object_vertices[1][0],
            object_vertices[2][0],
            object_vertices[3][0],
            object_vertices[4][0],
        ]
        z = [
            object_vertices[0][1],
            object_vertices[1][1],
            object_vertices[2][1],
            object_vertices[3][1],
            object_vertices[4][1],
        ]
        (x_min, x_max) = (min(x) / 100, max(x) / 100)
        (z_min, z_max) = (min(z) / 100, max(z) / 100)
        square = " " + f"[({x_min:.2f}, {z_min:.2f}), ({x_max:.2f}, {z_max:.2f})]"
        floor_object_properties = "|".join([wall_object_id, room, square])
        environment_details += floor_object_properties + "\n"
    environment_details += "Small objects:\n"
    small_objects = scene_des["small_objects"]
    for small_object in small_objects:
        small_object_id = small_object["id"] + " "
        room = " " + small_object["roomId"] + " "
        object_position = small_object["position"]
        object_position = (
            " "
            + f"[({object_position['x']:.2f}, {object_position['y']:.2f}, {object_position['z']:.2f})]"
        )
        small_objects_properties = "|".join([small_object_id, room, object_position])
        environment_details += small_objects_properties + "\n"
    return (environment_prompt, environment_details)
